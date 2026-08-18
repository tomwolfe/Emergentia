import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from .models import DiscoveryNet, TrajectoryScaler
from .registry import PhysicalBasisRegistry
from .utils import verify_equivalence
from .differentiable_solver import DifferentiableSimulator
from .unit_checker import UnitChecker, is_dimensionally_consistent
from .llm_priors import LLMPriorProvider, ZaiClient
from .preprocessing import AutoSmoother, TrajectorySmoother
from .physics_constraints import ConservativeForceField, InvariantLayer


# Protected functions for gplearn
def _protected_inv(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x) > 0.01, 1.0 / x, 0.0)


inv = make_function(function=_protected_inv, name="inv", arity=1)


def _protected_power(x, y):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        y_clamped = np.clip(y, -14, 14)
        abs_x = np.where(np.abs(x) < 1e-6, 1e-6, np.abs(x))
        result = np.power(abs_x, y_clamped)
        return np.where(np.isfinite(result), np.clip(result, -1e10, 1e10), 0.0)


power = make_function(function=_protected_power, name="power", arity=2)


def _protected_exp(x):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.exp(np.clip(x, -20, 20))
        return result


exp = make_function(function=_protected_exp, name="exp", arity=1)


def _protected_sqrt(x):
    return np.sqrt(np.abs(x))


sqrt = make_function(function=_protected_sqrt, name="sqrt", arity=1)


def _protected_log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x) > 1e-6, np.log(np.abs(x)), 0.0)


log = make_function(function=_protected_log, name="log", arity=1)


class DiscoveryPipeline:
    def __init__(
        self,
        mode="lj",
        potential=None,
        device="cpu",
        seed=42,
        enable_unit_checker=True,
        enable_llm_priors=False,
        enable_auto_smoother=True,
        use_conservative_field=False,
        **kwargs,
    ):
        self.mode = mode
        self.potential = potential  # Store the actual potential object
        self.device = device
        self.seed = seed
        self.model = DiscoveryNet().to(device)
        self.scaler = TrajectoryScaler(mode=mode)
        self.pairwise_dist_range = None  # Set during train_nn from actual data

        # Reliability layers
        self.enable_unit_checker = enable_unit_checker
        self.enable_llm_priors = enable_llm_priors
        self.enable_auto_smoother = enable_auto_smoother
        self.use_conservative_field = use_conservative_field

        # Wrap model with ConservativeForceField if requested
        if use_conservative_field:
            self.conservative_field = ConservativeForceField(self.model)
        else:
            self.conservative_field = None

        # Reliability layers
        self.enable_unit_checker = enable_unit_checker
        self.enable_llm_priors = enable_llm_priors
        self.enable_auto_smoother = enable_auto_smoother

        # Initialize reliability layers
        self.unit_checker = None
        self.llm_prior_provider = None
        self.auto_smoother = None

        if enable_unit_checker:
            self.unit_checker = UnitChecker(mode=mode)
            print(f"Unit-Checker initialized for mode: {mode}")

        if enable_llm_priors:
            try:
                self.llm_prior_provider = LLMPriorProvider(
                    model="glm-4.7-flash", max_candidates=10
                )
                print("LLM Prior Provider initialized (may use fallback if no API key)")
            except Exception as e:
                print(f"Warning: Could not initialize LLM Prior Provider: {e}")

        if enable_auto_smoother:
            self.auto_smoother = AutoSmoother(init_bandwidth=1.0)
            self.auto_smoother.set_noise_level(0.0)
            print("Auto-Smoother initialized with learnable bandwidth")

    def train_nn(self, p_traj, f_traj, epochs=5000, noise_std=0.0, auto_smoother=None):
        if torch.isnan(p_traj).any() or torch.isnan(f_traj).any():
            print("Warning: NaNs detected in trajectories. Clipping and filling.")
            p_traj = torch.nan_to_num(p_traj, nan=0.0)
            f_traj = torch.nan_to_num(f_traj, nan=0.0)

        if noise_std > 0 and auto_smoother is not None:
            # Use learnable Auto-Smoother
            p_traj = auto_smoother(p_traj)
            f_traj = auto_smoother(f_traj)
        elif noise_std > 0:
            # Use traditional Gaussian smoothing
            from scipy.ndimage import gaussian_filter1d

            p_np = p_traj.cpu().numpy()
            p_np = gaussian_filter1d(p_np, sigma=1.0, axis=0)
            p_traj = torch.from_numpy(p_np).to(self.device)

        # Compute pairwise distance range from actual trajectory data
        with torch.no_grad():
            p_np = p_traj.cpu().numpy()
            diff = p_np[:, :, np.newaxis, :] - p_np[:, np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            mask = ~np.eye(dists.shape[1], dtype=bool)
            valid_dists = dists[:, mask].ravel()
            valid_dists = valid_dists[valid_dists > 0.1]
            if len(valid_dists) > 0:
                r_min = max(0.3, float(np.percentile(valid_dists, 5)))
                r_max = min(10.0, float(np.percentile(valid_dists, 95)))
            else:
                r_min, r_max = 0.5, 5.0
            self.pairwise_dist_range = (r_min, r_max)
            print(f"Pairwise distance range: [{r_min:.3f}, {r_max:.3f}]")

        self.scaler.fit(p_traj, f_traj)
        p_s, f_s = self.scaler.transform(p_traj, f_traj)

        # Symmetric Log Transform for high dynamic range
        f_target = torch.sign(f_s) * torch.log1p(torch.abs(f_s))

        if torch.isnan(f_target).any():
            f_target = torch.nan_to_num(f_target, nan=0.0)

        p_s = p_s.to(self.device)
        f_target = f_target.to(self.device)

        # Train/val split for early stopping (last 20% as validation)
        split_idx = int(p_s.shape[0] * 0.8)
        p_train, f_train = p_s[:split_idx], f_target[:split_idx]
        p_val, f_val = p_s[split_idx:], f_target[split_idx:]

        base_lr = 2e-3
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=base_lr, weight_decay=1e-4
        )
        delta = 0.5 if noise_std > 0 else 0.1
        criterion = nn.HuberLoss(delta=delta)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=200, factor=0.5
        )

        warmup_epochs = 500
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        es_patience = 200

        print(f"Training NN for {self.mode} (noise_std={noise_std})...")
        for epoch in range(epochs):
            # LR Warm-up
            if epoch < warmup_epochs:
                lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            idxs = torch.randint(0, p_train.shape[0], (1024,), device=self.device)
            p_batch = p_train[idxs]
            f_batch = f_train[idxs]

            f_pred = self.conservative_field(p_batch) if self.conservative_field else self.model(p_batch)
            loss = criterion(f_pred, f_batch)

            if torch.isnan(loss):
                print(f"NaN Loss at epoch {epoch}. Stopping.")
                break

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            if epoch >= warmup_epochs:
                scheduler.step(loss)

            # Early stopping: check validation loss every 50 epochs
            if epoch % 50 == 0:
                with torch.no_grad():
                    val_pred = self.conservative_field(p_val) if self.conservative_field else self.model(p_val)
                    val_loss = criterion(val_pred, f_val).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 50
                if patience_counter >= es_patience:
                    print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.2e})")
                    break

            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.2e} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        # Restore best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return loss.item()

    def distill_symbolic(
        self,
        population_size=2000,
        generations=40,
        use_llm_priors=False,
        llm_priors=None,
    ):
        # Use data-driven range from train_nn, or fallback
        if self.pairwise_dist_range is not None:
            r_min, r_max = self.pairwise_dist_range
        else:
            r_min, r_max = 0.5, 5.0

        r_phys = np.linspace(r_min, r_max, 500).reshape(-1, 1).astype(np.float32)
        r_scaled = torch.tensor(r_phys / self.scaler.p_scale, device=self.device)

        with torch.no_grad():
            mag_scaled = self.model.predict_mag(r_scaled)
            # Inverse Symmetric Log Transform with clamping to prevent overflow
            mag_s = torch.sign(mag_scaled) * (
                torch.exp(torch.clamp(torch.abs(mag_scaled), max=20.0)) - 1
            )
            mag_phys = (mag_s * self.scaler.f_scale).cpu().numpy().ravel()

        # Clean up any remaining infinities or NaNs in mag_phys
        mag_phys = np.nan_to_num(mag_phys, nan=0.0, posinf=1e10, neginf=-1e10)

        # Build 6-feature matrix matching DiscoveryNet._get_features
        r_safe = np.clip(r_phys, 0.1, 50.0)
        X_sr = np.hstack(
            [
                r_safe,                  # X0 = r
                1.0 / r_safe,            # X1 = 1/r
                r_safe ** 2,             # X2 = r^2
                1.0 / (r_safe ** 2),     # X3 = 1/r^2
                np.exp(-r_safe),         # X4 = exp(-r)
                np.log(r_safe + 1.0),    # X5 = log(r+1)
            ]
        )

        parsimony = 0.08

        print(f"Running Symbolic Regression for {self.mode}...")

        # Apply LLM priors by adjusting GP parameters (never as data points)
        if use_llm_priors and llm_priors is not None and len(llm_priors) > 0:
            print(f"Using {len(llm_priors)} LLM priors to guide search")
            priors_str = [str(expr) for expr in llm_priors]
            print(f"LLM Priors: {priors_str}")
            # Lower parsimony and widen init_depth to let GP explore prior-like structures
            parsimony = 0.01

        est = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            function_set=("add", "sub", "mul", "div", inv, power, exp, sqrt, log),
            const_range=(-20.0, 20.0),
            parsimony_coefficient=parsimony,
            stopping_criteria=0.001,
            init_depth=(2, 6) if (use_llm_priors and llm_priors) else (2, 4),
            max_samples=0.9,
            n_jobs=-1,
            metric="mse",
            random_state=self.seed,
            verbose=1,
        )
        est.fit(X_sr, mag_phys)

        print(f"Best program: {est._program}")

        # Convert to SymPy - map all 6 features
        r = sp.Symbol("r")
        locals_dict = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: x / y,
            "inv": lambda x: 1 / x,
            "power": lambda x, y: sp.Pow(sp.Abs(x), y),
            "exp": lambda x: sp.exp(x),
        }

        expr = sp.sympify(str(est._program), locals=locals_dict)

        # Map feature indices back to symbolic expressions
        feature_subs = {
            "X0": r,
            "X1": 1 / r,
            "X2": r ** 2,
            "X3": 1 / r ** 2,
            "X4": sp.exp(-r),
            "X5": sp.log(r + 1),
        }
        for name, sym_expr in feature_subs.items():
            expr = expr.subs(sp.Symbol(name), sym_expr)

        expr = sp.simplify(expr)

        # Apply Unit-Checker if enabled
        if self.enable_unit_checker and self.unit_checker is not None:
            print("Running Unit-Checker validation...")
            is_valid, metric, signature, message = self.unit_checker.check_consistency(
                expr
            )
            print(f"Unit-Checker result: {message}")

            if not is_valid:
                print("Warning: Symbolic expression is dimensionally inconsistent!")
                return expr

        return expr

    def refine_constants(self, expr, p_traj, f_traj):
        from scipy.optimize import minimize

        # Identify numerical constants (Floats and Integers) and replace with symbols for optimization
        all_atoms = list(expr.atoms(sp.Number))
        tune_atoms = [
            a for a in all_atoms if not (isinstance(a, sp.Integer) and abs(a) <= 2)
        ]

        if not tune_atoms:
            return expr

        symbols = [sp.Symbol(f"c{i}") for i in range(len(tune_atoms))]
        param_map = {tune_atoms[i]: symbols[i] for i in range(len(tune_atoms))}
        param_expr = expr.subs(param_map)

        r_sym = sp.Symbol("r")
        func = sp.lambdify([r_sym] + symbols, param_expr, "numpy")

        # Derivatives for Jacobian
        grad_funcs = []
        for s in symbols:
            try:
                # Try to simplify and evaluate derivative
                ge = sp.diff(param_expr, s).doit()
                gf = sp.lambdify([r_sym] + symbols, ge, "numpy")
                grad_funcs.append(gf)
            except Exception:
                grad_funcs.append(None)

        p_np = p_traj.cpu().numpy()
        f_np = f_traj.cpu().numpy()

        # Precompute distances and directions
        diff = p_np[:, :, np.newaxis, :] - p_np[:, np.newaxis, :, :]  # (T, N, N, D)
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)  # (T, N, N, 1)
        dir_vec = diff / np.clip(dist, 1e-6, None)  # (T, N, N, D)

        # Expression was discovered in scaled space, so evaluate on scaled distances
        # and multiply the result by f_scale to get physical force magnitudes
        dist_scaled = dist / self.scaler.p_scale

        n_particles = p_np.shape[1]
        mask = (~np.eye(n_particles, dtype=bool))[np.newaxis, :, :, np.newaxis]

        def objective(params):
            with np.errstate(all="ignore"):
                mag = func(dist_scaled, *params)
                if not isinstance(mag, np.ndarray):
                    mag = np.full(dist_scaled.shape, mag)
                mag = np.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)
                mag = mag * self.scaler.f_scale

            pair_forces = mag * dir_vec * mask
            f_pred = np.sum(pair_forces, axis=2)
            return np.mean((f_pred - f_np) ** 2)

        def jacobian(params):
            with np.errstate(all="ignore"):
                mag = func(dist_scaled, *params)
                if not isinstance(mag, np.ndarray):
                    mag = np.full(dist_scaled.shape, mag)
                mag = np.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)
                mag = mag * self.scaler.f_scale

                f_pred = np.sum(mag * dir_vec * mask, axis=2)
                err = f_pred - f_np  # (T, N, D)

                jac = []
                for gf in grad_funcs:
                    if gf is None:
                        jac.append(0.0)
                        continue

                    d_mag = gf(dist_scaled, *params)
                    if not isinstance(d_mag, np.ndarray):
                        d_mag = np.full(dist_scaled.shape, d_mag)
                    d_mag = np.nan_to_num(d_mag, nan=0.0, posinf=1e6, neginf=-1e6)
                    d_mag = d_mag * self.scaler.f_scale

                    # dF_i/dc_k = sum_j (d_mag * dir_vec)
                    df_dc = np.sum(d_mag * dir_vec * mask, axis=2)  # (T, N, D)
                    # dJ/dc_k = 2/N * mean(err * df_dc)
                    jac.append(2.0 * np.mean(err * df_dc))

                return np.array(jac, dtype=np.float64)

        initial_guess = [float(v) for v in tune_atoms]
        res = minimize(
            objective, initial_guess, jac=jacobian, method="L-BFGS-B", tol=1e-4
        )

        final_map = {symbols[i]: res.x[i] for i in range(len(symbols))}
        return sp.simplify(param_expr.subs(final_map))

    def validate_conservativeness(self, expr):
        r = sp.Symbol("r")
        try:
            potential = sp.integrate(expr, r)
            if potential.is_constant():
                return False
            if potential.has(sp.I):
                return False
            return True
        except Exception:
            return False

    def run(
        self, sim, nn_epochs=5000, noise_std=0.0, sr_generations=40, sr_population=2000
    ):
        p_traj, f_traj = sim.generate(steps=2000, noise_std=noise_std)

        # Train/test split: 80% train, 20% test
        split_idx = int(p_traj.shape[0] * 0.8)
        train_p, train_f = p_traj[:split_idx], f_traj[:split_idx]
        test_p, test_f = p_traj[split_idx:], f_traj[split_idx:]

        # Configure Auto-Smoother with noise level
        if self.enable_auto_smoother and self.auto_smoother is not None:
            self.auto_smoother.set_from_noise_std(noise_std)
            print(
                f"Auto-Smoother configured for noise_std={noise_std}, bandwidth={self.auto_smoother.get_bandwidth():.2f}"
            )

        final_nn_loss = self.train_nn(
            train_p,
            train_f,
            epochs=nn_epochs,
            noise_std=noise_std,
            auto_smoother=self.auto_smoother,
        )

        # Generate dataset summary for LLM if needed
        llm_dataset_summary = {
            "min_force": float(np.min(train_f.cpu().numpy()))
            if isinstance(train_f, torch.Tensor)
            else np.min(train_f),
            "max_force": float(np.max(train_f.cpu().numpy()))
            if isinstance(train_f, torch.Tensor)
            else np.max(train_f),
            "noise_level": noise_std,
            "mode": self.mode,
        }

        # Generate LLM priors if enabled
        llm_priors = None
        if self.enable_llm_priors and self.llm_prior_provider is not None:
            llm_priors = self.llm_prior_provider.generate_priors_from_llm(
                llm_dataset_summary, self.mode
            )
            print(f"Generated {len(llm_priors)} LLM priors")

        discovered_expr = self.distill_symbolic(
            population_size=sr_population,
            generations=sr_generations,
            use_llm_priors=bool(llm_priors),
            llm_priors=llm_priors,
        )

        print(f"Raw discovered formula: {discovered_expr}")
        refined_expr = self.refine_constants(discovered_expr, train_p, train_f)
        print(f"Refined formula: {refined_expr}")

        # Verify equivalence on training data
        success, metrics = verify_equivalence(
            refined_expr, self.mode, potential=self.potential
        )

        # Verify on test data
        if self.potential is not None and test_p.shape[0] > 0:
            with torch.no_grad():
                # Get test pairwise distances and forces
                test_diff = test_p.cpu().numpy()[:, :, np.newaxis, :] - test_p.cpu().numpy()[:, np.newaxis, :, :]
                test_dist = np.linalg.norm(test_diff, axis=-1)
                mask = ~np.eye(test_dist.shape[1], dtype=bool)
                test_r_vals = test_dist[:, mask].ravel()
                test_r_vals = test_r_vals[test_r_vals > 0.1]
                if len(test_r_vals) > 0:
                    test_r_tensor = torch.tensor(test_r_vals, dtype=torch.float32).view(-1, 1)
                    test_y_target = self.potential.compute_force_magnitude(test_r_tensor).numpy().ravel()
                else:
                    test_r_vals, test_y_target = None, None
            if test_r_vals is not None:
                _, test_metrics = verify_equivalence(
                    refined_expr, self.mode, potential=self.potential,
                    test_r_vals=test_r_vals, test_y_target=test_y_target,
                )
                metrics["test_r2"] = test_metrics.get("test_r2", 0.0)
                metrics["test_mse"] = test_metrics.get("test_mse", 1e6)

        is_conservative = self.validate_conservativeness(refined_expr)

        return {
            "mode": self.mode,
            "nn_loss": final_nn_loss,
            "formula": str(refined_expr),
            "raw_formula": str(discovered_expr),
            "mse": metrics.get("mse", 1e6),
            "r2": metrics.get("r2", 0.0),
            "bic": metrics.get("bic", 1e6),
            "test_r2": metrics.get("test_r2", None),
            "test_mse": metrics.get("test_mse", None),
            "success": success,
            "conservative": is_conservative,
            "unit_checker_enabled": self.enable_unit_checker,
            "llm_priors_enabled": self.enable_llm_priors,
            "auto_smoother_enabled": self.enable_auto_smoother,
            "bandwidth": self.auto_smoother.get_bandwidth()
            if self.auto_smoother
            else None,
            "noise_std": noise_std,
        }

    def validate_trajectory(self, expr, sim, steps=500):
        """Validate discovered law by re-simulating and comparing trajectories."""
        from .simulator import PhysicsSim

        r_sym = sp.Symbol("r")
        force_fn = sp.lambdify(r_sym, expr, "numpy")

        class _ExprPotential:
            def compute_force_magnitude(self, dist):
                result = force_fn(dist.numpy())
                if np.isscalar(result):
                    result = np.full_like(dist.numpy(), result)
                return torch.tensor(result, dtype=torch.float32)

        # Create a new simulator with the discovered potential
        test_sim = PhysicsSim(
            n=sim.n, dim=sim.dim, potential=_ExprPotential(),
            seed=sim.seed + 1, device=sim.device,
        )
        test_traj, _ = test_sim.generate(steps=steps, noise_std=0.0)
        ground_traj, _ = sim.generate(steps=steps, noise_std=0.0)

        min_len = min(test_traj.shape[0], ground_traj.shape[0])
        pos_error = torch.mean((test_traj[:min_len] - ground_traj[:min_len]) ** 2).item()

        return {"trajectory_mape": pos_error}


class DifferentiableDiscoveryPipeline(DiscoveryPipeline):
    def train_nn(self, p_traj, f_traj, epochs=2000, noise_std=0.0):
        self.scaler.fit(p_traj, f_traj)
        p_s, f_s = self.scaler.transform(p_traj, f_traj)

        dt = 0.001

        # Infer dimensions from data
        n_particles = p_s.shape[1]
        dim = p_s.shape[2]

        # Estimate velocities (scaled)
        vel_s = p_s[1:] - p_s[:-1]
        vel_s = torch.cat([vel_s, vel_s[-1:]], dim=0)

        # Use float64 for ODE stability
        states = torch.cat(
            [p_s.view(p_s.shape[0], -1), vel_s.view(vel_s.shape[0], -1)], dim=-1
        ).to(torch.float64)

        self.model.to(torch.float64)
        simulator = DifferentiableSimulator(
            self.model, dim=dim, n_particles=n_particles
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)

        print(f"Training Differentiable NN for {self.mode}...")

        # Multi-step rollout: curriculum learning (1->5->10->20 steps)
        curriculum_schedule = [1, 5, 10, 20]

        max_idx_base = states.shape[0] - 2 - 1

        batch_size = 32

        for epoch in range(epochs):
            # Update rollout steps based on curriculum
            n_steps = 1
            for i, ce in enumerate([epochs // 4, 2 * (epochs // 4), 3 * (epochs // 4)]):
                if epoch >= ce:
                    n_steps = curriculum_schedule[i]
                else:
                    break

            # Recompute max_idx and time grid for current n_steps
            max_idx = states.shape[0] - n_steps - 1
            if max_idx < 1:
                print("Not enough timesteps for rollout. Skipping training.")
                self.model.to(torch.float32)
                return 0.0

            t = torch.linspace(0.0, dt * n_steps, n_steps + 1, dtype=torch.float32).to(self.device)

            idx = torch.randint(0, max_idx, (batch_size,))
            x0 = states[idx].to(self.device)
            target_pos = p_s[idx[0].item() + 1: idx[0].item() + n_steps + 1].to(self.device).to(torch.float64)

            try:
                # Use adaptive solver (dopri5) after checkpoint, otherwise fixed-step RK4
                use_adaptive = epoch >= epochs // 2
                if use_adaptive:
                    simulator = DifferentiableSimulator(
                        self.model, dim=dim, n_particles=n_particles
                    ).to(self.device)
                    pred_states = simulator(x0, t, method='dopri5', atol=1e-6, rtol=1e-4)
                else:
                    pred_states = simulator(x0, t)

                # Energy conservation loss: minimize Hamiltonian drift
                # Kinetic energy at start (from initial velocities in state)
                v0 = x0[:, -dim:]
                ke_0 = 0.5 * torch.sum(v0 ** 2, dim=1)

                # Position components from pred_states
                pos_dim = n_particles * dim
                pred_pos = pred_states[1:, :, :pos_dim].view(n_steps, batch_size, n_particles, dim)

                # Potential energy approximation from model's invariant layer
                # Use position components from pred_states
                pos_dim = n_particles * dim
                pred_pos_0 = pred_states[0, :, :pos_dim]
                pred_pos_end = pred_states[-1, :, :pos_dim]

                # Simplified potential energy: quadratic (harmonic approx)
                pe_0 = 0.1 * torch.sum(pred_pos_0 ** 2, dim=1)
                pe_end = 0.1 * torch.sum(pred_pos_end ** 2, dim=1)

                # Hamiltonian: H = KE + PE
                # Approximate potential energy from position magnitudes
                pe_0 = 0.1 * torch.sum(pred_pos_0 ** 2, dim=1)  # simplified harmonic
                pe_end = 0.1 * torch.sum(pred_pos_end ** 2, dim=1)  # simplified harmonic
                
                # KE at end: approximate from position differences
                v_end = (pred_pos_end[:, :dim] - pred_pos_0[:, :dim]) / (dt * n_steps)
                ke_end = 0.5 * torch.sum(v_end ** 2, dim=1)
                
                H_0 = ke_0 + pe_0
                H_end = ke_end + pe_end  # KE at end + PE at end

                # Energy conservation loss: H_end - H_0 should be ~0
                loss_energy = torch.mean((H_end - H_0) ** 2)

                # Total loss: trajectory + energy conservation
                loss = torch.mean((pred_pos - target_pos) ** 2) + 0.1 * loss_energy
            except Exception as e:
                print(f"Error during ODE integration at epoch {epoch}: {e}")
                break

            if epoch % 200 == 0:
                print(f"Epoch {epoch} | Trajectory Loss: {loss.item():.2e}")

        self.model.to(torch.float32)
        return loss.item()
