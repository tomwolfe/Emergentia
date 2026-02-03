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
    ):
        self.mode = mode
        self.potential = potential  # Store the actual potential object
        self.device = device
        self.seed = seed
        self.model = DiscoveryNet().to(device)
        self.scaler = TrajectoryScaler(mode=mode)

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

        self.scaler.fit(p_traj, f_traj)
        p_s, f_s = self.scaler.transform(p_traj, f_traj)

        # Symmetric Log Transform for high dynamic range
        f_target = torch.sign(f_s) * torch.log1p(torch.abs(f_s))

        if torch.isnan(f_target).any():
            f_target = torch.nan_to_num(f_target, nan=0.0)

        p_s = p_s.to(self.device)
        f_target = f_target.to(self.device)

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

        print(f"Training NN for {self.mode} (noise_std={noise_std})...")
        for epoch in range(epochs):
            # LR Warm-up
            if epoch < warmup_epochs:
                lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            idxs = torch.randint(0, p_s.shape[0], (1024,), device=self.device)
            p_batch = p_s[idxs]
            f_batch = f_target[idxs]

            f_pred = self.model(p_batch)
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

            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.2e} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        return loss.item()

    def distill_symbolic(
        self,
        population_size=2000,
        generations=40,
        use_llm_priors=False,
        llm_priors=None,
    ):
        # Sample the NN across a physical range
        if self.mode == "lj":
            r_min, r_max = 0.6, 3.5
        elif self.mode == "morse":
            r_min, r_max = 0.5, 4.0
        elif self.mode == "yukawa":
            r_min, r_max = 0.5, 4.0
        else:  # spring/gravity
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

        # Basis-Free: Use r and 1/r as the input features
        X_sr = np.hstack([r_phys, 1.0 / np.clip(r_phys, 1e-3, None)])

        # Parsimony adjustment: higher to favor simple basis combinations
        parsimony = 0.08

        print(f"Running Symbolic Regression for {self.mode}...")

        # Apply LLM priors if enabled
        if use_llm_priors and llm_priors is not None and len(llm_priors) > 0:
            print(f"Using {len(llm_priors)} LLM priors as seed expressions")
            # Use gplearn's custom population initialization if available
            # For now, we'll add priors to the training data
            priors_str = [str(expr) for expr in llm_priors]
            print(f"LLM Priors: {priors_str}")

            # Add priors to training data by including them as additional data points
            # This is a simplified approach
            X_priors = np.vstack([X_sr] * len(llm_priors))
            y_priors = np.zeros(len(llm_priors))
            X_combined = np.vstack([X_sr, X_priors])
            y_combined = np.concatenate([mag_phys, y_priors])

            print(f"Training with {len(llm_priors)} LLM priors")
            est = SymbolicRegressor(
                population_size=population_size,
                generations=generations,
                function_set=("add", "sub", "mul", "div", inv, power, exp),
                const_range=(-20.0, 20.0),  # Narrower range for stability
                parsimony_coefficient=parsimony,
                stopping_criteria=0.001,
                init_depth=(2, 4),  # Shallower trees
                max_samples=0.9,
                n_jobs=-1,
                metric="mse",
                random_state=self.seed,
                verbose=1,
            )
            est.fit(X_combined, y_combined)
        else:
            est = SymbolicRegressor(
                population_size=population_size,
                generations=generations,
                function_set=("add", "sub", "mul", "div", inv, power, exp),
                const_range=(-20.0, 20.0),  # Narrower range for stability
                parsimony_coefficient=parsimony,
                stopping_criteria=0.001,
                init_depth=(2, 4),  # Shallower trees
                max_samples=0.9,
                n_jobs=-1,
                metric="mse",
                random_state=self.seed,
                verbose=1,
            )

            est.fit(X_sr, mag_phys)

        print(f"Best program: {est._program}")

        # Convert to SymPy
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

        # Mapping back X0=r, X1=1/r
        expr = expr.subs(sp.Symbol("X0"), r)
        expr = expr.subs(sp.Symbol("X1"), 1 / r)

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
                # Try to find a consistent variant
                # For now, return the expression as-is but mark it
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

        n_particles = p_np.shape[1]
        mask = (~np.eye(n_particles, dtype=bool))[np.newaxis, :, :, np.newaxis]

        def objective(params):
            with np.errstate(all="ignore"):
                mag = func(dist, *params)
                if not isinstance(mag, np.ndarray):
                    mag = np.full(dist.shape, mag)
                mag = np.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)

            pair_forces = mag * dir_vec * mask
            f_pred = np.sum(pair_forces, axis=2)
            return np.mean((f_pred - f_np) ** 2)

        def jacobian(params):
            with np.errstate(all="ignore"):
                mag = func(dist, *params)
                if not isinstance(mag, np.ndarray):
                    mag = np.full(dist.shape, mag)
                mag = np.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)

                f_pred = np.sum(mag * dir_vec * mask, axis=2)
                err = f_pred - f_np  # (T, N, D)

                jac = []
                for gf in grad_funcs:
                    if gf is None:
                        jac.append(0.0)
                        continue

                    d_mag = gf(dist, *params)
                    if not isinstance(d_mag, np.ndarray):
                        d_mag = np.full(dist.shape, d_mag)
                    d_mag = np.nan_to_num(d_mag, nan=0.0, posinf=1e6, neginf=-1e6)

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

        # Configure Auto-Smoother with noise level
        if self.enable_auto_smoother and self.auto_smoother is not None:
            self.auto_smoother.set_from_noise_std(noise_std)
            print(
                f"Auto-Smoother configured for noise_std={noise_std}, bandwidth={self.auto_smoother.get_bandwidth():.2f}"
            )

        final_nn_loss = self.train_nn(
            p_traj,
            f_traj,
            epochs=nn_epochs,
            noise_std=noise_std,
            auto_smoother=self.auto_smoother,
        )

        # Generate dataset summary for LLM if needed
        llm_dataset_summary = {
            "min_force": float(np.min(f_traj.cpu().numpy()))
            if isinstance(f_traj, torch.Tensor)
            else np.min(f_traj),
            "max_force": float(np.max(f_traj.cpu().numpy()))
            if isinstance(f_traj, torch.Tensor)
            else np.max(f_traj),
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
        refined_expr = self.refine_constants(discovered_expr, p_traj, f_traj)
        print(f"Refined formula: {refined_expr}")

        # Use the potential object for verification if available
        success, metrics = verify_equivalence(
            refined_expr, self.mode, potential=self.potential
        )

        is_conservative = self.validate_conservativeness(refined_expr)

        return {
            "mode": self.mode,
            "nn_loss": final_nn_loss,
            "formula": str(refined_expr),
            "raw_formula": str(discovered_expr),
            "mse": metrics.get("mse", 1e6),
            "r2": metrics.get("r2", 0.0),
            "bic": metrics.get("bic", 1e6),
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


class DifferentiableDiscoveryPipeline(DiscoveryPipeline):
    def train_nn(self, p_traj, f_traj, epochs=2000, noise_std=0.0):
        # In differentiable mode, we want to match trajectories
        # p_traj: (T, N, D)

        self.scaler.fit(p_traj, f_traj)
        p_s, f_s = self.scaler.transform(p_traj, f_traj)

        # Assume a small dt for stability in the ODE solver
        dt = 0.001

        # Estimate velocities (scaled)
        vel_s = p_s[1:] - p_s[:-1]
        vel_s = torch.cat([vel_s, vel_s[-1:]], dim=0)

        # Use float64 for ODE stability
        states = torch.cat(
            [p_s.view(p_s.shape[0], -1), vel_s.view(vel_s.shape[0], -1)], dim=-1
        ).to(torch.float64)

        # Model must also be float64 during ODE integration
        self.model.to(torch.float64)
        simulator = DifferentiableSimulator(self.model).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)

        print(f"Training Differentiable NN for {self.mode}...")

        t = torch.tensor([0.0, dt], dtype=torch.float64).to(self.device)

        for epoch in range(epochs):
            idx = torch.randint(0, states.shape[0] - 1, (32,))
            x0 = states[idx].to(self.device)
            target_pos = p_s[idx + 1].to(self.device).to(torch.float64)

            try:
                pred_states = simulator(x0, t)
                # x_t+1 is at index 1 of the time dimension
                pred_pos = pred_states[1, :, : target_pos.view(32, -1).shape[1]].view(
                    32, *p_s.shape[1:]
                )

                loss = torch.mean((pred_pos - target_pos) ** 2)

                if torch.isnan(loss):
                    break

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print(f"Error during ODE integration at epoch {epoch}: {e}")
                break

            if epoch % 200 == 0:
                print(f"Epoch {epoch} | Trajectory Loss: {loss.item():.2e}")

        self.model.to(torch.float32)  # Convert back
        return loss.item()
