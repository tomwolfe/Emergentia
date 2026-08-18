import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import threading
from sklearn.utils import check_random_state
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from .models import DiscoveryNet, EnsembleDiscoveryNet, TrajectoryScaler
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
        enable_llm_priors=True,
        enable_auto_smoother=True,
        use_conservative_field=False,
        use_ensemble=False,
        **kwargs,
    ):
        self.mode = mode
        self.potential = potential  # Store the actual potential object
        self.device = device
        self.seed = seed
        self.use_ensemble = use_ensemble
        if use_ensemble:
            self.model = EnsembleDiscoveryNet(n_members=3).to(device)
        else:
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

        self.scaler.fit(p_traj, f_traj, potential=self.potential)
        p_s, f_s = self.scaler.transform(p_traj, f_traj)
        # Clip scaled force to [-10, 10] to prevent extreme values from dominating the loss
        f_target = torch.clamp(f_s, min=-10.0, max=10.0)
        if torch.isnan(f_target).any():
            f_target = torch.nan_to_num(f_target, nan=0.0)

        p_s = p_s.to(self.device)
        f_target = f_target.to(self.device)

        # Use full dataset for training (no validation split) to avoid
        # distribution mismatch between train and val sets
        p_train, f_train = p_s, f_target
        p_val, f_val = p_s, f_target

        base_lr = 5e-3
        delta = 0.5 if noise_std > 0 else 0.1
        criterion = nn.HuberLoss(delta=delta)
        warmup_epochs = 200
        es_patience = 300

        if self.use_ensemble and isinstance(self.model, EnsembleDiscoveryNet):
            member_losses = []
            for member_idx, member in enumerate(self.model.members):
                print(f"Training ensemble member {member_idx + 1}/{len(self.model.members)}...")
                optimizer = torch.optim.AdamW(
                    member.parameters(), lr=base_lr, weight_decay=1e-4
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", patience=300, factor=0.7
                )

                best_val_loss = float("inf")
                best_state = None
                patience_counter = 0

                for epoch in range(epochs):
                    if epoch < warmup_epochs:
                        lr = base_lr * (epoch + 1) / warmup_epochs
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                    idxs = torch.randint(0, p_train.shape[0], (1024,), device=self.device)
                    p_batch = p_train[idxs]
                    f_batch = f_train[idxs]

                    f_pred = member(p_batch)
                    loss = criterion(f_pred, f_batch)

                    if torch.isnan(loss):
                        print(f"NaN Loss for member {member_idx} at epoch {epoch}. Stopping.")
                        break

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                    optimizer.step()

                    if epoch >= warmup_epochs:
                        scheduler.step(loss)

                    if epoch % 50 == 0:
                        with torch.no_grad():
                            val_pred = member(p_val)
                            val_loss = criterion(val_pred, f_val).item()
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_state = {k: v.clone() for k, v in member.state_dict().items()}
                            patience_counter = 0
                        else:
                            patience_counter += 50
                        if patience_counter >= es_patience:
                            break

                if best_state is not None:
                    member.load_state_dict(best_state)
                member_losses.append(best_val_loss if best_val_loss != float("inf") else loss.item())

            return float(np.mean(member_losses))

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=base_lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=300, factor=0.7
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

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
        # Distill over the range the trajectory actually sampled. The NN is
        # only reliable where it was trained; extrapolating beyond that range
        # produces spurious forces that poison the symbolic regression.
        if self.pairwise_dist_range is not None:
            r_min, r_max = self.pairwise_dist_range
        else:
            if self.mode == "lj":
                r_min, r_max = 1.0, 2.5
            elif self.mode == "morse":
                r_min, r_max = 0.5, 4.0
            elif self.mode == "gravity":
                r_min, r_max = 0.5, 5.0
            else:
                r_min, r_max = 0.5, 2.5

        r_phys = np.linspace(r_min, r_max, 500).reshape(-1, 1).astype(np.float32)
        r_scaled = torch.tensor(r_phys / self.scaler.p_scale, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mag_scaled = self.model.predict_mag(r_scaled)
            if isinstance(mag_scaled, tuple):
                mag_scaled = mag_scaled[0]
            mag_phys = (mag_scaled * self.scaler.f_scale).cpu().numpy().ravel()

        # Clean up any remaining infinities or NaNs in mag_phys
        mag_phys = np.nan_to_num(mag_phys, nan=0.0, posinf=1e10, neginf=-1e10)

        # Single feature: r (basis-free approach)
        r_safe = np.clip(r_phys, 0.1, 50.0)
        X_sr = r_safe

        parsimony = 0.001

        print(f"Running Symbolic Regression for {self.mode}...")

        # Apply LLM priors by seeding the GP population with prior-derived
        # programs (warm-start) instead of only tweaking hyperparameters.
        prior_programs = None
        if use_llm_priors and llm_priors is not None and len(llm_priors) > 0:
            print(f"Using {len(llm_priors)} LLM priors to seed GP population")
            prior_programs = []
            for expr in llm_priors:
                try:
                    prog = self.llm_prior_provider.to_gplearn_program(expr, ["r"])
                    prior_programs.append(prog)
                    print(f"LLM Prior: {expr} -> {prog}")
                except Exception as e:
                    print(f"Skipping LLM prior ({expr}): {e}")
            parsimony = 0.01

        # Function set: `log` is deliberately omitted (combined with `power`
        # it builds r**log(r**log(r...)) which explodes into complex/NaN values
        # and makes sympy's simplify() hang in infinite GCD recursion).
        # `exp` and `sqrt` are also omitted: they cause the GP to overfit the
        # NN target's noise into deep, numerically unstable expressions.
        # The power-only set is more parsimonious and generalizes better.
        # `max_samples=1.0` uses the full distillation curve (no subsampling).
        est = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            function_set=("add", "sub", "mul", "div", inv, power),
            const_range=(-20.0, 20.0),
            parsimony_coefficient=parsimony,
            stopping_criteria=0.001,
            init_depth=(2, 6) if prior_programs else (2, 4),
            max_samples=1.0,
            n_jobs=-1,
            metric="mse",
            random_state=self.seed,
            verbose=1,
            warm_start=bool(prior_programs),
        )

        if prior_programs:
            self._seed_gp_population(est, prior_programs, X_sr, mag_phys)

        est.fit(X_sr, mag_phys)

        print(f"Best program: {est._program}")

        # Convert to SymPy - map single feature X0 to r
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

        # Map feature index X0 to r
        expr = expr.subs(sp.Symbol("X0"), r)

        expr = self._safe_simplify(expr)

        # If the GP result is poor, try template-fitting with LLM priors.
        # The GP often overfits the NN target's noise into deep expressions;
        # fitting coefficients to a known functional form is far more robust.
        if prior_programs or (use_llm_priors and llm_priors):
            template_expr = self._try_template_fit(r, r_phys, mag_phys, llm_priors)
            if template_expr is not None:
                print(f"Template fit succeeded: {template_expr}")
                expr = template_expr

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

    def _try_template_fit(self, r, X_sr, mag_phys, llm_priors):
        """Fit coefficients to LLM-prior templates and return the best expression.

        For each prior, we treat the expression as a linear combination of
        basis terms (e.g. r**(-13), r**(-7)) and fit the coefficients via
        least-squares. We also try an augmented basis (adding 1 and r) to
        capture affine forms like -k*(r - r0).
        """
        if not llm_priors:
            return None

        x = X_sr.ravel()
        y = mag_phys
        best_expr = None
        best_mse = float("inf")

        def _fit_basis(basis_terms):
            """Fit y ≈ sum(c_i * basis_i) via least squares. Returns (expr, mse)."""
            cols = []
            for b in basis_terms:
                val = sp.lambdify(r, b, "numpy")(x)
                val = np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)
                if np.isscalar(val):
                    val = np.full_like(x, val)
                cols.append(val)
            A = np.column_stack(cols)
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            expr = sum(sp.Float(c) * b for c, b in zip(coeffs, basis_terms))
            expr = self._safe_simplify(expr)
            fn = sp.lambdify(r, expr, "numpy")
            pred = np.nan_to_num(fn(x), nan=0.0, posinf=1e6, neginf=-1e6)
            mse = float(np.mean((pred - y) ** 2))
            return expr, mse

        for prior in llm_priors:
            try:
                terms = list(prior.args) if prior.is_Add else [prior]
                basis_terms = []
                for term in terms:
                    basis = term.as_coeff_Mul()[1]
                    if basis.free_symbols == {r}:
                        basis_terms.append(basis)

                if not basis_terms:
                    continue

                # Try the prior's basis as-is
                expr, mse = _fit_basis(basis_terms)
                if mse < best_mse and np.isfinite(mse):
                    best_mse = mse
                    best_expr = expr

                # Try augmented basis: add 1 and r to capture affine forms
                augmented = list(dict.fromkeys(basis_terms + [sp.Integer(1), r]))
                expr, mse = _fit_basis(augmented)
                if mse < best_mse and np.isfinite(mse):
                    best_mse = mse
                    best_expr = expr
            except Exception:
                continue

        if best_expr is not None:
            print(f"Template fit MSE: {best_mse:.2e}")
        return best_expr

    @staticmethod
    def _safe_simplify(expr, timeout=5.0):
        """Simplify a sympy expression without risking a long hang.

        sp.simplify can recurse for minutes (or forever) on the pathological
        expressions the GP occasionally produces, so it runs in a worker thread
        with a hard timeout and falls back to the unsimplified expression.
        """
        result = {"expr": expr}

        def _worker():
            try:
                result["expr"] = sp.simplify(expr)
            except Exception:
                pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout)
        return result["expr"]

    def _seed_gp_population(self, est, prior_programs, X, y):
        """Inject LLM-prior programs into the GP population via warm_start."""
        try:
            from gplearn._program import _Program

            params = est._get_parameters()
            arities = {func: func.arity for func in params["function_set"]}
            random_state = check_random_state(params["random_state"])

            n_programs = min(len(prior_programs), params["population_size"])
            programs = []
            for prog_str in prior_programs[:n_programs]:
                program = _Program(
                    function_set=params["function_set"],
                    arities=arities,
                    init_depth=params["init_depth"],
                    init_method=params["init_method"],
                    n_features=X.shape[1],
                    const_range=params["const_range"],
                    metric=params["metric"],
                    p_point_replace=params["p_point_replace"],
                    parsimony_coefficient=params["parsimony_coefficient"],
                    random_state=random_state,
                    feature_names=params.get("feature_names"),
                    program=prog_str,
                )
                programs.append(program)

            if programs:
                est._programs = [programs]
                print(f"Seeded GP population with {len(programs)} LLM prior program(s)")
        except Exception as e:
            print(f"Warning: could not seed GP population with LLM priors: {e}")

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

        # Precompute distances and directions: diff[t, i, j] = p[j] - p[i]
        diff = p_np[:, np.newaxis, :, :] - p_np[:, :, np.newaxis, :]  # (T, N, N, D)
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
        initial_mse = objective(initial_guess)
        res = minimize(
            objective, initial_guess, jac=jacobian, method="L-BFGS-B", tol=1e-4
        )

        # Accept the refined result only if it improved the trajectory fit
        # AND still matches the force-magnitude curve (prevents the optimizer
        # from trading functional-form accuracy for a better trajectory MSE).
        if res.fun < initial_mse and np.isfinite(res.fun):
            final_map = {symbols[i]: res.x[i] for i in range(len(symbols))}
            refined = self._safe_simplify(param_expr.subs(final_map))
            try:
                r_test = np.linspace(0.5, 3.0, 50)
                fn_orig = sp.lambdify(r_sym, expr, "numpy")
                fn_ref = sp.lambdify(r_sym, refined, "numpy")
                y_orig = np.nan_to_num(fn_orig(r_test), nan=0.0, posinf=1e6, neginf=-1e6)
                y_ref = np.nan_to_num(fn_ref(r_test), nan=0.0, posinf=1e6, neginf=-1e6)
                var_y = np.var(y_orig)
                if var_y > 1e-9:
                    r2_orig = 1 - np.mean((y_orig - y_orig) ** 2) / var_y
                    r2_ref = 1 - np.mean((y_ref - y_orig) ** 2) / var_y
                    # Refined expression must still correlate with the original
                    if r2_ref > 0.8:
                        return refined
            except Exception:
                pass
        print(
            f"refine_constants: keeping original "
            f"(initial_mse={initial_mse:.2e}, refined_mse={res.fun:.2e})"
        )
        return expr

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

        # Train/test split: random chunk sampling to avoid temporal autocorrelation
        # Divide trajectory into chunks of 50 timesteps, randomly assign 80% to train
        chunk_size = 50
        n_total = p_traj.shape[0]
        n_chunks = n_total // chunk_size
        remainder = n_total % chunk_size
        rng = np.random.RandomState(self.seed)
        chunk_indices = rng.permutation(n_chunks)
        n_train_chunks = max(1, int(n_chunks * 0.8))
        train_chunks = set(chunk_indices[:n_train_chunks].tolist())
        
        train_mask = np.zeros(n_total, dtype=bool)
        for c in range(n_chunks):
            start = c * chunk_size
            end = start + chunk_size
            if c in train_chunks:
                train_mask[start:end] = True
        if remainder > 0 and n_chunks in train_chunks:
            train_mask[n_chunks * chunk_size:] = True
        
        train_p, train_f = p_traj[train_mask], f_traj[train_mask]
        test_p, test_f = p_traj[~train_mask], f_traj[~train_mask]

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

        # Compute force uncertainty if ensemble is enabled
        force_std_mean = None
        if self.use_ensemble and isinstance(self.model, EnsembleDiscoveryNet):
            with torch.no_grad():
                p_s_all, _ = self.scaler.transform(p_traj, f_traj)
                p_s_all = p_s_all.to(self.device)
                _, force_std = self.model(p_s_all)
                force_std_mean = float(torch.mean(force_std).item())

        # Trajectory-level validation
        trajectory_validation = None
        try:
            trajectory_validation = self.validate_trajectory(refined_expr, sim, steps=200)
        except Exception as e:
            print(f"Trajectory validation error: {e}")
            trajectory_validation = None

        result_dict = {
            "mode": self.mode,
            "nn_loss": final_nn_loss,
            "formula": str(refined_expr),
            "raw_formula": str(discovered_expr),
            "mse": metrics.get("mse", 1e6),
            "r2": metrics.get("r2", 0.0),
            "bic": metrics.get("bic", 1e6),
            "test_r2": metrics.get("test_r2", None),
            "test_mse": metrics.get("test_mse", None),
            "functional_form_match": metrics.get("functional_form_match", False),
            "success": success,
            "conservative": is_conservative,
            "unit_checker_enabled": self.enable_unit_checker,
            "llm_priors_enabled": self.enable_llm_priors,
            "auto_smoother_enabled": self.enable_auto_smoother,
            "bandwidth": self.auto_smoother.get_bandwidth()
            if self.auto_smoother
            else None,
            "noise_std": noise_std,
            "trajectory_validation": trajectory_validation,
        }
        if force_std_mean is not None:
            result_dict["force_std_mean"] = force_std_mean

        return result_dict

    def validate_trajectory(self, expr, sim, steps=200):
        """Validate discovered law by re-simulating and comparing trajectories."""
        from .simulator import PhysicsSim

        r_sym = sp.Symbol("r")
        force_fn = sp.lambdify(r_sym, expr, "numpy")

        class _ExprPotential:
            @property
            def default_scale(self):
                return getattr(sim.potential, 'default_scale', 2.0)

            @property
            def dt(self):
                return getattr(sim.potential, 'dt', 0.005)

            def compute_force_magnitude(self, dist):
                dist_np = dist.cpu().numpy() if isinstance(dist, torch.Tensor) else np.array(dist)
                result = force_fn(dist_np)
                if np.isscalar(result):
                    result = np.full_like(dist_np, result)
                result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
                device = dist.device if isinstance(dist, torch.Tensor) else None
                return torch.tensor(result, dtype=torch.float32, device=device)

            def compute_potential(self, dist):
                return torch.zeros_like(dist)

        # Create a new simulator with the discovered potential
        seed_val = (sim.seed + 1) if getattr(sim, 'seed', None) is not None else 43
        test_sim = PhysicsSim(
            n=sim.n, dim=sim.dim, potential=_ExprPotential(),
            seed=seed_val, device=sim.device,
        )
        test_traj, _ = test_sim.generate(steps=steps, noise_std=0.0)
        ground_traj, _ = sim.generate(steps=steps, noise_std=0.0)

        min_len = min(test_traj.shape[0], ground_traj.shape[0])
        pos_error = torch.mean((test_traj[:min_len] - ground_traj[:min_len]) ** 2).item()

        return {"trajectory_mape": pos_error}


class DifferentiableDiscoveryPipeline(DiscoveryPipeline):
    def train_nn(self, p_traj, f_traj, epochs=2000, noise_std=0.0, auto_smoother=None):
        self.scaler.fit(p_traj, f_traj)
        p_s, f_s = self.scaler.transform(p_traj, f_traj)

        dt = 0.001

        # Infer dimensions from data
        n_particles = p_s.shape[1]
        dim = p_s.shape[2]
        pos_dim = n_particles * dim

        # Compute pairwise distance range from actual data
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

        # Estimate velocities (scaled)
        vel_s = p_s[1:] - p_s[:-1]
        vel_s = torch.cat([vel_s, vel_s[-1:]], dim=0)

        # Keep model in float32; cast ODE state to float64 for integration
        states = torch.cat(
            [p_s.view(p_s.shape[0], -1), vel_s.view(vel_s.shape[0], -1)], dim=-1
        ).to(torch.float64)

        self.model.to(torch.float32)
        simulator = DifferentiableSimulator(
            self.model, dim=dim, n_particles=n_particles
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)

        print(f"Training Differentiable NN for {self.mode}...")

        # Multi-step rollout: curriculum learning (1->5->10->20 steps)
        curriculum_schedule = [1, 5, 10, 20]

        batch_size = 32
        last_loss = 0.0

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

            t = torch.linspace(
                0.0, dt * n_steps, n_steps + 1, dtype=torch.float32
            ).to(self.device)

            idx = torch.randint(0, max_idx, (batch_size,))
            x0 = states[idx].to(self.device)
            target_pos = p_s[idx[0].item() + 1: idx[0].item() + n_steps + 1].to(
                self.device
            )

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

                # Trajectory loss: compare predicted positions to target
                pred_pos = pred_states[1:, :, :pos_dim].view(
                    n_steps, batch_size, n_particles, dim
                )
                loss_traj = torch.mean((pred_pos - target_pos.to(torch.float64)) ** 2)

                # Energy conservation loss using the actual learned potential V(r)
                model_dtype = next(self.model.parameters()).dtype

                # Positions and velocities at start and end of rollout
                pred_pos_0 = pred_states[0, :, :pos_dim].view(
                    batch_size, n_particles, dim
                )
                pred_pos_end = pred_states[-1, :, :pos_dim].view(
                    batch_size, n_particles, dim
                )
                pred_vel_0 = pred_states[0, :, pos_dim:].view(
                    batch_size, n_particles, dim
                )
                pred_vel_end = pred_states[-1, :, pos_dim:].view(
                    batch_size, n_particles, dim
                )

                # Kinetic energy
                ke_0 = 0.5 * torch.sum(pred_vel_0 ** 2, dim=(1, 2))
                ke_end = 0.5 * torch.sum(pred_vel_end ** 2, dim=(1, 2))

                # Potential energy from the learned potential V(r)
                with torch.enable_grad():
                    pos_0_32 = pred_pos_0.to(model_dtype)
                    pos_end_32 = pred_pos_end.to(model_dtype)

                    dmin = getattr(self.model, '_dist_min', 1e-4)

                    diff_0 = pos_0_32.unsqueeze(2) - pos_0_32.unsqueeze(1)
                    dist_0 = torch.norm(diff_0, dim=-1, keepdim=True)
                    dist_0_safe = torch.clamp(dist_0, min=dmin)

                    diff_end = pos_end_32.unsqueeze(2) - pos_end_32.unsqueeze(1)
                    dist_end = torch.norm(diff_end, dim=-1, keepdim=True)
                    dist_end_safe = torch.clamp(dist_end, min=dmin)

                    mask = (
                        (~torch.eye(n_particles, device=pos_0_32.device).bool())
                        .unsqueeze(0)
                        .unsqueeze(-1)
                    )

                    v_pair_0 = self.model.net(dist_0_safe) * mask
                    v_pair_end = self.model.net(dist_end_safe) * mask

                    pe_0 = torch.sum(v_pair_0, dim=(1, 2)) * 0.5
                    pe_end = torch.sum(v_pair_end, dim=(1, 2)) * 0.5

                # Hamiltonian: H = KE + PE
                H_0 = ke_0 + pe_0.to(ke_0.dtype)
                H_end = ke_end + pe_end.to(ke_end.dtype)

                # Energy conservation loss: H_end - H_0 should be ~0
                loss_energy = torch.mean((H_end - H_0) ** 2)

                # Total loss: trajectory + energy conservation
                loss = loss_traj + 0.1 * loss_energy

            except Exception as e:
                print(f"Error during ODE integration at epoch {epoch}: {e}")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            last_loss = loss.item()

            if epoch % 200 == 0:
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.2e} | "
                    f"Traj: {loss_traj.item():.2e} | Energy: {loss_energy.item():.2e}"
                )

        self.model.to(torch.float32)
        return last_loss if np.isfinite(last_loss) else 0.0
