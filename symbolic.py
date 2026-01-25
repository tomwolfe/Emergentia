import numpy as np
import torch
from torch_geometric.data import Data, Batch
import sympy as sp
import re
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Sanitized Symbolic Primitives
from symbolic_utils import create_safe_functions, gp_to_sympy
safe_sqrt, safe_log, safe_div, safe_inv, safe_square, safe_inv_square, safe_inv_power, safe_inv_r6, safe_inv_r12, safe_inv_power2, safe_inv_power3, safe_inv_power4 = create_safe_functions()
from balanced_features import BalancedFeatureTransformer as FeatureTransformer

class OptimizedExpressionProgram:
    def __init__(self, expr_str, original_program=None):
        self.expr_str = expr_str
        self.original_program = original_program
        try:
            # Use the robust gp_to_sympy logic for consistency
            self.sympy_expr = gp_to_sympy(expr_str)
            self.symbols = sorted(list(self.sympy_expr.free_symbols), key=lambda s: str(s))
            self.func = sp.lambdify(self.symbols, self.sympy_expr, modules=['numpy'])
            self.length_ = len(self.sympy_expr.args) if self.sympy_expr.is_Add else 1
        except Exception as e:
            # print(f"DEBUG: OptimizedExpressionProgram failed for {expr_str}: {e}")
            self.func = None
            self.length_ = getattr(original_program, 'length_', 1) if original_program else 1

    def execute(self, X):
        if self.func:
            try:
                if X.ndim == 1: X = X.reshape(1, -1)
                args = []
                for sym in self.symbols:
                    match = re.search(r'(\d+)', str(sym))
                    idx = int(match.group(1)) if match else 0
                    args.append(X[:, idx] if idx < X.shape[1] else np.zeros(X.shape[0]))
                res = np.asarray(self.func(*args))
                if res.ndim == 0: return np.full(X.shape[0], res)
                return res.flatten()[:X.shape[0]]
            except: pass
        return self.original_program.execute(X) if self.original_program else np.zeros(X.shape[0])

    def __str__(self): return self.expr_str

class SymbolicDistiller:
    def __init__(self, populations=500, generations=10, stopping_criteria=0.001, max_features=8):
        self.populations = populations
        self.generations = generations
        self.stopping_criteria = stopping_criteria
        self.max_features = max_features
        self.transformer = None
        self.recovered_constants = {} # NEW: store recovered constants

    def get_complexity(self, program):
        """Returns the length of the program as a measure of complexity."""
        return getattr(program, 'length_', 1)

    def _get_regressor(self, pop, gen, parsimony=0.005, n_jobs=1):
        # Default to n_jobs=1 to avoid contention with outer parallel loops
        return SymbolicRegressor(population_size=pop, generations=gen, parsimony_coefficient=parsimony,
                                 function_set=('add', 'sub', 'mul', safe_div, safe_sqrt, safe_log, 'abs', 'neg', safe_inv, safe_square, safe_inv_square),
                                 max_samples=0.9, n_jobs=n_jobs, random_state=42, verbose=0)

    def _select_features(self, X, y, sim_type=None, feature_names=None, skip_mi=False, quick=False):
        # Sample data if it's too large
        if X.shape[0] > 1000:
            indices = np.random.choice(X.shape[0], 1000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y

        # 1. Fast F-test pass to narrow down candidates
        f_scores, _ = f_regression(X_sample, y_sample)
        f_scores = np.nan_to_num(f_scores)
        
        # 2. Only run slow Mutual Information on top candidates (e.g., top 50)
        # SKIP if skip_mi is True, or quick is True, or if X is very small
        if skip_mi or quick or X.shape[0] < 50:
            combined_scores = f_scores / (f_scores.max() + 1e-9)
        else:
            top_k_f = min(50, X.shape[1])
            top_f_indices = np.argsort(f_scores)[-top_k_f:]
            
            mi_scores_subset = mutual_info_regression(X_sample[:, top_f_indices], y_sample, random_state=42)
            
            # Combine scores
            combined_scores = np.zeros(X.shape[1])
            combined_scores[top_f_indices] = 0.5 * (f_scores[top_f_indices] / (f_scores[top_f_indices].max() + 1e-9)) + \
                                             0.5 * (mi_scores_subset / (mi_scores_subset.max() + 1e-9))
            
            # For those not in top_k_f, use only f_score (but scaled down)
            remaining_mask = np.ones(X.shape[1], dtype=bool)
            remaining_mask[top_f_indices] = False
            if f_scores.max() > 0:
                combined_scores[remaining_mask] = 0.3 * (f_scores[remaining_mask] / (f_scores.max() + 1e-9))
        
        mask = np.zeros(X.shape[1], dtype=bool)
        
        # We use feature_names if provided, otherwise fallback to transformer names if X matches in size
        if feature_names is None and hasattr(self, 'transformer') and hasattr(self.transformer, 'feature_names'):
            if X.shape[1] == len(self.transformer.feature_names):
                feature_names = self.transformer.feature_names

        mask[np.argsort(combined_scores)[-self.max_features:]] = True
        return mask

    def validate_stability(self, program, X_sample):
        """Simple stability check: ensure no NaNs or Infs for a sample."""
        try:
            if hasattr(X_sample, 'detach'): X_sample = X_sample.detach().cpu().numpy()
            y = program.execute(X_sample.reshape(1, -1))
            return np.all(np.isfinite(y))
        except:
            return False

    def _distill_single_target(self, i, X, y, targets_shape_1=None, latent_states_shape_1=None, is_hamiltonian=False, skip_deep_search=False, sim_type=None, quick=False):
        # Check for NaN values in X and y before processing
        if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f"    -> [Target {i}] ERROR: Input X or y contains NaN or Inf values. Returning None.")
            return None, None, 0.0

        print(f"    -> [Target {i}] Selecting top {self.max_features} features...")
        mask = self._select_features(X, y, sim_type=sim_type, quick=quick)
        X_sel = X[:, mask]

        # Display selected feature names if available
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'feature_names'):
            names = [self.transformer.feature_names[j] for j in np.where(mask)[0]]
            print(f"    -> [Target {i}] Selected features: {names}")

        print(f"    -> [Target {i}] Fitting LassoCV for baseline...")
        lasso = LassoCV(cv=5).fit(X_sel, y)
        lasso_score = lasso.score(X_sel, y)

        # PHYSICALITY GATE: Check if Lasso found a simple identity
        is_identity = False
        if lasso_score > 0:
            coeffs = np.abs(lasso.coef_)
            if np.sum(coeffs > 1e-3) == 1 and np.max(coeffs) > 0.9:
                is_identity = True

        # Soften Lasso gate: allow GP even if Lasso is very good, unless it's nearly perfect
        # Also, IMPORTANT: Use relative indices (j) for the Lasso expression because it will be executed on X_sel
        if lasso_score > 0.9999:
            print(f"    -> [Target {i}] Lasso found near-perfect fit (R2={lasso_score:.4f}). Skipping GP.")
            return OptimizedExpressionProgram(str(sp.simplify(sum(c*sp.Symbol(f'X{j}') for j, c in enumerate(lasso.coef_)) + lasso.intercept_))), mask, lasso_score

        # OPTIMIZATION: Use fastest GPU-accelerated GP if available
        try:
            from fast_gpu_symbolic import FastGPUSymbolicRegressor
            print(f"    -> [Target {i}] Starting Fast GPU-accelerated GP search (Pop: {min(self.populations, 2000)}, Gen: {min(self.generations, 30)}, Samples: 90%)...")

            # Use fast GPU-accelerated regressor with reduced parameters for speed
            est = FastGPUSymbolicRegressor(
                population_size=min(self.populations, 2000),  # Reduce for speed
                generations=min(self.generations, 30),        # Reduce for speed
                parsimony_coefficient=0.005
            ).fit(X_sel, y)
            score = est.score(X_sel, y)
            program_str = str(est._program)
            print(f"    -> [Target {i}] Fast GPU-accelerated GP R2: {score:.4f}")

        except ImportError:
            try:
                from gpu_accelerated_symbolic import GPUSymbolicRegressor
                print(f"    -> [Target {i}] Starting GPU-accelerated GP search (Pop: {min(self.populations, 2000)}, Gen: {min(self.generations, 30)}, Samples: 90%)...")

                # Use GPU-accelerated regressor with reduced parameters for speed
                est = GPUSymbolicRegressor(
                    population_size=min(self.populations, 2000),  # Reduce for speed
                    generations=min(self.generations, 30),        # Reduce for speed
                    parsimony_coefficient=0.005
                ).fit(X_sel, y)
                score = est.score(X_sel, y)
                program_str = str(est._program)
                print(f"    -> [Target {i}] GPU-accelerated GP R2: {score:.4f}")

            except ImportError:
                # Fall back to original gplearn approach
                print(f"    -> [Target {i}] Starting initial GP search (Pop: {self.populations}, Gen: {self.generations}, Samples: 90%)...")
                est = self._get_regressor(self.populations, self.generations).fit(X_sel, y)
                score = est.score(X_sel, y)
                program_str = str(est._program)
                print(f"    -> [Target {i}] Initial GP R2: {score:.4f}")

        # PHYSICALITY GATE: Trigger Deep Search if score is low or result is trivial
        is_gp_trivial = re.match(r'^X\d+$', program_str) or re.match(r'^neg\(X\d+\)$', program_str) or len(program_str) < 5

        if not skip_deep_search and ((score < 0.9) or (is_gp_trivial and score < 0.99) or is_identity):
            # Reduce deep search parameters to improve performance while maintaining quality
            p_deep = min(self.populations * 2, 5000)  # Reduced from populations * 4 to populations * 2
            g_deep = min(self.generations * 2, 100)  # Reduced from generations * 4 to generations * 2
            print(f"    -> [Target {i}] Physicality Gate triggered. Starting Deep Search (Pop: {p_deep}, Gen: {g_deep})...")

            # Try fastest GPU-accelerated GP first if available
            try:
                from fast_gpu_symbolic import FastGPUSymbolicRegressor
                print(f"    -> [Target {i}] Using Fast GPU-accelerated GP for Deep Search...")
                est = FastGPUSymbolicRegressor(
                    population_size=p_deep,
                    generations=g_deep,
                    parsimony_coefficient=0.005
                ).fit(X_sel, y)
                score = est.score(X_sel, y)
                program_str = str(est._program)
                print(f"    -> [Target {i}] Fast GPU-accelerated Deep Search R2: {score:.4f}")
            except ImportError:
                try:
                    from gpu_accelerated_symbolic import GPUSymbolicRegressor
                    print(f"    -> [Target {i}] Using GPU-accelerated GP for Deep Search...")
                    est = GPUSymbolicRegressor(
                        population_size=p_deep,
                        generations=g_deep,
                        parsimony_coefficient=0.005
                    ).fit(X_sel, y)
                    score = est.score(X_sel, y)
                    program_str = str(est._program)
                    print(f"    -> [Target {i}] GPU-accelerated Deep Search R2: {score:.4f}")
                except ImportError:
                    # Fall back to original gplearn approach
                    est = self._get_regressor(p_deep, g_deep, parsimony=0.005).fit(X_sel, y)
                    score = est.score(X_sel, y)
                    program_str = str(est._program)
                    print(f"    -> [Target {i}] CPU-based Deep Search R2: {score:.4f}")

        return OptimizedExpressionProgram(program_str), mask, score

    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None, hamiltonian=False, quick=False, sim_type=None):
        # Check for NaN values in targets before processing
        if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
            print(f"  -> ERROR: Targets contain NaN or Inf values. Cannot proceed with distillation.")
            return None

        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size, sim_type=sim_type)
        self.transformer.fit(latent_states, targets)
        X_norm = self.transformer.normalize_x(self.transformer.transform(latent_states))
        Y_norm = self.transformer.normalize_y(targets)
        
        if hamiltonian:
            # Regress a SINGLE target: the scalar Hamiltonian H
            # We assume the first column of targets is H if hamiltonian=True is passed
            # or that extract_latent_data was called with include_hamiltonian=True
            res, mask, score = self._distill_single_target(0, X_norm, Y_norm[:, 0], 1, latent_dim, is_hamiltonian=True, skip_deep_search=quick, sim_type=sim_type, quick=quick)
            return [res]
        
        # Avoid Parallel overhead for single target even if not hamiltonian
        if targets.shape[1] == 1:
            res, mask, score = self._distill_single_target(0, X_norm, Y_norm[:, 0], targets.shape[1], latent_dim, skip_deep_search=quick, sim_type=sim_type, quick=quick)
            return [res]

        results = Parallel(n_jobs=-1)(delayed(self._distill_single_target)(i, X_norm, Y_norm[:, i], targets.shape[1], latent_dim, skip_deep_search=quick, sim_type=sim_type, quick=quick) for i in range(targets.shape[1]))
        return [r[0] for r in results]

def extract_latent_data(model, dataset, dt, include_hamiltonian=False, return_both=False):
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Batch the entire dataset for fast encoding
    if isinstance(dataset, list):
        full_batch = Batch.from_data_list(dataset).to(device)
        seq_len = len(dataset)
    else:
        full_batch = dataset.to(device)
        seq_len = getattr(full_batch, 'seq_len', 1)
    
    with torch.no_grad():
        # Encode all steps at once
        z_all, _, _, _ = model.encode(full_batch.x, full_batch.edge_index, full_batch.batch)
        
        # Check for NaN in z before proceeding
        if torch.isnan(z_all).any():
            z_all = torch.nan_to_num(z_all)

        # Reshape to [T, K*D]
        z_flat = z_all.reshape(seq_len, -1)
        
        ode_device = next(model.ode_func.parameters()).device
        z_torch = z_flat.to(ode_device)
        
        h_vals = None
        dz_vals = None
        
        if include_hamiltonian and hasattr(model.ode_func, 'hamiltonian'):
            # Ensure we get a scalar H per sample: [T, 1]
            h_out = model.ode_func.hamiltonian(z_torch)
            if h_out.ndim == 1:
                h_vals = h_out.cpu().numpy().reshape(-1, 1)
            else:
                h_vals = h_out.cpu().numpy()
            
        if not include_hamiltonian or return_both:
            t_span = torch.zeros(1, device=ode_device)
            dz_vals = model.ode_func(t_span, z_torch)
            if torch.isnan(dz_vals).any():
                dz_vals = torch.nan_to_num(dz_vals)
            dz_vals = dz_vals.cpu().numpy()

        states = z_flat.cpu().numpy()
        t_states = np.linspace(0, seq_len * dt, seq_len)

    if return_both:
        return states, h_vals, dz_vals, t_states
    
    derivs = h_vals if include_hamiltonian else dz_vals
    return states, derivs, t_states

class DiscoveryOrchestrator:
    """
    Consolidates symbolic discovery logic to prevent configuration drift.
    Coordinates between data extraction, feature transformation, and various distillers.
    """
    def __init__(self, n_super_nodes, latent_dim, config=None):
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.config = config or {}
        self.use_gpu_acceleration = self.config.get('use_gpu_acceleration', False)
        
    def discover(self, model, dataset, dt, hamiltonian=False, sim_type='spring', quick=False, stats=None):
        from enhanced_symbolic import EnsembleSymbolicDistiller
        from physics_benchmark import run_benchmark
        from symmetry_checks import NoetherChecker
        
        print(f"\n[Orchestrator] Starting {'Hamiltonian' if hamiltonian else 'Standard'} Discovery...")
        
        # 1. Extract Data
        if hamiltonian:
            z_states, h_targets, dz_states, t_states = extract_latent_data(
                model, dataset, dt, include_hamiltonian=True, return_both=True
            )
            targets = h_targets
        else:
            z_states, targets, t_states = extract_latent_data(
                model, dataset, dt, include_hamiltonian=False
            )
            dz_states = targets
            
        # 2. Setup Resources
        pop_size = self.config.get('pop', 5000)
        gen_size = self.config.get('gen', 40)
        max_retries = self.config.get('max_retries', 0 if quick else 2)
        ensemble_size = 1 if quick else 3
        
        best_equations = []
        best_distiller = None
        best_bench = {}
        best_score = -float('inf')
        
        # --- SELF-CORRECTION LOOP ---
        for attempt in range(max_retries + 1):
            print(f"\n[Orchestrator] Attempt {attempt+1}/{max_retries+1}...")
            
            # Adjust parsimony and complexity based on attempt
            # Respect config if provided
            parsimony = self.config.get('parsimony', 0.01 if attempt == 0 else (0.05 if attempt == 1 else 0.1))
            max_f = self.config.get('max_features', 12 if attempt == 0 else (8 if attempt == 1 else 5))
            
            if attempt == max_retries and not quick:
                pop_size = max(pop_size, 10000)
                gen_size = max(gen_size, 50)

            # Choose distiller based on hamiltonian flag
            if hamiltonian:
                from hamiltonian_symbolic import HamiltonianSymbolicDistiller
                distiller = HamiltonianSymbolicDistiller(
                    populations=pop_size,
                    generations=gen_size,
                    max_features=max_f,
                    enforce_hamiltonian_structure=True,
                    parsimony=parsimony
                )
            else:
                distiller = EnsembleSymbolicDistiller(
                    populations=pop_size,
                    generations=gen_size,
                    ensemble_size=ensemble_size,
                    consensus_threshold=max(1, ensemble_size - 1),
                    secondary_optimization=True,
                    parsimony=parsimony,
                    max_features=max_f
                )
            
            try:
                if hamiltonian:
                    equations = distiller.distill(
                        z_states, targets, self.n_super_nodes, self.latent_dim,
                        sim_type=sim_type, model=model, quick=quick
                    )
                else:
                    equations = distiller.distill(
                        z_states, targets, self.n_super_nodes, self.latent_dim,
                        sim_type=sim_type, hamiltonian=hamiltonian
                    )

                # Check if distillation returned None
                if equations is None:
                    print(f"  [Orchestrator] Distillation returned None, skipping to next attempt")
                    continue
            except Exception as e:
                print(f"  [Orchestrator] Distillation failed: {e}")
                continue

            # Check if distillation produced valid equations
            if not equations or distiller is None:
                print(f"  [Orchestrator] Distillation produced no equations, skipping to next attempt")
                continue

            # If Hamiltonian, ensure single equation
            if hamiltonian and len(equations) > 1:
                equations = [equations[0]]

            if not best_equations:
                best_equations = equations
                best_distiller = distiller

            # 4. Calculate Symbolic R2 for basic validation
            try:
                # Create a temporary proxy for R2 calculation
                from symbolic_proxy import SymbolicProxy
                temp_proxy = SymbolicProxy(
                    self.n_super_nodes, self.latent_dim, equations, distiller.transformer,
                    hamiltonian=hamiltonian
                ).to(next(model.parameters()).device)

                device = next(temp_proxy.parameters()).device
                with torch.no_grad():
                    z_torch = torch.tensor(z_states, dtype=torch.float32, device=device)
                    dz_pred = temp_proxy(0, z_torch).cpu().numpy()

                r2s = []
                for i in range(dz_states.shape[1]):
                    y_true = dz_states[:, i]
                    y_pred = dz_pred[:, i]
                    r2 = 1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-9)
                    r2s.append(r2)
                mean_r2 = np.mean(r2s)
            except Exception as e:
                print(f"  [Orchestrator] R2 calculation failed: {e}")
                mean_r2 = 0.0
                r2s = [mean_r2]

            # 5. Run Physics Benchmark
            try:
                bench_report = run_benchmark(model, equations, r2s, distiller.transformer, stats=stats, quick=quick, sim_type=sim_type)
                
                # Symmetry Checks
                device = next(temp_proxy.parameters()).device
                n_checker = NoetherChecker(temp_proxy, self.latent_dim)
                rot_error = n_checker.check_rotational_invariance(torch.tensor(z_states[0:1], dtype=torch.float32, device=device))
                bench_report['rotational_invariance_error'] = rot_error
                
                energy_drift = bench_report.get('energy_conservation_error', 1.0)
                ood_r2 = bench_report.get('symbolic_r2_ood', 0.0)
                
                print(f"  [Orchestrator] Results: OOD R2: {ood_r2:.4f}, Energy Drift: {energy_drift:.2e}, Rot Error: {rot_error:.2e}")
                
                # SUCCESS CRITERIA - ADJUSTED FOR Lennard-Jones SYSTEM
                # For LJ system, we need both good OOD R2 and excellent energy conservation
                energy_threshold = 1e-6 if sim_type == 'lj' else 1e-5
                r2_threshold = 0.95 if sim_type == 'lj' else 0.98  # Slightly relaxed for LJ

                if energy_drift < energy_threshold and ood_r2 > r2_threshold:
                    print(f"  [Orchestrator] Criteria met! Finalizing discovery.")
                    best_equations, best_distiller, best_bench = equations, distiller, bench_report
                    break
                
                # Keep best effort if not perfect
                current_score = ood_r2 - 0.1 * np.log10(energy_drift + 1e-12)
                if current_score > best_score:
                    best_score = current_score
                    best_equations, best_distiller, best_bench = equations, distiller, bench_report
                    
            except Exception as e:
                print(f"  [Orchestrator] Benchmark failed: {e}")
                if not best_equations:
                    best_equations, best_distiller, best_bench = equations, distiller, {}

        return {
            'equations': best_equations if best_equations is not None else [],
            'distiller': best_distiller,
            'z_states': z_states,
            'targets': targets,
            'dz_states': dz_states,
            't_states': t_states,
            'bench_report': best_bench if best_bench is not None else {}
        }

