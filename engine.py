import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
import numpy as np
from scipy.spatial import KDTree

def compute_stats(pos, vel, box_size=None):
    # pos, vel: [T, N, 2]
    # Use min-max for positions to keep them in a fixed range
    if box_size is not None:
        pos_min = np.zeros(2)
        pos_max = np.array(box_size)
    else:
        pos_min = pos.min(axis=(0, 1))
        pos_max = pos.max(axis=(0, 1))
        
    pos_range = np.maximum(pos_max - pos_min, 1e-6)
    
    vel_mean = vel.mean(axis=(0, 1))
    vel_std = vel.std(axis=(0, 1)) + 1e-6
    return {'pos_min': pos_min, 'pos_max': pos_max, 'pos_range': pos_range,
            'vel_mean': vel_mean, 'vel_std': vel_std}

def prepare_data(pos, vel, radius=1.1, stats=None, device='cpu', cache_edges=True, box_size=None):
    # pos, vel: [T, N, 2]
    T, N, _ = pos.shape

    if stats is None:
        stats = compute_stats(pos, vel, box_size=box_size)

    # Map pos to [-1, 1] and clip to prevent Tanh saturation
    pos_norm = 2.0 * (pos - stats['pos_min']) / stats['pos_range'] - 1.0
    pos_norm = np.clip(pos_norm, -0.99, 0.99)
    vel_norm = (vel - stats['vel_mean']) / stats['vel_std']

    dataset = []
    # Pre-convert to float32 for speed
    pos_norm = pos_norm.astype(np.float32)
    vel_norm = vel_norm.astype(np.float32)

    # Cache edge indices if they are consistent across time steps (for efficiency)
    cached_edges = None
    if cache_edges and T > 0:
        # Check if positions are similar enough to reuse edges (for near-equilibrium systems)
        initial_tree = KDTree(pos[0])
        initial_pairs = initial_tree.query_pairs(radius)
        if initial_pairs:
            initial_edges = np.array(list(initial_pairs), dtype=np.int64)
            cached_edges = torch.from_numpy(
                np.concatenate([initial_edges, initial_edges[:, [1, 0]]], axis=0).T
            ).to(device)

    for t in range(T):
        if cached_edges is not None:
            edge_index = cached_edges
        else:
            curr_pos = pos[t]
            tree = KDTree(curr_pos)
            # query_pairs is already optimized in cKDTree (which KDTree uses)
            pairs = tree.query_pairs(radius)

            if pairs:
                edges = np.array(list(pairs), dtype=np.int64)
                # Undirected graph: add both directions
                edge_index = np.concatenate([edges, edges[:, [1, 0]]], axis=0).T
                edge_index = torch.from_numpy(edge_index).to(device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        # Fast tensor creation
        x = torch.cat([torch.from_numpy(pos_norm[t]).to(device),
                       torch.from_numpy(vel_norm[t]).to(device)], dim=1)
        data = Data(x=x, edge_index=edge_index)
        dataset.append(data)
    return dataset, stats

def analyze_latent_space(model, dataset, pos_raw, tau=0.1, device='cpu'):
    """
    Analyzes the physical meaning of latent variables Z by correlating them
    with the Center of Mass (CoM) of assigned particles.
    """
    model.eval()
    z_list, com_list = [], []

    with torch.no_grad():
        for t, data in enumerate(dataset):
            batch = Batch.from_data_list([data]).to(device)
            z, s, _, _ = model.encode(batch.x, batch.edge_index, batch.batch, tau=tau)

            # Compute CoM for each super-node based on assignment weights s
            # s: [N, K], pos_raw[t]: [N, 2]
            s_sum = s.sum(dim=0, keepdim=True) + 1e-9
            s_norm = s / s_sum
            curr_pos = torch.tensor(pos_raw[t], dtype=torch.float, device=device)
            com = torch.matmul(s_norm.t(), curr_pos) # [K, 2]

            z_list.append(z[0].cpu().numpy())
            com_list.append(com.cpu().numpy())

    z_all = np.array(z_list)   # [T, K, D]
    com_all = np.array(com_list) # [T, K, 2]

    avg_corrs = []
    for k in range(model.encoder.n_super_nodes):
        z_k = z_all[:, k, :]
        com_k = com_all[:, k, :]
        # Correlate each Z dimension with each CoM dimension
        corrs = np.array([[np.clip(np.corrcoef(z_k[:, i], com_k[:, j])[0, 1], -1.0, 1.0)
                          for j in range(2)] for i in range(z_k.shape[1])])
        avg_corrs.append(np.nan_to_num(corrs))

    return np.array(avg_corrs)


def enhance_physical_mapping(model, dataset, pos_raw, vel_raw, tau=0.1, device='cpu'):
    """
    Enhanced analysis that correlates latent variables with both position and velocity
    to improve physical interpretability.
    """
    model.eval()
    z_list, pos_com_list, vel_com_list = [], [], []

    with torch.no_grad():
        for t, data in enumerate(dataset):
            batch = Batch.from_data_list([data]).to(device)
            z, s, _, _ = model.encode(batch.x, batch.edge_index, batch.batch, tau=tau)

            # Compute CoM for each super-node based on assignment weights s
            # s: [N, K], pos_raw[t]: [N, 2], vel_raw[t]: [N, 2]
            s_sum = s.sum(dim=0, keepdim=True) + 1e-9
            s_norm = s / s_sum
            curr_pos = torch.tensor(pos_raw[t], dtype=torch.float, device=device)
            curr_vel = torch.tensor(vel_raw[t], dtype=torch.float, device=device)

            pos_com = torch.matmul(s_norm.t(), curr_pos)  # [K, 2]
            vel_com = torch.matmul(s_norm.t(), curr_vel)  # [K, 2]

            z_list.append(z[0].cpu().numpy())
            pos_com_list.append(pos_com.cpu().numpy())
            vel_com_list.append(vel_com.cpu().numpy())

    z_all = np.array(z_list)           # [T, K, D]
    pos_com_all = np.array(pos_com_list)  # [T, K, 2]
    vel_com_all = np.array(vel_com_list)  # [T, K, 2]

    # Calculate correlations for both position and velocity
    all_corrs = []
    for k in range(model.encoder.n_super_nodes):
        z_k = z_all[:, k, :]
        pos_k = pos_com_all[:, k, :]
        vel_k = vel_com_all[:, k, :]

        # Correlate each Z dimension with position and velocity
        pos_corrs = np.array([[np.clip(np.corrcoef(z_k[:, i], pos_k[:, j])[0, 1], -1.0, 1.0)
                              for j in range(2)] for i in range(z_k.shape[1])])
        vel_corrs = np.array([[np.clip(np.corrcoef(z_k[:, i], vel_k[:, j])[0, 1], -1.0, 1.0)
                              for j in range(2)] for i in range(z_k.shape[1])])

        # Combine position and velocity correlations
        combined_corrs = np.sqrt(pos_corrs**2 + vel_corrs**2)  # Magnitude of correlation vector
        combined_corrs = np.clip(combined_corrs, 0.0, 1.0) # Clip to 1.0 as requested
        all_corrs.append(np.nan_to_num(combined_corrs))

    return np.array(all_corrs)

class LossTracker:
    """Tracks running averages and full history of loss components to help with balancing and visualization."""
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.history = {}
        self.weights = {}
        self.history_list = {}  # Store full history for plotting
        self.weights_list = {}

    def update(self, components, weights=None):
        for k, v in components.items():
            val = v.item() if hasattr(v, 'item') else v
            if k not in self.history:
                self.history[k] = val
                self.history_list[k] = [val]
            else:
                self.history[k] = self.alpha * self.history[k] + (1 - self.alpha) * val
                self.history_list[k].append(val)
        
        if weights is not None:
            for k, v in weights.items():
                val = v.item() if hasattr(v, 'item') else v
                if k not in self.weights:
                    self.weights[k] = val
                    self.weights_list[k] = [val]
                else:
                    self.weights[k] = self.alpha * self.weights[k] + (1 - self.alpha) * val
                    self.weights_list[k].append(val)

    def get_stats(self):
        # For plotting, we return the history lists
        return {**self.history_list, **{f"w_{k}": v for k, v in self.weights_list.items()}}

    def get_running_averages(self):
        return {**self.history, **{f"w_{k}": v for k, v in self.weights.items()}}

class SymbolicProxy(torch.nn.Module):
    """
    A differentiable proxy for the discovered symbolic laws.
    Allows end-to-end gradient flow from symbolic equations back to the GNN.
    """
    def __init__(self, n_super_nodes, latent_dim, equations, transformer):
        super().__init__()
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.is_hamiltonian = False
        self.dissipation_coeffs = None

        # 1. Initialize differentiable feature transformer
        from enhanced_symbolic import TorchFeatureTransformer, SymPyToTorch
        self.torch_transformer = TorchFeatureTransformer(transformer)

        # 2. Initialize differentiable symbolic modules
        self.sym_modules = torch.nn.ModuleList()
        from symbolic import gp_to_sympy

        # Calculate n_inputs as the number of features AFTER feature selection
        # The number of inputs to SymPyToTorch should match what TorchFeatureTransformer.forward() outputs
        # Since there might be inconsistencies in how the feature selection is handled,
        # we'll determine the actual number of features by running a dummy forward pass
        try:
            # Use same device and dtype as the transformer buffers
            dummy_input = torch.zeros(1, n_super_nodes * latent_dim, 
                                     dtype=self.torch_transformer.x_poly_mean.dtype, 
                                     device=self.torch_transformer.x_poly_mean.device)
            with torch.no_grad():
                dummy_output = self.torch_transformer(dummy_input)
                n_inputs = dummy_output.size(1)
                print(f"DEBUG: Determined actual output size from TorchFeatureTransformer: {n_inputs}")
        except Exception as e:
            # Fallback to using the size of feature mask if dummy pass fails
            if hasattr(self.torch_transformer, 'feature_mask') and self.torch_transformer.feature_mask.numel() > 0:
                n_inputs = self.torch_transformer.feature_mask.size(0)
                print(f"DEBUG: Using feature_mask size fallback: {n_inputs}")
            else:
                # Use the size of normalization parameters as last resort
                n_inputs = self.torch_transformer.x_poly_mean.size(0)
                print(f"DEBUG: Using normalization parameter size fallback: {n_inputs}")

        for eq in equations:
            if eq is not None:
                # Check if it's already a wrapped program with a sympy expression
                if hasattr(eq, 'sympy_expr'):
                    sympy_expr = eq.sympy_expr
                else:
                    sympy_expr = gp_to_sympy(str(eq))

                # For Hamiltonian equations, we may need to handle them differently
                if hasattr(eq, 'compute_derivatives'):
                    # This is a Hamiltonian equation that computes derivatives differently
                    self.is_hamiltonian = True
                    if hasattr(eq, 'dissipation_coeffs'):
                        self.register_buffer('dissipation', torch.from_numpy(eq.dissipation_coeffs).float())
                    self.sym_modules.append(SymPyToTorch(sympy_expr, n_inputs))
                else:
                    self.sym_modules.append(SymPyToTorch(sympy_expr, n_inputs))
            else:
                self.sym_modules.append(None)

    def forward(self, z_flat):
        # z_flat: [Batch, K * D]
        # Ensure input is float32 for MPS stability
        z_flat = z_flat.to(torch.float32)

        if self.is_hamiltonian:
            # Enable gradients for Hamiltonian derivative computation
            # We need to compute dH/dz. If we want to backprop through this (second-order),
            # we must ensure the input z_flat has requires_grad=True.
            with torch.enable_grad():
                # If input already requires grad, don't detach it to preserve the graph
                if not z_flat.requires_grad:
                    z_flat = z_flat.detach().requires_grad_(True)
                
                X_norm = self.torch_transformer(z_flat)
                
                # Hamiltonian H is the output of the first (and only) sym_module
                H_norm = self.sym_modules[0](X_norm)
                
                # Denormalize H to physical units
                H = self.torch_transformer.denormalize_y(H_norm)
                
                # For Hamiltonian symbolic proxy, we do need higher-order derivatives
                # Use create_graph=True but with retain_graph=True to avoid the error
                dH_dz = torch.autograd.grad(H.sum(), z_flat, create_graph=True, retain_graph=True)[0]
            
            # Continue with derivative computation (gradients no longer needed)
            # Split z and dH_dz into q and p
            batch_size = z_flat.size(0)
            d_sub = self.latent_dim // 2
            
            dz_dt = torch.zeros_like(z_flat)
            
            for k in range(self.n_super_nodes):
                q_start = k * self.latent_dim
                q_end = q_start + d_sub
                p_start = q_end
                p_end = p_start + d_sub
                
                # dq/dt = dH/dp
                dz_dt[:, q_start:q_end] = dH_dz[:, p_start:p_end]
                
                # dp/dt = -dH/dq - gamma*p
                # Conservative part
                dz_dt[:, p_start:p_end] = -dH_dz[:, q_start:q_end]
                
                # Dissipative part
                if hasattr(self, 'dissipation'):
                    gamma = self.dissipation[k]
                    dz_dt[:, p_start:p_end] -= gamma * z_flat[:, p_start:p_end]
            
            return dz_dt

        # 1. Differentiable feature transformation
        X_norm = self.torch_transformer(z_flat)

        # 2. Execute each symbolic equation
        y_preds = []
        for sym_mod in self.sym_modules:
            if sym_mod is not None:
                # At this point, X_norm has been transformed and masked by torch_transformer
                # The torch_transformer applies the feature mask AFTER normalization
                # So X_norm.size(1) should match the number of selected features
                # But double-check and handle mismatches gracefully
                if X_norm.size(1) != sym_mod.n_inputs:
                    # Adjust X_norm to match expected input size
                    if X_norm.size(1) > sym_mod.n_inputs:
                        X_input = X_norm[:, :sym_mod.n_inputs]
                    elif X_norm.size(1) < sym_mod.n_inputs:
                        # Pad with zeros to match expected size
                        padding = torch.zeros((X_norm.size(0), sym_mod.n_inputs - X_norm.size(1)),
                                              device=X_norm.device, dtype=X_norm.dtype)
                        X_input = torch.cat([X_norm, padding], dim=1)
                    else:
                        X_input = X_norm
                else:
                    X_input = X_norm

                y_preds.append(sym_mod(X_input))
            else:
                y_preds.append(torch.zeros(z_flat.size(0), device=z_flat.device, dtype=torch.float32))

        Y_norm_pred = torch.stack(y_preds, dim=1)

        # 3. Denormalize
        Y_pred = self.torch_transformer.denormalize_y(Y_norm_pred)

        return Y_pred.to(torch.float32)  # Ensure output is float32 for MPS stability

class Trainer:
    def __init__(self, model, lr=5e-4, device='cpu', stats=None, align_anneal_epochs=1000,
                 warmup_epochs=20, max_epochs=1000, sparsity_scheduler=None, hard_assignment_start=0.7,
                 skip_consistency_freq=2, enable_gradient_accumulation=False, grad_acc_steps=1,
                 enhanced_balancer=None, consistency_weight=1.0, spatial_weight=1.0):
        self.model = model.to(device)
        self.device = device
        self.loss_tracker = LossTracker()
        self.s_history = []
        self.max_s_history = 10
        self.align_anneal_epochs = align_anneal_epochs
        self.warmup_epochs = warmup_epochs
        self.sparsity_scheduler = sparsity_scheduler
        self.hard_assignment_start = hard_assignment_start
        self.skip_consistency_freq = skip_consistency_freq  # Skip consistency loss every N epochs
        self.enable_gradient_accumulation = enable_gradient_accumulation
        self.grad_acc_steps = grad_acc_steps
        self.enhanced_balancer = enhanced_balancer
        self.consistency_weight = consistency_weight
        self.spatial_weight = spatial_weight

        # Learnable log-scaling for alignment to ensure it stays positive and stable
        # Initialize to a value that brings the initial alignment loss below 1.0
        self.log_align_scale = torch.nn.Parameter(torch.tensor(0.0, device=device))  # exp(0) = 1.0

        # Manually re-balance initial log_vars for stability - PRIORITIZE RECONSTRUCTION
        with torch.no_grad():
            self.model.log_vars.fill_(0.0)

        # Significantly increase spatial and connectivity loss multipliers by 10x
        if hasattr(self.model.encoder.pooling, 'temporal_consistency_weight'):
            self.model.encoder.pooling.temporal_consistency_weight = 50.0 # Increased from 20.0

        # Symbolic-in-the-loop
        self.symbolic_proxy = None
        self.symbolic_weight = 0.0
        self.symbolic_confidence = 0.0
        self.min_symbolic_confidence = 0.7

        # MPS fix: torchdiffeq has issues with MPS (float64 defaults and stability)
        self.use_mps_fix = (str(device) == 'mps')
        if self.use_mps_fix:
            # Move the ODE function to CPU for stability
            self.model.ode_func.to('cpu')
            self.mps_ode_on_cpu = True
        else:
            self.mps_ode_on_cpu = False

        # Separate H_net, GNNEncoder.assign_mlp, and GNNEncoder.output_layer parameters to apply specific weight decay
        h_net_params = []
        assign_mlp_params = []
        output_layer_params = []
        other_params = []

        if hasattr(self.model.ode_func, 'H_net'):
            h_net_params = list(self.model.ode_func.H_net.parameters())

        if hasattr(self.model.encoder.pooling, 'assign_mlp'):
            assign_mlp_params = list(self.model.encoder.pooling.assign_mlp.parameters())
            
        if hasattr(self.model.encoder, 'output_layer'):
            output_layer_params = list(self.model.encoder.output_layer.parameters())

        h_ids = {id(p) for p in h_net_params}
        assign_ids = {id(p) for p in assign_mlp_params}
        output_ids = {id(p) for p in output_layer_params}
        
        other_params = [p for p in self.model.parameters() 
                       if id(p) not in h_ids and id(p) not in assign_ids 
                       and id(p) not in output_ids and id(p) != id(self.model.log_vars)]

        param_groups = [
            {'params': other_params, 'weight_decay': 1e-5},
            {'params': [self.log_align_scale], 'lr': 1e-3},
            {'params': [self.model.log_vars], 'lr': 1e-1}, # Increased from 1e-2
        ]
        if h_net_params:
            param_groups.append({'params': h_net_params, 'weight_decay': 1e-2}) # Increased from 1e-3
        if assign_mlp_params:
            param_groups.append({'params': assign_mlp_params, 'weight_decay': 1e-2}) # Penalize rapid assignment changes
        if output_layer_params:
            param_groups.append({'params': output_layer_params, 'weight_decay': 1e-2}) # Prevent high-frequency latent oscillations

        self.optimizer = optim.Adam(param_groups, lr=lr)
        self.criterion = torch.nn.MSELoss().to(device)
        self.stats = stats

    def update_symbolic_proxy(self, symbolic_proxy_or_equations, transformer=None, weight=0.1, confidence=0.0):
        """Update the symbolic proxy model with new discovered equations."""
        if hasattr(symbolic_proxy_or_equations, 'forward'):  # It's already a proxy module
            self.symbolic_proxy = symbolic_proxy_or_equations
        else:  # It's a list of equations
            self.symbolic_proxy = SymbolicProxy(
                self.model.encoder.n_super_nodes,
                self.model.encoder.latent_dim,
                symbolic_proxy_or_equations,
                transformer
            )
            # Ensure all parameters and buffers of the symbolic proxy are on the same device as the model
            self.symbolic_proxy = self.symbolic_proxy.to(self.device)

        self.symbolic_weight = weight
        self.symbolic_confidence = confidence
        print(f"Symbolic proxy updated. Weight: {weight}, Confidence: {confidence:.3f}")
        
    def _get_schedules(self, epoch, max_epochs):
        progress = epoch / max_epochs
        tau = max(0.1, 1.0 * np.exp(-epoch / 25.0))
        
        hard = False
        if progress > self.hard_assignment_start:
            hard_prob = min(1.0, (progress - self.hard_assignment_start) / (1.0 - self.hard_assignment_start))
            hard = np.random.random() < hard_prob
            
        tf_ratio = max(0.0, 0.8 * (1.0 - epoch / (0.5 * max_epochs + 1e-9)))
        entropy_weight = 0.5 + 2.0 * progress
        
        return tau, hard, tf_ratio, entropy_weight

    def _compute_initial_recon_loss(self, batch_0, tau, hard, entropy_weight, epoch, max_epochs):
        z_curr, s_0, losses_0, mu_0 = self.model.encode(batch_0.x, batch_0.edge_index, batch_0.batch, tau=tau, hard=hard, current_epoch=epoch, total_epochs=max_epochs)

        if torch.isnan(z_curr).any() or torch.isinf(z_curr).any():
            z_curr = torch.nan_to_num(z_curr, nan=0.0, posinf=1.0, neginf=-1.0)

        # Use normalized reconstruction loss for better scale consistency
        recon_0 = self.model.decode(z_curr, s_0, batch_0.batch, stats=None)
        loss_rec = self.criterion(recon_0, batch_0.x)
        
        # Enhanced assignment loss including stability terms
        loss_assign = (
            entropy_weight * losses_0['entropy'] +
            5.0 * losses_0['diversity'] + # Increased from 1.0
            5.0 * self.spatial_weight * losses_0['spatial'] + # Increased from 0.1 and multiplied by spatial_weight
            1.0 * losses_0.get('collapse_prevention', 0.0) + # Increased from 0.1
            1.0 * losses_0.get('balance', 0.0) + # Increased from 0.1
            1.0 * losses_0.get('temporal_consistency', 0.0) # Increased from 0.1
        )
        
        initial_results = {
            'z_curr': z_curr, 's_0': s_0, 'mu_0': mu_0,
            'loss_rec': loss_rec, 'loss_assign': loss_assign,
            'losses_0': losses_0, 'x_0_target': batch_0.x
        }
        return initial_results

    def _compute_hamiltonian_curv_loss(self, z_curr):
        loss_curv = torch.tensor(0.0, device=self.device)
        if self.model.hamiltonian:
            with torch.set_grad_enabled(True):
                z_in = z_curr.reshape(z_curr.size(0), -1).detach().requires_grad_(True)
                if self.mps_ode_on_cpu:
                    z_in_cpu = z_in.cpu()
                    H_vals = self.model.ode_func.H_net(z_in_cpu)
                    H = H_vals.sum()
                    dH = torch.autograd.grad(H, z_in_cpu, create_graph=True, retain_graph=True)[0]
                    if dH is not None:
                        # Add small L2 regularization on H output to prevent scaling to infinity
                        loss_curv = torch.norm(dH.to(self.device), p=2) + 0.01 * torch.mean(H_vals**2).to(self.device)
                else:
                    H_vals = self.model.ode_func.H_net(z_in)
                    H = H_vals.sum()
                    dH = torch.autograd.grad(H, z_in, create_graph=True, retain_graph=True)[0]
                    if dH is not None:
                        # Add small L2 regularization on H output to prevent scaling to infinity
                        loss_curv = torch.norm(dH, p=2) + 0.01 * torch.mean(H_vals**2)
        return loss_curv

    def _compute_latent_smoothing_loss(self, z_preds):
        """
        Compute a loss that penalizes the second derivative of latent trajectories
        to eliminate high-frequency jitter.
        """
        if z_preds.size(0) < 3:  # Need at least 3 points to compute second derivative
            return torch.tensor(0.0, device=self.device)

        # Compute first differences
        first_diff = z_preds[1:] - z_preds[:-1]  # [T-1, B, K, D]

        # Compute second differences (second derivative approximation)
        second_diff = first_diff[1:] - first_diff[:-1]  # [T-2, B, K, D]

        # Return the mean squared second difference
        return torch.mean(second_diff**2)

    def _compute_symbolic_loss(self, z_curr, is_warmup):
        loss_sym = torch.tensor(0.0, device=self.device)
        if (self.symbolic_proxy is not None and not is_warmup and
            self.symbolic_confidence >= self.min_symbolic_confidence):
            z0_flat = z_curr.view(z_curr.size(0), -1)
            if self.mps_ode_on_cpu:
                z0_flat_cpu = z0_flat.cpu()
                gnn_dz_cpu = self.model.ode_func(0, z0_flat_cpu)
                sym_dz_cpu = self.symbolic_proxy(z0_flat_cpu)
                loss_sym = torch.nn.functional.huber_loss(gnn_dz_cpu, sym_dz_cpu, delta=0.1).to(self.device)
            else:
                gnn_dz = self.model.ode_func(0, z0_flat)
                sym_dz = self.symbolic_proxy(z0_flat)
                loss_sym = torch.nn.functional.huber_loss(gnn_dz, sym_dz, delta=0.1)
        return loss_sym

    def _integrate_trajectories(self, z_curr, data_list, dt, tf_ratio, tau, hard, is_warmup, epoch):
        z_preds = [z_curr]
        seq_len = len(data_list)

        # NEW: Gradual Integration Schedule - start with 2 steps and increase to full length
        # Only apply during dynamics phase (after warmup)
        if not is_warmup:
            # Calculate how many time steps to integrate based on training progress
            # Start with 2 steps and gradually increase to full sequence length
            progress_after_warmup = max(0.0, (epoch - self.warmup_epochs) / max(1, 100))  # Over 100 epochs
            target_integration_steps = max(2, int(progress_after_warmup * seq_len))

            # MPS fix: torchdiffeq has issues with MPS. If we are on MPS,
            # we ensure the ODE integration happens on CPU.
            use_mps_fix = (str(self.device) == 'mps')

            for t in range(1, seq_len):
                if np.random.random() < tf_ratio:
                    batch_t_prev = Batch.from_data_list([data_list[t-1]]).to(self.device)
                    z_curr_forced, _, _, _ = self.model.encode(batch_t_prev.x, batch_t_prev.edge_index, batch_t_prev.batch, tau=tau, hard=hard)
                    z_curr = torch.nan_to_num(z_curr_forced)

                # NEW: Use variable integration steps based on schedule
                # If we're still in the gradual integration phase, only integrate a few steps
                if t <= target_integration_steps:
                    t_span = torch.tensor([0, dt], device=self.device, dtype=torch.float32)
                else:
                    # For steps beyond our current integration capacity, use shorter spans
                    t_span = torch.tensor([0, min(dt, dt * (target_integration_steps / seq_len))], device=self.device, dtype=torch.float32)

                try:
                    if use_mps_fix:
                        # Move to CPU for integration
                        z_curr_cpu = z_curr.cpu()
                        t_span_cpu = t_span.cpu()
                        # forward_dynamics already handles moving to ode_func's device (which is CPU)
                        z_next_seq = self.model.forward_dynamics(z_curr_cpu, t_span_cpu)
                        # Move back to MPS
                        z_curr = torch.nan_to_num(z_next_seq[1].to(self.device), nan=0.0, posinf=1.0, neginf=-1.0)
                    else:
                        z_next_seq = self.model.forward_dynamics(z_curr, t_span)
                        z_curr = torch.nan_to_num(z_next_seq[1], nan=0.0, posinf=1.0, neginf=-1.0)
                except Exception:
                    z_curr = z_curr.detach()
                z_preds.append(z_curr)
        else:
            for t in range(1, seq_len):
                batch_t = Batch.from_data_list([data_list[t]]).to(self.device)
                z_t, _, _, _ = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch, tau=tau, hard=hard)
                z_preds.append(z_t)
        return torch.stack(z_preds)

    def train_step(self, data_list, dt, epoch=0, max_epochs=2000):
        if self.sparsity_scheduler is not None:
            new_weight = self.sparsity_scheduler.step()
            if hasattr(self.model.encoder.pooling, 'set_sparsity_weight'):
                self.model.encoder.pooling.set_sparsity_weight(new_weight)

        # Make stages relative to max_epochs
        stage1_end = int(max_epochs * 0.25)
        stage2_start = stage1_end
        
        is_stage1 = epoch < stage1_end # Stage 1: Focus on reconstruction and structure
        is_stage2 = epoch >= stage2_start # Stage 2: Unfreeze ode_func with sigmoid ramp
        is_warmup = epoch < self.warmup_epochs

        # Multiplier tensor for Stage 1 masking
        loss_multipliers = torch.ones(17, device=self.device)
        if is_stage1:
            # Mask out dynamics-related losses during Stage 1
            loss_multipliers[1] = 0.0  # cons
            loss_multipliers[4] = 0.01 # l2
            loss_multipliers[5] = 0.01 # lvr
            loss_multipliers[12] = 0.0 # sym
            # Prioritize reconstruction and structural stability
            loss_multipliers[0] = 50.0 # rec
            loss_multipliers[2] = 10.0 # assign
            loss_multipliers[6] = 20.0 # align
            loss_multipliers[9] = 20.0 # conn
            loss_multipliers[16] = 50.0 # anchor

        # Two-stage training: Stage 1 freezes ODE, Stage 2 unfreezes it
        for p in self.model.ode_func.parameters(): p.requires_grad = is_stage2
        if self.symbolic_proxy is not None:
            for p in self.symbolic_proxy.parameters(): p.requires_grad = is_stage2

        if epoch > 0 and epoch % 50 == 0:
            if hasattr(self.model.encoder.pooling, 'apply_hard_revival'):
                self.model.encoder.pooling.apply_hard_revival()

        compute_consistency = (epoch >= self.warmup_epochs) and (epoch % self.skip_consistency_freq == 0)
        if self.enable_gradient_accumulation: self.optimizer.zero_grad(set_to_none=True)

        tau, hard, tf_ratio, entropy_weight = self._get_schedules(epoch, max_epochs)
        
        # OPTIMIZATION: Use pre-batched data if provided
        if isinstance(data_list, Batch):
            full_batch = data_list
            # For pre-batched data, we need to know the sequence length
            # Assuming seq_len is stored in full_batch or can be inferred
            if hasattr(full_batch, 'seq_len'):
                seq_len = full_batch.seq_len
            else:
                # Infer seq_len if possible (e.g., from num_graphs)
                seq_len = getattr(full_batch, 'num_graphs', 1)
        else:
            full_batch = Batch.from_data_list(data_list).to(self.device)
            seq_len = len(data_list)
        
        batch_size = 1 # Assuming single trajectory batch for now based on unified_train.py

        # Encode everything in one go
        z_all_target, s_all, losses_all, mu_all = self.model.encode(
            full_batch.x, full_batch.edge_index, full_batch.batch, 
            tau=tau, hard=hard, current_epoch=epoch, total_epochs=max_epochs
        )
        
        # Reshape to [T, B, K, D]
        z_all_target = z_all_target.reshape(seq_len, batch_size, self.model.encoder.n_super_nodes, -1)
        s_all = s_all.reshape(seq_len * batch_size * (full_batch.num_nodes // seq_len), -1) # s is [N_total, K]
        # mu_all is [B_total, K, 2] -> [T, B, K, 2]
        mu_all = mu_all.reshape(seq_len, batch_size, self.model.encoder.n_super_nodes, 2)

        z_curr = z_all_target[0]
        s_0 = s_all[:full_batch.num_nodes // seq_len]
        
        # Initial reconstruction loss
        recon_0 = self.model.decode(z_curr, s_0, torch.zeros_like(full_batch.batch[:full_batch.num_nodes // seq_len]), stats=None)
        loss_rec = self.criterion(recon_0, full_batch.x[:full_batch.num_nodes // seq_len])
        
        # Assignment loss for step 0
        loss_assign = (
            entropy_weight * losses_all['entropy'] +
            5.0 * losses_all['diversity'] +
            5.0 * self.spatial_weight * losses_all['spatial'] +
            1.0 * losses_all.get('collapse_prevention', 0.0) +
            1.0 * losses_all.get('balance', 0.0) +
            1.0 * losses_all.get('temporal_consistency', 0.0)
        )

        loss_curv = self._compute_hamiltonian_curv_loss(z_curr)
        loss_sym = self._compute_symbolic_loss(z_curr, is_warmup)

        # INTEGRATION OPTIMIZATION: 
        # If tf_ratio is 0 and not warmup, we can integrate the whole sequence in one odeint call
        if not is_warmup and tf_ratio == 0:
            t_span = torch.linspace(0, (seq_len - 1) * dt, seq_len, device=self.device)
            z_preds = self.model.forward_dynamics(z_curr, t_span) # [T, B, K, D]
        else:
            # Fallback to step-by-step for teacher forcing or warmup
            z_preds_list = [z_curr]
            z_step = z_curr
            for t in range(1, seq_len):
                if not is_warmup and np.random.random() < tf_ratio:
                    z_step = z_all_target[t]
                
                if not is_warmup:
                    t_span = torch.tensor([0, dt], device=self.device)
                    z_next_seq = self.model.forward_dynamics(z_step, t_span)
                    z_step = z_next_seq[1]
                else:
                    z_step = z_all_target[t]
                z_preds_list.append(z_step)
            z_preds = torch.stack(z_preds_list)

        loss_l2 = torch.mean(z_preds**2)
        
        dt_reg = max(dt, 0.01) 
        z_vel = (z_preds[1:] - z_preds[:-1]) / dt
        z_acc = (z_vel[1:] - z_vel[:-1]) / dt if len(z_vel) > 1 else torch.tensor(0.0, device=self.device)
        
        loss_lvr = torch.mean(torch.clamp(z_acc**2, 0, 100)) if len(z_vel) > 1 else torch.tensor(0.0, device=self.device)
        loss_lvr += 0.1 * torch.mean(torch.clamp(z_vel**2, 0, 100)) if len(z_vel) > 0 else torch.tensor(0.0, device=self.device)

        loss_smooth = self._compute_latent_smoothing_loss(z_preds) * (dt / 0.001)**2

        loss_pruning = losses_all['pruning']
        loss_sparsity = losses_all['sparsity']
        loss_sep = losses_all.get('separation', torch.tensor(0.0, device=self.device))
        
        # Connectivity and ortho losses for step 0
        loss_conn = 10.0 * self.model.get_connectivity_loss(s_0, full_batch.edge_index[:, full_batch.edge_index[0] < (full_batch.num_nodes // seq_len)])
        loss_ortho = self.model.get_ortho_loss(s_0)
        loss_var = self.model.get_latent_variance_loss(z_curr)
        loss_hinge = torch.mean(torch.relu(0.1 - torch.norm(z_preds, dim=-1)))

        loss_align = torch.tensor(0.0, device=self.device)
        loss_mi = torch.tensor(0.0, device=self.device)
        loss_cons = torch.tensor(0.0, device=self.device)
        loss_anchor = torch.tensor(0.0, device=self.device)

        # Batch compute all subsequent reconstruction and alignment losses
        processed_indices = [t for t in range(seq_len) if (t % 2 == 0 or t == seq_len - 1)]
        processed_steps = len(processed_indices)
        
        # OPTIMIZATION: Batch decode all processed steps
        # z_preds is [T, B, K, D]
        z_batch = z_preds[processed_indices].view(-1, self.model.encoder.n_super_nodes, self.model.encoder.latent_dim) # [processed_steps, K, D]
        
        # Prepare s_batch and x_target_batch
        nodes_per_step = full_batch.num_nodes // seq_len
        s_indices = []
        x_indices = []
        for t in processed_indices:
            s_indices.extend(range(t*nodes_per_step, (t+1)*nodes_per_step))
            x_indices.extend(range(t*nodes_per_step, (t+1)*nodes_per_step))
        
        s_batch = s_all[s_indices] # [processed_steps * N, K]
        x_target_batch = full_batch.x[x_indices] # [processed_steps * N, 4]
        
        # Create a batch index for the combined decode [0,0...0, 1,1...1, ..., P,P...P]
        decode_batch_idx = torch.arange(processed_steps, device=self.device).repeat_interleave(nodes_per_step)
        
        recon_all = self.model.decode(z_batch, s_batch, decode_batch_idx, stats=None)
        loss_rec = self.criterion(recon_all, x_target_batch)

        # Alignment weight
        rec_val = loss_rec.item()
        alignment_base_weight = 0.0
        if epoch > self.warmup_epochs:
            progress_after_warmup = min(1.0, (epoch - self.warmup_epochs) / max(1, self.align_anneal_epochs))
            alignment_base_weight = 0.05 + 0.95 * progress_after_warmup
        
        rec_quality_factor = 1.0 - min(1.0, rec_val / 0.5)
        alignment_weight = max(0.01, alignment_base_weight * (0.2 + 0.8 * rec_quality_factor))
        
        d_align = min(self.model.encoder.latent_dim // 2 if self.model.hamiltonian else 2, 2)
        huber_loss = torch.nn.SmoothL1Loss(beta=0.1)

        # Batch compute anchor and alignment if possible
        # mu_all is [T, B, K, 2]
        mu_processed = mu_all[processed_indices].view(processed_steps, self.model.encoder.n_super_nodes, 2)
        z_processed = z_preds[processed_indices].view(processed_steps, self.model.encoder.n_super_nodes, -1)
        
        loss_anchor = self.criterion(z_processed[:, :, :2], mu_processed)
        
        z_clamped = torch.clamp(z_processed[:, :, :d_align], -1.0, 1.0)
        mu_clamped = torch.clamp(mu_processed[:, :, :d_align], -1.0, 1.0)
        loss_align = 2.0 * alignment_weight * huber_loss(z_clamped * torch.exp(self.log_align_scale), mu_clamped)
        
        # MI loss is harder to batch because it involves shuffling, but we can do it
        # for all processed steps at once
        loss_mi = alignment_weight * 5.0 * self.model.get_mi_loss(
            z_processed.view(-1, self.model.encoder.n_super_nodes, self.model.encoder.latent_dim),
            mu_processed.view(-1, self.model.encoder.n_super_nodes, 2)
        )

        # Vectorized consistency and assignment stability
        if len(processed_indices) > 1:
            # Shifted indices for calculating differences [1, 2, ..., T-1] and [0, 1, ..., T-2]
            # to compute loss between consecutive PROCESSED steps
            curr_idx = processed_indices[1:]
            prev_idx = processed_indices[:-1]
            
            # Slices for s_all
            # s_all is [T*N, K]
            s_curr = s_all.view(seq_len, nodes_per_step, -1)[curr_idx]
            s_prev = s_all.view(seq_len, nodes_per_step, -1)[prev_idx]
            
            # Slices for mu_all and z_all_target
            # mu_all: [T, B, K, 2], z_all_target: [T, B, K, D]
            mu_curr = mu_all[curr_idx]
            mu_prev = mu_all[prev_idx]
            z_t_curr = z_all_target[curr_idx]
            z_t_prev = z_all_target[prev_idx]
            
            loss_assign += (
                self.criterion(s_curr, s_prev) + 
                10.0 * self.criterion(mu_curr, mu_prev) + 
                0.5 * self.criterion(z_t_curr, z_t_prev)
            )
            
            if compute_consistency and not is_warmup:
                # z_preds: [T, B, K, D]
                z_p_curr = z_preds[curr_idx]
                loss_cons = self.criterion(z_p_curr, z_t_curr)

        # Normalize losses
        loss_assign = loss_assign / processed_steps
        loss_pruning = loss_pruning / processed_steps
        loss_sparsity = loss_sparsity / processed_steps
        loss_ortho = loss_ortho / processed_steps
        loss_sep = loss_sep / processed_steps
        loss_conn = loss_conn / processed_steps
        loss_anchor = loss_anchor / processed_steps
            
        loss_align /= (processed_steps * self.model.encoder.n_super_nodes)
        loss_mi /= (processed_steps * self.model.encoder.n_super_nodes)

        lvars = torch.clamp(self.model.log_vars, min=-6.0, max=5.0)
        
        raw_losses = {
            'rec': loss_rec, 'cons': loss_cons, 'assign': loss_assign, 'ortho': loss_ortho, 
            'l2': loss_l2, 'lvr': loss_lvr, 'align': loss_align, 'pruning': loss_pruning, 
            'sep': loss_sep, 'conn': loss_conn, 'sparsity': loss_sparsity, 'mi': loss_mi, 
            'sym': loss_sym, 'var': loss_var, 'hinge': loss_hinge, 'smooth': loss_smooth, 
            'anchor': loss_anchor
        }

        # Activity penalty
        if len(z_vel) > 0:
            loss_activity = torch.relu(1.0 - torch.norm(z_vel, dim=-1)).mean()
            raw_losses['activity'] = loss_activity * 50.0
        else:
            loss_activity = self.model.get_activity_penalty(z_preds)
            raw_losses['activity'] = loss_activity * 50.0

        for k in raw_losses:
            raw_losses[k] = raw_losses[k].to(torch.float32)

        base_keys = list(raw_losses.keys())[:17]
        weights = {k: torch.exp(-lvars[i]) * loss_multipliers[i] for i, k in enumerate(base_keys)}
        if 'activity' in raw_losses: weights['activity'] = 1.0

        if self.enhanced_balancer and not is_stage1:
            balanced = self.enhanced_balancer.get_balanced_losses(raw_losses, self.model.parameters())
            for k in weights:
                if k in balanced and raw_losses[k].item() != 0: weights[k] = balanced[k] / raw_losses[k]

        stage2_factor = 0.0
        if is_stage2:
            ramp_duration = max(1, int(max_epochs * 0.1))
            progress_in_stage2 = min(1.0, (epoch - stage2_start) / ramp_duration)
            import math
            stage2_factor = 1.0 / (1 + math.exp(-12.0 * (progress_in_stage2 - 0.5)))

        discovery_loss = sum(weights[k] * torch.clamp(raw_losses[k], -100, 100) + (lvars[i] if i < len(lvars) else 0.0) for i, k in enumerate(raw_losses.keys()) if k not in ['cons', 'l2', 'lvr'])
        discovery_loss += (weights['sym'] * (self.symbolic_weight - 1.0) * torch.clamp(loss_sym.to(torch.float32), 0, 100) * stage2_factor)
        discovery_loss += 1e-4 * torch.clamp(loss_curv.to(torch.float32), 0, 100)

        loss = discovery_loss + (weights['l2'] * 1e-6 * torch.clamp(loss_l2, 0, 100) + lvars[4]) + (weights['lvr'] * 0.1 * torch.clamp(loss_lvr, 0, 100) + lvars[5])
        if compute_consistency: loss += (weights['cons'] * self.consistency_weight * torch.clamp(loss_cons.to(torch.float32), 0, 100) * stage2_factor + lvars[1])

        smooth_idx = 15
        loss += (weights['smooth'] * 10.0 * torch.clamp(loss_smooth, 0, 100) + lvars[smooth_idx])
        
        loss = torch.clamp(loss + 0.1 * torch.sum(lvars**2), 0, 1e4)
        loss = loss.to(torch.float32)
        if not torch.isfinite(loss): loss = loss_rec if torch.isfinite(loss_rec) else torch.tensor(1.0, device=self.device, requires_grad=True)

        if self.enable_gradient_accumulation:
            (loss / self.grad_acc_steps).backward()
            if ((epoch + 1) % self.grad_acc_steps == 0) or (epoch == max_epochs - 1):
                if is_stage1: torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 1.0)
                if epoch > self.warmup_epochs and stage2_factor >= 0.95: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                else: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if is_stage1: torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 1.0)
            if epoch > self.warmup_epochs and stage2_factor >= 0.95: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            else: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad() # Added to clear gradients

        self.loss_tracker.update({'total': loss.to(torch.float32), 'rec_raw': loss_rec, 'cons_raw': loss_cons, 'assign': loss_assign, 'align': loss_align, 'mi': loss_mi, 'sym': loss_sym, 'lvar_raw': loss_var, 'curv_raw': loss_curv, 'hinge_raw': loss_hinge, 'smooth_raw': loss_smooth, 'anchor_raw': loss_anchor, 'lvars_mean': lvars.mean()}, weights=weights)
        
        if epoch % 100 == 0:
            fidelity_loss = loss_rec + loss_cons
            structural_loss = loss_assign + loss_ortho + loss_pruning + loss_sep + loss_conn + loss_sparsity
            physical_loss = loss_align + loss_mi + loss_curv + loss_sym + loss_anchor
            reg_loss = loss_l2 + loss_lvr + loss_var + loss_hinge + loss_smooth + loss_activity
            
            print(f"  [Loss Meta] Fidelity: {fidelity_loss:.4f} | Structure: {structural_loss:.4f} | Physics: {physical_loss:.4f} | Reg: {reg_loss:.4f}")
            print(f"  [Loss Detail] Rec: {loss_rec:.4f} | Cons: {loss_cons:.4f} | Assign: {loss_assign:.4f} | Align: {loss_align:.4f} | Anchor: {loss_anchor:.4f} | Sym: {loss_sym:.4f}")

        return loss.item(), loss_rec.item(), loss_cons.item() if compute_consistency else 0.0

        self.loss_tracker.update({'total': loss.to(torch.float32), 'rec_raw': loss_rec, 'cons_raw': loss_cons, 'assign': loss_assign, 'align': loss_align, 'mi': loss_mi, 'sym': loss_sym, 'lvar_raw': loss_var, 'curv_raw': loss_curv, 'hinge_raw': loss_hinge, 'smooth_raw': loss_smooth, 'anchor_raw': loss_anchor, 'lvars_mean': lvars.mean()}, weights=weights)
        
        if epoch % 100 == 0:
            # Grouped loss reporting
            fidelity_loss = loss_rec + loss_cons
            structural_loss = loss_assign + loss_ortho + loss_pruning + loss_sep + loss_conn + loss_sparsity
            physical_loss = loss_align + loss_mi + loss_curv + loss_sym + loss_anchor
            reg_loss = loss_l2 + loss_lvr + loss_var + loss_hinge + loss_smooth + loss_activity
            
            print(f"  [Loss Meta] Fidelity: {fidelity_loss:.4f} | Structure: {structural_loss:.4f} | Physics: {physical_loss:.4f} | Reg: {reg_loss:.4f}")
            print(f"  [Loss Detail] Rec: {loss_rec:.4f} | Cons: {loss_cons:.4f} | Assign: {loss_assign:.4f} | Align: {loss_align:.4f} | Anchor: {loss_anchor:.4f} | Sym: {loss_sym:.4f}")

        return loss.item(), loss_rec.item(), loss_cons.item() if compute_consistency else 0.0

if __name__ == "__main__":
    n_particles = 16
    spring_dist = 1.0
    sim = SpringMassSimulator(n_particles=n_particles, spring_dist=spring_dist, dynamic_radius=1.5)
    
    # Generate 'train' data to compute stats
    train_pos, train_vel = sim.generate_trajectory(steps=100)
    stats = compute_stats(train_pos, train_vel)
    
    # Generate 'eval' data and prepare it using training stats
    eval_pos, eval_vel = sim.generate_trajectory(steps=100)
    dataset, _ = prepare_data(eval_pos, eval_vel, radius=1.5, stats=stats)
    
    model = DiscoveryEngineModel(n_particles=n_particles, n_super_nodes=4)
    trainer = Trainer(model)
    
    for epoch in range(100):
        # Pick a random starting point in the trajectory
        idx = np.random.randint(0, len(dataset) - 1)
        batch_data = [dataset[idx], dataset[idx+1]]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f} (Rec: {rec:.6f}, Cons: {cons:.6f})")
