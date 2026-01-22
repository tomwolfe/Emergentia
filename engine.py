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
        
        # 1. Initialize differentiable feature transformer
        from enhanced_symbolic import TorchFeatureTransformer, SymPyToTorch
        self.torch_transformer = TorchFeatureTransformer(transformer)
        
        # 2. Initialize differentiable symbolic modules
        self.sym_modules = torch.nn.ModuleList()
        from symbolic import gp_to_sympy
        
        n_inputs = self.torch_transformer.x_poly_mean.size(0)
        
        for eq in equations:
            if eq is not None:
                # Check if it's already a wrapped program with a sympy expression
                if hasattr(eq, 'sympy_expr'):
                    sympy_expr = eq.sympy_expr
                else:
                    sympy_expr = gp_to_sympy(str(eq))
                
                self.sym_modules.append(SymPyToTorch(sympy_expr, n_inputs))
            else:
                self.sym_modules.append(None)
        
    def forward(self, z_flat):
        # z_flat: [Batch, K * D]
        # 1. Differentiable feature transformation
        X_norm = self.torch_transformer(z_flat)
        
        # 2. Execute each symbolic equation
        y_preds = []
        for sym_mod in self.sym_modules:
            if sym_mod is not None:
                y_preds.append(sym_mod(X_norm))
            else:
                y_preds.append(torch.zeros(z_flat.size(0), device=z_flat.device))
        
        Y_norm_pred = torch.stack(y_preds, dim=1)
        
        # 3. Denormalize
        Y_pred = self.torch_transformer.denormalize_y(Y_norm_pred)
        
        return Y_pred

class Trainer:
    def __init__(self, model, lr=5e-4, device='cpu', stats=None, align_anneal_epochs=1000,
                 warmup_epochs=20, max_epochs=1000, sparsity_scheduler=None, hard_assignment_start=0.7,
                 skip_consistency_freq=2, enable_gradient_accumulation=False, grad_acc_steps=1,
                 enhanced_balancer=None):
        self.model = model.to(device)
        self.device = device
        self.loss_tracker = LossTracker()
        self.s_history = []
        self.max_s_history = 10
        self.align_anneal_epochs = align_anneal_epochs
        # Cap warmup_epochs to 20% of max_epochs
        self.warmup_epochs = min(warmup_epochs, int(max_epochs * 0.2))
        self.sparsity_scheduler = sparsity_scheduler
        self.hard_assignment_start = hard_assignment_start
        self.skip_consistency_freq = skip_consistency_freq  # Skip consistency loss every N epochs
        self.enable_gradient_accumulation = enable_gradient_accumulation
        self.grad_acc_steps = grad_acc_steps
        self.enhanced_balancer = enhanced_balancer
        
        # Learnable scaling for alignment to prevent scale mismatch
        self.align_scale = torch.nn.Parameter(torch.tensor(1.0, device=device))

        # Manually re-balance initial log_vars for stability
        with torch.no_grad():
            self.model.log_vars[0].fill_(-5.0)  # Very high priority for reconstruction
            self.model.log_vars[1].fill_(1.0)   # Higher priority for consistency (increased from 2.0)
            self.model.log_vars[2].fill_(1.0)   # Higher priority for assignment (increased from 2.0)
            self.model.log_vars[3].fill_(4.0)   # Lower initial priority for ortho
            self.model.log_vars[4].fill_(4.0)   # Lower initial priority for l2
            self.model.log_vars[5].fill_(2.0)   # Moderate priority for lvr
            self.model.log_vars[6].fill_(6.0)   # Lower initial priority for alignment (decreased priority)
            self.model.log_vars[7].fill_(4.0)   # Lower initial priority for pruning
            self.model.log_vars[8].fill_(4.0)   # Lower initial priority for sep
            self.model.log_vars[9].fill_(4.0)   # Lower initial priority for conn
            self.model.log_vars[10].fill_(4.0)  # Lower initial priority for sparsity
            self.model.log_vars[11].fill_(5.0)  # Lower initial priority for mi (increased priority from 6.0)
            self.model.log_vars[12].fill_(4.0)  # Lower initial priority for sym
            self.model.log_vars[13].fill_(4.0)  # Lower initial priority for var
            self.model.log_vars[14].fill_(2.0)  # Moderate priority for hinge to prevent shrinkage

        # Significantly increase temporal consistency weight
        if hasattr(self.model.encoder.pooling, 'temporal_consistency_weight'):
            self.model.encoder.pooling.temporal_consistency_weight = 10.0 # Increased from 5.0

        # Symbolic-in-the-loop
        self.symbolic_proxy = None
        self.symbolic_weight = 0.0
        self.symbolic_confidence = 0.0
        self.min_symbolic_confidence = 0.7

        # MPS fix: torchdiffeq has issues with MPS (float64 defaults and stability)
        # But we need to ensure all components are on the same device
        if str(device) == 'mps':
            # Move the ODE function to CPU but keep track of the device for other operations
            self.model.ode_func.to('cpu')
            self.mps_ode_on_cpu = True
        else:
            self.mps_ode_on_cpu = False

        # Separate H_net and GNNEncoder.assign_mlp parameters to apply specific weight decay
        h_net_params = []
        assign_mlp_params = []
        other_params = []
        
        if hasattr(self.model.ode_func, 'H_net'):
            h_net_params = list(self.model.ode_func.H_net.parameters())
        
        if hasattr(self.model.encoder.pooling, 'assign_mlp'):
            assign_mlp_params = list(self.model.encoder.pooling.assign_mlp.parameters())

        h_ids = {id(p) for p in h_net_params}
        assign_ids = {id(p) for p in assign_mlp_params}
        other_params = [p for p in self.model.parameters() if id(p) not in h_ids and id(p) not in assign_ids]

        param_groups = [
            {'params': other_params, 'weight_decay': 1e-5},
            {'params': [self.align_scale], 'lr': 1e-3}, # Specific learning rate for scale
        ]
        if h_net_params:
            param_groups.append({'params': h_net_params, 'weight_decay': 1e-3})
        if assign_mlp_params:
            param_groups.append({'params': assign_mlp_params, 'weight_decay': 1e-2}) # Penalize rapid assignment changes

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

    def _compute_initial_recon_loss(self, batch_0, tau, hard, entropy_weight):
        z_curr, s_0, losses_0, mu_0 = self.model.encode(batch_0.x, batch_0.edge_index, batch_0.batch, tau=tau, hard=hard)

        if torch.isnan(z_curr).any() or torch.isinf(z_curr).any():
            z_curr = torch.nan_to_num(z_curr, nan=0.0, posinf=1.0, neginf=-1.0)

        # Use normalized reconstruction loss for better scale consistency
        recon_0 = self.model.decode(z_curr, s_0, batch_0.batch, stats=None)
        loss_rec = self.criterion(recon_0, batch_0.x)
        
        # Enhanced assignment loss including stability terms
        loss_assign = (
            entropy_weight * losses_0['entropy'] + 
            5.0 * losses_0['diversity'] + # Increased from 2.0
            0.1 * losses_0['spatial'] +
            1.0 * losses_0.get('collapse_prevention', 0.0) + # Increased from 0.1
            1.0 * losses_0.get('balance', 0.0) + # Increased from 0.1
            2.0 * losses_0.get('temporal_consistency', 0.0) # Increased from 1.0
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
                z_in = z_curr.view(z_curr.size(0), -1).detach().requires_grad_(True)
                if self.mps_ode_on_cpu:
                    z_in_cpu = z_in.cpu()
                    H = self.model.ode_func.H_net(z_in_cpu).sum()
                    dH = torch.autograd.grad(H, z_in_cpu, create_graph=True)[0]
                    if dH is not None:
                        loss_curv = torch.norm(dH.to(self.device), p=2)
                else:
                    H = self.model.ode_func.H_net(z_in).sum()
                    dH = torch.autograd.grad(H, z_in, create_graph=True)[0]
                    if dH is not None:
                        loss_curv = torch.norm(dH, p=2)
        return loss_curv

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

    def _integrate_trajectories(self, z_curr, data_list, dt, tf_ratio, tau, hard, is_warmup):
        z_preds = [z_curr]
        seq_len = len(data_list)
        
        # MPS fix: torchdiffeq has issues with MPS. If we are on MPS,
        # we ensure the ODE integration happens on CPU.
        use_mps_fix = (str(self.device) == 'mps')
        
        if not is_warmup:
            for t in range(1, seq_len):
                if np.random.random() < tf_ratio:
                    batch_t_prev = Batch.from_data_list([data_list[t-1]]).to(self.device)
                    z_curr_forced, _, _, _ = self.model.encode(batch_t_prev.x, batch_t_prev.edge_index, batch_t_prev.batch, tau=tau, hard=hard)
                    z_curr = torch.nan_to_num(z_curr_forced)
                
                t_span = torch.tensor([0, dt], device=self.device, dtype=torch.float32)
                
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

        is_warmup = epoch < self.warmup_epochs
        if is_warmup:
            with torch.no_grad():
                for idx, val in [(0, -8.0), (6, 5.0), (3, 5.0), (11, 5.0), (1, 5.0), (2, 5.0), (7, 5.0), (10, 5.0)]:
                    self.model.log_vars[idx].fill_(val)

        is_stage1 = is_warmup
        for p in self.model.ode_func.parameters(): p.requires_grad = not is_stage1
        if self.symbolic_proxy is not None:
            for p in self.symbolic_proxy.parameters(): p.requires_grad = not is_stage1

        if epoch > 0 and epoch % 100 == 0:
            if hasattr(self.model.encoder.pooling, 'apply_hard_revival'):
                self.model.encoder.pooling.apply_hard_revival()

        compute_consistency = (epoch >= self.warmup_epochs) and (not is_stage1) and (epoch % self.skip_consistency_freq == 0)
        if self.enable_gradient_accumulation: self.optimizer.zero_grad(set_to_none=True)

        tau, hard, tf_ratio, entropy_weight = self._get_schedules(epoch, max_epochs)
        batch_0 = Batch.from_data_list([data_list[0]]).to(self.device)
        
        init_res = self._compute_initial_recon_loss(batch_0, tau, hard, entropy_weight)
        z_curr, s_0, mu_0 = init_res['z_curr'], init_res['s_0'], init_res['mu_0']
        loss_rec, loss_assign = init_res['loss_rec'], init_res['loss_assign']

        loss_curv = self._compute_hamiltonian_curv_loss(z_curr)
        loss_sym = self._compute_symbolic_loss(z_curr, is_warmup)
        
        z_preds = self._integrate_trajectories(z_curr, data_list, dt, tf_ratio, tau, hard, is_warmup)
        
        loss_l2 = torch.mean(z_preds**2)
        z_vel = (z_preds[1:] - z_preds[:-1]) / dt
        loss_lvr = torch.mean((z_vel[1:] - z_vel[:-1])**2) if len(z_vel) > 1 else torch.tensor(0.0, device=self.device)
        loss_lvr += 0.1 * torch.mean(z_vel**2) if len(z_vel) > 0 else torch.tensor(0.0, device=self.device)

        loss_pruning = init_res['losses_0']['pruning']
        loss_sparsity = init_res['losses_0']['sparsity']
        loss_sep = init_res['losses_0'].get('separation', torch.tensor(0.0, device=self.device))
        loss_conn = self.model.get_connectivity_loss(s_0, batch_0.edge_index)
        loss_ortho = self.model.get_ortho_loss(s_0)
        loss_var = self.model.get_latent_variance_loss(z_curr)
        
        # Hinge loss to prevent latent variable shrinkage (force them to have some minimum magnitude)
        # Calculates norm for each [Batch, K] and applies hinge at 0.1
        loss_hinge = torch.mean(torch.relu(0.1 - torch.norm(z_preds, dim=-1)))
        
        loss_align = torch.tensor(0.0, device=self.device)
        loss_mi = torch.tensor(0.0, device=self.device)
        loss_cons = torch.tensor(0.0, device=self.device)

        s_prev, mu_prev, z_enc_prev = s_0, mu_0, z_curr
        mu_min = torch.tensor(self.stats['pos_min'], device=self.device, dtype=torch.float32) if self.stats else 0
        mu_range = torch.tensor(self.stats['pos_range'], device=self.device, dtype=torch.float32) if self.stats else 1
        seq_len = len(data_list)

        for t in range(0, seq_len, 2):
            batch_t = Batch.from_data_list([data_list[t]]).to(self.device)
            z_t_target, s_t, losses_t, mu_t = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch, tau=tau, hard=hard)
            z_t_target = torch.nan_to_num(z_t_target)
            
            # Use normalized targets for stability
            recon_t = self.model.decode(z_preds[t], s_t, batch_t.batch, stats=None)
            loss_rec += self.criterion(recon_t, batch_t.x)
            
            loss_assign += (
                entropy_weight * losses_t['entropy'] + 
                5.0 * losses_t['diversity'] + 
                0.1 * losses_t['spatial'] +
                1.0 * losses_t.get('collapse_prevention', 0.0) + 
                1.0 * losses_t.get('balance', 0.0) +
                2.0 * losses_t.get('temporal_consistency', 0.0)
            )
            loss_pruning += losses_t['pruning']
            loss_sparsity += losses_t['sparsity']
            loss_sep += losses_t.get('separation', torch.tensor(0.0, device=self.device))
            loss_conn += self.model.get_connectivity_loss(s_t, batch_t.edge_index)
            loss_ortho += self.model.get_ortho_loss(s_t)

            # mu_t is already in normalized range [-1, 1] because it's computed from normalized x
            mu_t_norm = mu_t 
            d_align = min(self.model.encoder.latent_dim // 2 if self.model.hamiltonian else 2, 2)
            
            # Soft-start for alignment loss: keep at 0 for first 100 epochs, then anneal in
            align_weight = 0.0
            if epoch > 100:
                align_weight = min(1.0, (epoch - 100) / self.align_anneal_epochs)
            
            # Use learnable scale for alignment
            loss_align += 2.0 * align_weight * self.criterion(z_preds[t, :, :, :d_align] * self.align_scale, mu_t_norm[:, :, :d_align])
            loss_mi += self.model.get_mi_loss(z_preds[t], mu_t_norm)

            if t > 0:
                loss_assign += self.criterion(s_t, s_prev) + 10.0 * self.criterion(mu_t, mu_prev) + 0.5 * self.criterion(z_t_target, z_enc_prev)
                if compute_consistency and not is_warmup: loss_cons += self.criterion(z_preds[t], z_t_target)
            s_prev, mu_prev, z_enc_prev = s_t, mu_t, z_t_target

        n_steps = (seq_len // 2)
        loss_rec /= n_steps
        loss_cons /= (n_steps - 1) if (not is_warmup and compute_consistency) else 1
        for l in [loss_assign, loss_pruning, loss_sparsity, loss_ortho, loss_sep, loss_conn]: l /= n_steps
        loss_align /= (n_steps * self.model.encoder.n_super_nodes)
        loss_mi /= (n_steps * self.model.encoder.n_super_nodes)

        lvars = torch.clamp(self.model.log_vars, min=-6.0, max=5.0)
        if is_warmup: lvars[11], lvars[3], lvars[12] = 5.0, 5.0, 5.0

        raw_losses = {'rec': loss_rec, 'cons': loss_cons, 'assign': loss_assign, 'ortho': loss_ortho, 'l2': loss_l2, 'lvr': loss_lvr, 'align': loss_align, 'pruning': loss_pruning, 'sep': loss_sep, 'conn': loss_conn, 'sparsity': loss_sparsity, 'mi': loss_mi, 'sym': loss_sym, 'var': loss_var, 'hinge': loss_hinge}
        
        weights = {k: torch.exp(-lvars[i]) for i, k in enumerate(raw_losses.keys())}
        if self.enhanced_balancer:
            balanced = self.enhanced_balancer.get_balanced_losses(raw_losses, self.model.parameters())
            for k in weights:
                if k in balanced and raw_losses[k].item() != 0: weights[k] = balanced[k] / raw_losses[k]

        discovery_loss = sum(weights[k] * torch.clamp(raw_losses[k].to(torch.float32), 0, 100) + lvars[i] for i, k in enumerate(raw_losses.keys()) if k not in ['cons', 'l2', 'lvr'])
        discovery_loss += (weights['sym'] * (self.symbolic_weight - 1.0) * torch.clamp(loss_sym.to(torch.float32), 0, 100)) # Adjust sym weight
        discovery_loss += 1e-4 * torch.clamp(loss_curv.to(torch.float32), 0, 100)
        
        loss = discovery_loss + (weights['l2'] * 1e-4 * torch.clamp(loss_l2, 0, 100) + lvars[4]) + (weights['lvr'] * 1e-4 * torch.clamp(loss_lvr, 0, 100) + lvars[5])
        if compute_consistency: loss += (weights['cons'] * torch.clamp(loss_cons, 0, 100) + lvars[1])
        
        loss = torch.clamp(loss + 0.1 * torch.sum(lvars**2), 0, 1e4)
        if not torch.isfinite(loss): loss = loss_rec if torch.isfinite(loss_rec) else torch.tensor(1.0, device=self.device, requires_grad=True)

        if self.enable_gradient_accumulation:
            (loss / self.grad_acc_steps).backward()
            if ((epoch + 1) % self.grad_acc_steps == 0) or (epoch == max_epochs - 1):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self.loss_tracker.update({'total': loss, 'rec_raw': loss_rec, 'cons_raw': loss_cons, 'assign': loss_assign, 'align': loss_align, 'mi': loss_mi, 'sym': loss_sym, 'lvar_raw': loss_var, 'curv_raw': loss_curv, 'hinge_raw': loss_hinge, 'lvars_mean': lvars.mean()}, weights=weights)
        
        if epoch % 100 == 0:
            print(f"  [Loss Detail] Rec: {loss_rec:.4f} | Cons: {loss_cons:.4f} | Assign: {loss_assign:.4f} | Ortho: {loss_ortho:.4f} | Align: {loss_align:.4f} | Sym: {loss_sym:.4f}")

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
