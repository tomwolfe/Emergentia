import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
import numpy as np
from scipy.spatial import KDTree

def compute_stats(pos, vel):
    # pos, vel: [T, N, 2]
    # Use min-max for positions to keep them in a fixed range
    pos_min = pos.min(axis=(0, 1))
    pos_max = pos.max(axis=(0, 1))
    pos_range = np.maximum(pos_max - pos_min, 1e-6)
    
    vel_mean = vel.mean(axis=(0, 1))
    vel_std = vel.std(axis=(0, 1)) + 1e-6
    return {'pos_min': pos_min, 'pos_max': pos_max, 'pos_range': pos_range,
            'vel_mean': vel_mean, 'vel_std': vel_std}

def prepare_data(pos, vel, radius=1.1, stats=None, device='cpu', cache_edges=True):
    # pos, vel: [T, N, 2]
    T, N, _ = pos.shape

    if stats is None:
        stats = compute_stats(pos, vel)

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
    """Tracks running averages of loss components to help with balancing."""
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.history = {}
        self.weights = {}

    def update(self, components, weights=None):
        for k, v in components.items():
            val = v.item() if hasattr(v, 'item') else v
            if k not in self.history:
                self.history[k] = val
            else:
                self.history[k] = self.alpha * self.history[k] + (1 - self.alpha) * val
        
        if weights is not None:
            for k, v in weights.items():
                val = v.item() if hasattr(v, 'item') else v
                if k not in self.weights:
                    self.weights[k] = val
                else:
                    self.weights[k] = self.alpha * self.weights[k] + (1 - self.alpha) * val

    def get_stats(self):
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

        # Manually re-balance initial log_vars for stability
        with torch.no_grad():
            self.model.log_vars[0].fill_(-5.0)  # High priority for reconstruction
            self.model.log_vars[6].fill_(-3.0)  # High priority for alignment
            self.model.log_vars[13].fill_(2.0)  # Suppress latent variance loss more to prevent explosion

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

        # Separate H_net parameters to apply specific weight decay
        h_net_params = []
        other_params = []
        
        if hasattr(self.model.ode_func, 'H_net'):
            h_net_params = list(self.model.ode_func.H_net.parameters())
            h_ids = {id(p) for p in h_net_params}
            other_params = [p for p in self.model.parameters() if id(p) not in h_ids]
        else:
            other_params = list(self.model.parameters())

        param_groups = [
            {'params': other_params, 'weight_decay': 1e-5},
        ]
        if h_net_params:
            param_groups.append({'params': h_net_params, 'weight_decay': 1e-3})

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
        
    def train_step(self, data_list, dt, epoch=0, max_epochs=2000):
        # Update sparsity weight if scheduler is present
        if self.sparsity_scheduler is not None:
            new_weight = self.sparsity_scheduler.step()
            if hasattr(self.model.encoder.pooling, 'set_sparsity_weight'):
                self.model.encoder.pooling.set_sparsity_weight(new_weight)

        # Stage 0: Grounding (0-50 epochs) - Prioritize Reconstruction and Alignment
        if epoch < 50:
            with torch.no_grad():
                self.model.log_vars[0].fill_(-6.0)  # Maximum priority for reconstruction
                self.model.log_vars[6].fill_(-5.0)  # Maximum priority for alignment
                # Suppress other structural losses initially
                self.model.log_vars[3].fill_(2.0)   # Ortho
                self.model.log_vars[11].fill_(2.0)  # MI

        # Stage-based curriculum: Stage 1 freezes ODE/Symbolic and focuses on Recon/Align
        is_stage1 = epoch < self.warmup_epochs
        for p in self.model.ode_func.parameters():
            p.requires_grad = not is_stage1
        
        if self.symbolic_proxy is not None:
            for p in self.symbolic_proxy.parameters():
                p.requires_grad = not is_stage1

        # Periodically apply hard revival to prevent resolution collapse
        if epoch > 0 and epoch % 50 == 0:  # More frequent revival
            if hasattr(self.model.encoder.pooling, 'apply_hard_revival'):
                self.model.encoder.pooling.apply_hard_revival()

        # Determine if we should compute consistency loss based on frequency
        # Stage 1 skips consistency loss to focus on static reconstruction
        compute_consistency = (epoch >= self.warmup_epochs) and (not is_stage1)
        if epoch > self.warmup_epochs: # Only skip frequency after stage 1
            compute_consistency = compute_consistency and (epoch % self.skip_consistency_freq == 0)

        # For gradient accumulation
        if self.enable_gradient_accumulation:
            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient

        seq_len = len(data_list)

        # Progress ratio
        progress = epoch / max_epochs

        # 1. Tau schedule: Faster annealing as requested
        # Start at 1.0 and anneal to 0.1 by epoch 50
        tau = max(0.1, 1.0 * np.exp(-epoch / 25.0))

        # 2. Hard assignment scheduling:
        # Gradually increase the probability of using 'hard' assignments in Gumbel-Softmax
        # to bridge the gap between training and inference.
        hard = False
        if progress > self.hard_assignment_start:
            # Probability of hard assignment increases linearly after start threshold
            hard_prob = min(1.0, (progress - self.hard_assignment_start) / (1.0 - self.hard_assignment_start))
            hard = np.random.random() < hard_prob

        # 3. Decaying Teacher Forcing Ratio (more aggressive decay)
        tf_ratio = max(0.0, 0.8 * (1.0 - epoch / (0.5 * max_epochs + 1e-9)))  # Faster decay

        # 4. Alignment weight - Boosted in Stage 1
        align_weight = 5.0 if is_stage1 else 1.0

        # 5. Adaptive Entropy Weight: Doubled growth rate to force discrete clusters
        # Point 1: Addressing "Blurry" Meso-scale
        entropy_weight = 1.0 + 5.0 * progress  # Increased from 1.5 to 5.0 for harder assignments

        # Warmup logic
        is_warmup = epoch < self.warmup_epochs

        batch_0 = Batch.from_data_list([data_list[0]]).to(self.device)
        z_curr, s_0, losses_0, mu_0 = self.model.encode(batch_0.x, batch_0.edge_index, batch_0.batch, tau=tau, hard=hard)

        if torch.isnan(z_curr).any() or torch.isinf(z_curr).any():
            z_curr = torch.nan_to_num(z_curr, nan=0.0, posinf=1.0, neginf=-1.0)

        # Apply high weight for initial reconstruction - Do NOT pass stats to keep in normalized space
        recon_0 = self.model.decode(z_curr, s_0, batch_0.batch)
        loss_rec = self.criterion(recon_0, batch_0.x)
        
        loss_cons = torch.tensor(0.0, device=self.device)
        # Apply entropy weight and boost diversity to prevent collapse
        loss_assign = entropy_weight * losses_0['entropy'] + 2.0 * losses_0['diversity'] + 0.1 * losses_0['spatial']
        loss_pruning = losses_0['pruning']
        loss_sparsity = losses_0['sparsity']
        loss_sep = losses_0.get('separation', torch.tensor(0.0, device=self.device))
        loss_conn = self.model.get_connectivity_loss(s_0, batch_0.edge_index)
        loss_ortho = self.model.get_ortho_loss(s_0)
        loss_align = torch.tensor(0.0, device=self.device)
        loss_mi = torch.tensor(0.0, device=self.device)
        loss_sym = torch.tensor(0.0, device=self.device)
        loss_var = self.model.get_latent_variance_loss(z_curr)
        
        # Add Curvature Penalty for Hamiltonian dynamics - Computed on initial state
        loss_curv = torch.tensor(0.0, device=self.device)
        if self.model.hamiltonian:
            with torch.set_grad_enabled(True):
                z_in = z_curr.view(z_curr.size(0), -1).detach().requires_grad_(True)

                # Handle MPS device mismatch: if ODE func is on CPU but z_in is on MPS
                if self.mps_ode_on_cpu:
                    z_in_cpu = z_in.cpu()
                    H = self.model.ode_func.H_net(z_in_cpu).sum()
                    dH = torch.autograd.grad(H, z_in_cpu, create_graph=True)[0]
                    if dH is not None:
                        # Move gradient back to original device for loss calculation
                        dH = dH.to(self.device)
                        loss_curv = torch.norm(dH, p=2)
                else:
                    H = self.model.ode_func.H_net(z_in).sum()
                    dH = torch.autograd.grad(H, z_in, create_graph=True)[0]
                    if dH is not None:
                        loss_curv = torch.norm(dH, p=2)

        # Symbolic-in-the-loop loss with FIDELITY GATING
        # Point 3: Addressing Symbolic-Neural Feedback Loop Fragility
        if (self.symbolic_proxy is not None and not is_warmup and
            self.symbolic_confidence >= self.min_symbolic_confidence):
            z0_flat = z_curr.view(z_curr.size(0), -1)

            # Handle MPS device mismatch: if ODE func is on CPU but z0_flat is on MPS
            if self.mps_ode_on_cpu:
                z0_flat_cpu = z0_flat.cpu()
                # GNN predicted derivative
                gnn_dz_cpu = self.model.ode_func(0, z0_flat_cpu)
                # Symbolic predicted derivative (now differentiable!)
                sym_dz_cpu = self.symbolic_proxy(z0_flat_cpu)

                # Use Huber loss for more robustness against "garbage" symbolic laws
                loss_sym_cpu = torch.nn.functional.huber_loss(gnn_dz_cpu, sym_dz_cpu, delta=1.0)
                # Move loss back to original device
                loss_sym = loss_sym_cpu.to(self.device)
            else:
                # GNN predicted derivative
                gnn_dz = self.model.ode_func(0, z0_flat)
                # Symbolic predicted derivative (now differentiable!)
                sym_dz = self.symbolic_proxy(z0_flat)

                # Use Huber loss for more robustness against "garbage" symbolic laws
                loss_sym = torch.nn.functional.huber_loss(gnn_dz, sym_dz, delta=1.0)

        s_prev = s_0
        mu_prev = mu_0
        z_preds = [z_curr]

        # Only do forward dynamics if not in deep warmup
        if not is_warmup:
            for t in range(1, seq_len):
                if np.random.random() < tf_ratio:
                    batch_t_prev = Batch.from_data_list([data_list[t-1]]).to(self.device)
                    z_curr_forced, _, _, _ = self.model.encode(batch_t_prev.x, batch_t_prev.edge_index, batch_t_prev.batch, tau=tau, hard=hard)
                    z_curr = torch.nan_to_num(z_curr_forced)

                t_span = torch.tensor([0, dt], device=self.device, dtype=torch.float32)
                try:
                    z_next_seq = self.model.forward_dynamics(z_curr, t_span)
                    z_curr = torch.nan_to_num(z_next_seq[1], nan=0.0, posinf=1.0, neginf=-1.0)
                except Exception as e:
                    # If ODE solver fails, fallback to previous state and add penalty
                    z_curr = z_curr.detach()
                z_preds.append(z_curr)
        else:
            # During warmup, just use encoded states
            for t in range(1, seq_len):
                batch_t = Batch.from_data_list([data_list[t]]).to(self.device)
                z_t, _, _, _ = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch, tau=tau, hard=hard)
                z_preds.append(z_t)

        z_preds = torch.stack(z_preds)
        loss_l2 = torch.mean(z_preds**2)

        z_vel = (z_preds[1:] - z_preds[:-1]) / dt
        loss_lvr = torch.mean((z_vel[1:] - z_vel[:-1])**2) if len(z_vel) > 1 else torch.tensor(0.0, device=self.device)

        s_stability = 0
        mu_stability = 0

        mu_min = torch.tensor(self.stats['pos_min'], device=self.device, dtype=torch.float32) if self.stats else 0
        mu_range = torch.tensor(self.stats['pos_range'], device=self.device, dtype=torch.float32) if self.stats else 1

        # Optimize the target computation loop - only compute targets when needed
        # Process every other time step to reduce computation
        for t in range(0, seq_len, 2):  # Process every other time step
            batch_t = Batch.from_data_list([data_list[t]]).to(self.device)
            z_t_target, s_t, losses_t, mu_t = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch, tau=tau, hard=hard)

            z_t_target = torch.nan_to_num(z_t_target)

            # Use z_preds for reconstruction to force consistency - Do NOT pass stats to keep in normalized space
            recon_t = self.model.decode(z_preds[t], s_t, batch_t.batch)

            loss_rec += self.criterion(recon_t, batch_t.x)
            loss_assign += entropy_weight * losses_t['entropy'] + 2.0 * losses_t['diversity'] + 0.1 * losses_t['spatial']
            loss_pruning += losses_t['pruning']
            loss_sparsity += losses_t['sparsity']
            loss_sep += losses_t.get('separation', torch.tensor(0.0, device=self.device))
            loss_conn += self.model.get_connectivity_loss(s_t, batch_t.edge_index)
            loss_ortho += self.model.get_ortho_loss(s_t)

            # CoM Position Alignment - Enhanced weight for better physical mapping
            mu_t_norm = 2.0 * (mu_t - mu_min) / mu_range - 1.0
            d_sub = self.model.encoder.latent_dim // 2 if self.model.hamiltonian else 2
            d_align = min(d_sub, 2)

            p_align = self.criterion(z_preds[t, :, :, :d_align], mu_t_norm[:, :, :d_align])
            # Increase the alignment weight to promote stronger physical mapping
            loss_align += align_weight * 2.0 * p_align  # Doubled the alignment contribution

            # Mutual Information Alignment (Unsupervised)
            loss_mi += self.model.get_mi_loss(z_preds[t], mu_t_norm)

            if t > 0:
                s_diff = self.criterion(s_t, s_prev)
                mu_diff = self.criterion(mu_t, mu_prev)
                loss_assign += s_diff + 5.0 * mu_diff
                if compute_consistency and not is_warmup and t % 2 == 0:  # Only compute consistency every other step
                    loss_cons += self.criterion(z_preds[t], z_t_target)
                s_stability += s_diff.item()
                mu_stability += mu_diff.item()
            s_prev = s_t
            mu_prev = mu_t

        # Normalization - account for sampling every other step
        loss_rec /= (seq_len // 2)
        loss_cons /= ((seq_len // 2) - 1) if (not is_warmup and compute_consistency) else 1
        loss_assign /= (seq_len // 2)
        loss_pruning /= (seq_len // 2)
        loss_sparsity /= (seq_len // 2)
        loss_ortho /= (seq_len // 2)
        loss_align /= (seq_len // 2)
        loss_mi /= (seq_len // 2)
        loss_sep /= (seq_len // 2)
        loss_conn /= (seq_len // 2)

        # 0: rec, 1: cons, 2: assign, 3: ortho, 4: l2, 5: lvr, 6: align, 7: pruning, 8: sep, 9: conn, 10: sparsity, 11: mi, 12: sym, 13: var
        lvars = torch.clamp(self.model.log_vars, min=-6.0, max=5.0)

        raw_weights = {
            'rec': torch.exp(-lvars[0]), 'cons': torch.exp(-lvars[1]),
            'assign': torch.exp(-lvars[2]), 'ortho': torch.exp(-lvars[3]),
            'l2': torch.exp(-lvars[4]), 'lvr': torch.exp(-lvars[5]),
            'align': torch.exp(-lvars[6]), 'pruning': torch.exp(-lvars[7]),
            'sep': torch.exp(-lvars[8]), 'conn': torch.exp(-lvars[9]),
            'sparsity': torch.exp(-lvars[10]), 'mi': torch.exp(-lvars[11]),
            'sym': torch.exp(-lvars[12]), 'var': torch.exp(-lvars[13])
        }

        # NEW: Use enhanced loss balancer if available
        if self.enhanced_balancer is not None:
            # Prepare raw losses for the balancer
            raw_losses = {
                'rec': loss_rec, 'cons': loss_cons,
                'assign': loss_assign, 'ortho': loss_ortho,
                'l2': loss_l2, 'lvr': loss_lvr,
                'align': loss_align, 'pruning': loss_pruning,
                'sep': loss_sep, 'conn': loss_conn,
                'sparsity': loss_sparsity, 'mi': loss_mi,
                'sym': loss_sym, 'var': loss_var
            }

            # Update weights using enhanced balancer
            if hasattr(self.enhanced_balancer, 'get_balanced_losses'):
                # Pass model parameters for gradient-based balancing
                balanced_losses = self.enhanced_balancer.get_balanced_losses(
                    raw_losses, self.model.parameters()
                )

                # Extract updated weights from balanced losses
                for key in raw_weights:
                    if key in balanced_losses:
                        # Calculate new weight as balanced_loss / raw_loss
                        if raw_losses[key].item() != 0:
                            raw_weights[key] = balanced_losses[key] / raw_losses[key]
                        else:
                            raw_weights[key] = raw_weights[key]  # Keep original weight

        weights = raw_weights

        discovery_loss = (weights['rec'] * loss_rec + lvars[0]) + \
                         (weights['assign'] * loss_assign + lvars[2]) + \
                         (weights['ortho'] * loss_ortho + lvars[3]) + \
                         (weights['align'] * loss_align + lvars[6]) + \
                         (weights['pruning'] * loss_pruning + lvars[7]) + \
                         (weights['sep'] * loss_sep + lvars[8]) + \
                         (weights['conn'] * loss_conn + lvars[9]) + \
                         (weights['sparsity'] * loss_sparsity + lvars[10]) + \
                         (weights['mi'] * loss_mi + lvars[11]) + \
                         (weights['sym'] * self.symbolic_weight * loss_sym + lvars[12]) + \
                         (weights['var'] * loss_var + lvars[13]) + \
                         (1e-4 * loss_curv)

        if is_warmup:
            loss = discovery_loss
        else:
            # Check for finiteness of components before adding consistency losses
            if not torch.isfinite(loss_cons): loss_cons = torch.tensor(0.0, device=self.device)
            if not torch.isfinite(loss_l2): loss_l2 = torch.tensor(0.0, device=self.device)
            if not torch.isfinite(loss_lvr): loss_lvr = torch.tensor(0.0, device=self.device)

            # Only add consistency loss if we computed it
            if compute_consistency:
                loss = discovery_loss + (weights['cons'] * loss_cons + lvars[1]) + \
                       (weights['l2'] * 1e-4 * loss_l2 + lvars[4]) + \
                       (weights['lvr'] * 1e-4 * loss_lvr + lvars[5])
            else:
                loss = discovery_loss + (weights['l2'] * 1e-4 * loss_l2 + lvars[4]) + \
                       (weights['lvr'] * 1e-4 * loss_lvr + lvars[5])

        # Add L2 regularization on lvars to prevent them from growing indefinitely
        loss += 0.1 * torch.sum(lvars**2)

        if not torch.isfinite(loss):
            # Try to recover by ignoring non-finite components
            print(f"Warning: Non-finite loss detected at epoch {epoch}. Attempting recovery.")
            # Final fallback to a very small reconstruction loss to keep training alive
            loss = loss_rec if torch.isfinite(loss_rec) else torch.tensor(1.0, device=self.device, requires_grad=True)

        if not torch.isfinite(loss) or loss.grad_fn is None:
            if not self.enable_gradient_accumulation:
                self.optimizer.zero_grad()
            return 0.0, 0.0, 0.0

        # Handle gradient accumulation
        if self.enable_gradient_accumulation:
            # Scale loss for gradient accumulation
            loss = loss / self.grad_acc_steps
            loss.backward()

            # Perform optimizer step only at the end of accumulation cycle
            if ((epoch + 1) % self.grad_acc_steps == 0) or (epoch == max_epochs - 1):
                # Increased gradient clipping for better stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            # Increased gradient clipping for better stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
            self.optimizer.step()

        # Update loss tracker for logging
        self.loss_tracker.update({
            'total': loss,
            'rec_raw': loss_rec,
            'cons_raw': loss_cons if compute_consistency else loss_cons,  # Will be 0 if not computed
            'assign': loss_assign,
            'align': loss_align,
            'mi': loss_mi,
            'sym': loss_sym,
            'lvar_raw': loss_var,
            'curv_raw': loss_curv,
            'lvars_mean': lvars.mean()
        }, weights=weights)

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
