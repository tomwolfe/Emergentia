import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data, Batch
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
import numpy as np
from scipy.spatial import KDTree

from loss_functions import LossTracker, GradNormBalancer, LossFactory
from hardware_manager import HardwareManager
from symbolic_proxy import SymbolicProxy

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
    
    # OPTIMIZATION: Pre-convert to tensors
    pos_tensors = torch.from_numpy(pos_norm).to(device)
    vel_tensors = torch.from_numpy(vel_norm).to(device)

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
        x = torch.cat([pos_tensors[t], vel_tensors[t]], dim=1)
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

class Trainer:
    def __init__(self, model, lr=5e-4, device='cpu', stats=None, align_anneal_epochs=1000,
                 warmup_epochs=20, max_epochs=1000, sparsity_scheduler=None, hard_assignment_start=0.7,
                 skip_consistency_freq=2, enable_gradient_accumulation=False, grad_acc_steps=1,
                 enhanced_balancer=None, consistency_weight=1.0, spatial_weight=1.0, use_pcgrad=False):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.hardware = HardwareManager(device)
        self.loss_tracker = LossTracker()
        self.s_history = []
        self.max_s_history = 10
        self.align_anneal_epochs = align_anneal_epochs
        self.warmup_epochs = warmup_epochs
        self.sparsity_scheduler = sparsity_scheduler
        self.hard_assignment_start = hard_assignment_start
        self.skip_consistency_freq = skip_consistency_freq
        self.enable_gradient_accumulation = enable_gradient_accumulation
        self.grad_acc_steps = grad_acc_steps
        self.enhanced_balancer = enhanced_balancer
        self.consistency_weight = consistency_weight
        self.spatial_weight = spatial_weight
        self.use_pcgrad = use_pcgrad

        # Initialize consolidated loss modules from factory
        self.loss_modules = LossFactory.create_loss_modules()
        self.loss_groups = LossFactory.get_loss_groups()

        # Learnable log-scaling for alignment
        self.log_align_scale = torch.nn.Parameter(torch.tensor(0.0, device=device))

        with torch.no_grad():
            self.model.log_vars.fill_(0.0)

        if hasattr(self.model.encoder.pooling, 'temporal_consistency_weight'):
            self.model.encoder.pooling.temporal_consistency_weight = 1.0

        # Symbolic-in-the-loop
        self.symbolic_proxy = None
        self.symbolic_weight = 0.0
        self.symbolic_confidence = 0.0
        self.min_symbolic_confidence = 0.7

        # Hardware setup
        self.hardware.prepare_model_for_ode(self.model)

        # Separate parameters for specific weight decay
        param_groups = self._prepare_param_groups(lr)
        self.optimizer = optim.Adam(param_groups, lr=lr)
        self.criterion = torch.nn.MSELoss().to(device)
        self.stats = stats
        
        # Initialize GradNorm Balancer
        self.grad_norm_balancer = GradNormBalancer(n_tasks=len(self.loss_groups), device=device)
        self.gn_optimizer = optim.Adam(self.grad_norm_balancer.parameters(), lr=1e-2)

    def _prepare_param_groups(self, lr):
        h_net_params = []
        assign_mlp_params = []
        output_layer_params = []
        
        if hasattr(self.model.ode_func, 'H_net') and self.model.ode_func.H_net is not None:
            h_net_params = list(self.model.ode_func.H_net.parameters())
        elif hasattr(self.model.ode_func, 'V_net') and self.model.ode_func.V_net is not None:
            h_net_params = list(self.model.ode_func.V_net.parameters())

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
            {'params': [self.model.log_vars], 'lr': 1e-1},
        ]
        if h_net_params:
            param_groups.append({'params': h_net_params, 'weight_decay': 1e-5})
        if assign_mlp_params:
            param_groups.append({'params': assign_mlp_params, 'weight_decay': 1e-3})
        if output_layer_params:
            param_groups.append({'params': output_layer_params, 'weight_decay': 1e-3})
        return param_groups

    def update_symbolic_proxy(self, symbolic_proxy_or_equations, transformer=None, weight=0.1, confidence=0.0):
        """Update the symbolic proxy model with new discovered equations."""
        proxy_device = self.hardware.get_ode_device()

        if hasattr(symbolic_proxy_or_equations, 'forward'):
            self.symbolic_proxy = symbolic_proxy_or_equations.to(proxy_device)
        else:
            self.symbolic_proxy = SymbolicProxy(
                self.model.encoder.n_super_nodes,
                self.model.encoder.latent_dim,
                symbolic_proxy_or_equations,
                transformer
            ).to(proxy_device)

        self.symbolic_weight = weight
        self.symbolic_confidence = confidence
        print(f"Symbolic proxy updated. Weight: {weight}, Confidence: {confidence:.3f} on {proxy_device}")
        
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

    def _compute_hamiltonian_curv_loss(self, z_curr):
        loss_curv = torch.tensor(0.0, device=self.device)
        if self.model.hamiltonian:
            with torch.set_grad_enabled(True):
                z_in = z_curr.reshape(z_curr.size(0), -1).detach().requires_grad_(True)
                z_in_ode = self.hardware.to_ode_device(z_in)
                H_vals = self.model.ode_func.hamiltonian(z_in_ode)
                H = H_vals.sum()
                dH = torch.autograd.grad(H, z_in_ode, create_graph=True, retain_graph=True, allow_unused=True)[0]
                if dH is not None:
                    loss_curv = torch.norm(self.hardware.to_main_device(dH), p=2) + 0.01 * torch.mean(self.hardware.to_main_device(H_vals)**2)
        return loss_curv

    def _compute_latent_smoothing_loss(self, z_preds):
        if z_preds.size(0) < 3:
            return torch.tensor(0.0, device=self.device)
        first_diff = z_preds[1:] - z_preds[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]
        return torch.mean(second_diff**2)

    def _compute_symbolic_loss(self, z_curr, is_warmup):
        loss_sym = torch.tensor(0.0, device=self.device)
        if (self.symbolic_proxy is not None and not is_warmup and
            self.symbolic_confidence >= self.min_symbolic_confidence):
            z0_flat = z_curr.view(z_curr.size(0), -1)
            with torch.enable_grad():
                z_in = z0_flat.detach().requires_grad_(True)
                z_in_ode = self.hardware.to_ode_device(z_in)
                gnn_dz = self.model.ode_func(0, z_in_ode)
                sym_dz = self.symbolic_proxy(0, z_in_ode)
                
                loss_sym_val = torch.nn.functional.huber_loss(gnn_dz, sym_dz, delta=0.1)
                try:
                    v = torch.randn_like(gnn_dz)
                    grad_gnn = torch.autograd.grad(gnn_dz, z_in_ode, grad_outputs=v, create_graph=True)[0]
                    grad_sym = torch.autograd.grad(sym_dz, z_in_ode, grad_outputs=v, create_graph=True)[0]
                    loss_jac = torch.nn.functional.mse_loss(grad_gnn, grad_sym)
                    loss_sym = self.hardware.to_main_device(loss_sym_val + 0.1 * loss_jac)
                except Exception:
                    loss_sym = self.hardware.to_main_device(loss_sym_val)
        return loss_sym

    def _apply_pcgrad(self, head_losses):
        head_grads = []
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        for name, loss in head_losses.items():
            if loss.requires_grad:
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                flat_grads = []
                for g, p in zip(grads, params):
                    if g is not None:
                        flat_grads.append(g.flatten().to(self.device))
                    else:
                        flat_grads.append(torch.zeros_like(p).flatten().to(self.device))
                head_grads.append(torch.cat(flat_grads))
        
        if not head_grads: return

        import random
        random.shuffle(head_grads)
        pc_grads = [g.clone() for g in head_grads]
        for i in range(len(pc_grads)):
            for j in range(len(head_grads)):
                if i == j: continue
                dot_prod = torch.dot(pc_grads[i], head_grads[j])
                if dot_prod < 0:
                    pc_grads[i] -= (dot_prod / (torch.norm(head_grads[j])**2 + 1e-8)) * head_grads[j]
        
        final_grad = torch.stack(pc_grads).sum(dim=0)
        idx = 0
        for p in params:
            n = p.numel()
            if p.grad is None: p.grad = final_grad[idx:idx+n].view(p.shape).clone().to(p.device)
            else: p.grad.add_(final_grad[idx:idx+n].view(p.shape).to(p.device))
            idx += n

    def train_step(self, data_list, dt, epoch=0, max_epochs=2000):
        if self.sparsity_scheduler is not None:
            new_weight = self.sparsity_scheduler.step()
            if hasattr(self.model.encoder.pooling, 'set_sparsity_weight'):
                self.model.encoder.pooling.set_sparsity_weight(new_weight)

        stage1_end = int(max_epochs * 0.15)
        is_stage1 = epoch < stage1_end
        is_stage2 = epoch >= stage1_end
        is_warmup = epoch < int(max_epochs * 0.1)

        loss_multipliers = torch.ones(17, device=self.device)
        if is_stage1:
            loss_multipliers[1], loss_multipliers[4], loss_multipliers[5], loss_multipliers[12], loss_multipliers[15] = 0.0, 0.0001, 0.0001, 0.0, 0.01
            loss_multipliers[0], loss_multipliers[2], loss_multipliers[6], loss_multipliers[9], loss_multipliers[16] = 100.0, 5.0, 20.0, 20.0, 50.0

        for p in self.model.ode_func.parameters(): p.requires_grad = is_stage2
        if self.symbolic_proxy is not None:
            for p in self.symbolic_proxy.parameters(): p.requires_grad = is_stage2

        if epoch > 0 and epoch % 100 == 0:
            if hasattr(self.model.encoder.pooling, 'apply_hard_revival'):
                self.model.encoder.pooling.apply_hard_revival()

        compute_consistency = (epoch >= (int(max_epochs * 0.1) // 2)) and (epoch % self.skip_consistency_freq == 0)
        self.optimizer.zero_grad(set_to_none=True)

        tau, hard, tf_ratio, entropy_weight = self._get_schedules(epoch, max_epochs)
        
        if isinstance(data_list, Batch):
            full_batch = data_list
            seq_len = getattr(full_batch, 'seq_len', full_batch.num_graphs)
            batch_size = full_batch.num_graphs // seq_len
        else:
            full_batch = Batch.from_data_list(data_list).to(self.device)
            seq_len = len(data_list)
            batch_size = 1

        z_all_target, s_all, losses_all, mu_all = self.model.encode(
            full_batch.x, full_batch.edge_index, full_batch.batch, 
            tau=tau, hard=hard, current_epoch=epoch, total_epochs=max_epochs
        )
        
        z_all_target = z_all_target.reshape(seq_len, batch_size, self.model.encoder.n_super_nodes, -1)
        nodes_per_traj_step = full_batch.num_nodes // (seq_len * batch_size)
        mu_all = mu_all.reshape(seq_len, batch_size, self.model.encoder.n_super_nodes, 2)
        z_curr = z_all_target[0]
        
        s_reshaped = s_all.view(seq_len, batch_size, nodes_per_traj_step, -1)
        s_0 = s_reshaped[0].reshape(-1, self.model.encoder.n_super_nodes)
        
        x_reshaped = full_batch.x.view(seq_len, batch_size, nodes_per_traj_step, -1)
        x_target_0 = x_reshaped[0].reshape(-1, 4)
        
        batch_idx_0 = torch.arange(batch_size, device=self.device).repeat_interleave(nodes_per_traj_step)
        recon_0 = self.model.decode(z_curr, s_0, batch_idx_0, stats=None)
        loss_rec = self.criterion(recon_0, x_target_0)
        
        loss_assign = (
            (2.0 * entropy_weight) * losses_all['entropy'] +
            10.0 * losses_all['diversity'] +
            5.0 * self.spatial_weight * losses_all['spatial'] +
            1.0 * losses_all.get('collapse_prevention', 0.0) +
            1.0 * losses_all.get('balance', 0.0) +
            1.0 * losses_all.get('temporal_consistency', 0.0)
        )

        # OPTIMIZATION: Skip expensive losses if weight is zero
        loss_curv = torch.tensor(0.0, device=self.device)
        if is_stage2 and not is_stage1:
             loss_curv = self._compute_hamiltonian_curv_loss(z_curr)

        loss_sym = torch.tensor(0.0, device=self.device)
        if self.symbolic_proxy is not None and not is_warmup:
            loss_sym = self._compute_symbolic_loss(z_curr, is_warmup)

        if not is_warmup:
            progress_after_warmup = min(1.0, (epoch - int(max_epochs * 0.1)) / 100.0)
            target_integration_steps = max(4, int(progress_after_warmup * seq_len))
            z_curr_ode = self.hardware.to_ode_device(z_curr)
            
            if tf_ratio == 0:
                t_span = torch.linspace(0, (target_integration_steps - 1) * dt, target_integration_steps, device=self.hardware.get_ode_device())
                z_preds_integrated = self.model.forward_dynamics(z_curr_ode, t_span)
                z_preds_integrated = self.hardware.to_main_device(z_preds_integrated)
                z_preds = torch.cat([z_preds_integrated, z_all_target[target_integration_steps:]], dim=0) if target_integration_steps < seq_len else z_preds_integrated
            else:
                chunk_size = 10 if self.hardware.is_mps else 5
                z_preds_list, z_step = [z_curr], z_curr
                for i in range(1, target_integration_steps, chunk_size):
                    end_idx = min(i + chunk_size, target_integration_steps)
                    actual_chunk = end_idx - i
                    if np.random.random() < tf_ratio: z_step = z_all_target[i-1]
                    t_span_chunk = torch.linspace(0, actual_chunk * dt, actual_chunk + 1, device=self.hardware.get_ode_device())
                    z_chunk_preds = self.hardware.to_main_device(self.model.forward_dynamics(self.hardware.to_ode_device(z_step), t_span_chunk))
                    for j in range(1, actual_chunk + 1): z_preds_list.append(z_chunk_preds[j])
                    z_step = z_chunk_preds[-1]
                z_preds = torch.stack(z_preds_list)
                if len(z_preds) < seq_len: z_preds = torch.cat([z_preds, z_all_target[len(z_preds):]], dim=0)
                z_preds = z_preds[:seq_len]
        else:
            z_preds = z_all_target

        loss_l2 = torch.mean(z_preds**2) * 0.1
        z_vel = (z_preds[1:] - z_preds[:-1]) / dt
        z_acc = (z_vel[1:] - z_vel[:-1]) / dt if len(z_vel) > 1 else torch.tensor(0.0, device=self.device)
        loss_lvr = 0.1 * (torch.mean(torch.clamp(z_acc**2, 0, 100)) + torch.mean(torch.clamp(z_vel**2, 0, 100))) if len(z_vel) > 1 else torch.tensor(0.0, device=self.device)
        loss_smooth = self._compute_latent_smoothing_loss(z_preds) * (dt / 0.001)**2
        loss_pruning, loss_sparsity, loss_sep = losses_all['pruning'], losses_all['sparsity'], losses_all.get('separation', torch.tensor(0.0, device=self.device))
        loss_conn = 10.0 * self.model.get_connectivity_loss(s_0, full_batch.edge_index[:, full_batch.edge_index[0] < (batch_size * nodes_per_traj_step)])
        loss_ortho = self.model.get_ortho_loss(s_0)
        loss_var, loss_hinge = self.model.get_latent_variance_loss(z_curr) * 0.1, torch.mean(torch.relu(0.1 - torch.norm(z_preds, dim=-1))) * 0.1

        processed_indices = [t for t in range(seq_len) if (t % 2 == 0 or t == seq_len - 1)]
        processed_steps = len(processed_indices)
        z_batch = z_preds[processed_indices].view(-1, self.model.encoder.n_super_nodes, self.model.encoder.latent_dim)
        s_batch = s_reshaped[processed_indices].reshape(-1, self.model.encoder.n_super_nodes)
        x_target_batch = x_reshaped[processed_indices].reshape(-1, 4)
        decode_batch_idx = torch.arange(processed_steps * batch_size, device=self.device).repeat_interleave(nodes_per_traj_step)
        recon_all = self.model.decode(z_batch, s_batch, decode_batch_idx, stats=None)
        loss_rec = self.criterion(recon_all, x_target_batch)

        alignment_weight = 0.01
        if epoch > int(max_epochs * 0.1):
            progress_after_warmup = min(1.0, (epoch - int(max_epochs * 0.1)) / max(1, self.align_anneal_epochs))
            alignment_weight = max(0.01, (0.05 + 0.95 * progress_after_warmup) * (0.2 + 0.8 * (1.0 - min(1.0, loss_rec.item() / 0.5))))
        
        mu_processed = mu_all[processed_indices].view(-1, self.model.encoder.n_super_nodes, 2)
        z_processed = z_preds[processed_indices].view(-1, self.model.encoder.n_super_nodes, self.model.encoder.latent_dim)
        loss_anchor = self.criterion(z_processed[:, :, :2], mu_processed)
        loss_align = 2.0 * alignment_weight * torch.nn.functional.smooth_l1_loss(torch.clamp(z_processed[:, :, :2], -1.0, 1.0) * torch.exp(self.log_align_scale), torch.clamp(mu_processed, -1.0, 1.0), beta=0.1)
        loss_mi = alignment_weight * 5.0 * self.model.get_mi_loss(z_processed, mu_processed) if epoch % 5 == 0 or epoch > int(max_epochs * 0.75) else torch.tensor(0.0, device=self.device)

        loss_cons = torch.tensor(0.0, device=self.device)
        if len(processed_indices) > 1:
            curr_idx, prev_idx = processed_indices[1:], processed_indices[:-1]
            loss_assign += (self.criterion(s_reshaped[curr_idx], s_reshaped[prev_idx]) + 10.0 * self.criterion(mu_all[curr_idx], mu_all[prev_idx]) + 0.5 * self.criterion(z_all_target[curr_idx], z_all_target[prev_idx]))
            if compute_consistency and not is_warmup: loss_cons = self.criterion(z_preds[curr_idx], z_all_target[curr_idx])

        norm_factor = processed_steps * batch_size
        loss_assign, loss_pruning, loss_sparsity, loss_ortho, loss_sep, loss_conn, loss_anchor = loss_assign / norm_factor, loss_pruning / norm_factor, loss_sparsity / norm_factor, loss_ortho / norm_factor, loss_sep / norm_factor, loss_conn / norm_factor, loss_anchor / norm_factor
        loss_align, loss_mi = loss_align / (norm_factor * self.model.encoder.n_super_nodes), loss_mi / (norm_factor * self.model.encoder.n_super_nodes)

        with torch.no_grad(): self.model.log_vars[4:6].clamp_(min=0.0)
        lvars = torch.clamp(self.model.log_vars, min=-6.0, max=5.0)
        raw_losses = {'rec': loss_rec, 'cons': loss_cons, 'assign': loss_assign, 'ortho': loss_ortho, 'l2': loss_l2, 'lvr': loss_lvr, 'align': loss_align, 'pruning': loss_pruning, 'sep': loss_sep, 'conn': loss_conn, 'sparsity': loss_sparsity, 'mi': loss_mi, 'sym': loss_sym, 'var': loss_var, 'hinge': loss_hinge, 'smooth': loss_smooth, 'anchor': loss_anchor, 'curv': loss_curv, 'activity': (torch.relu(1.0 - torch.norm(z_vel, dim=-1)).mean() if len(z_vel) > 0 else self.model.get_activity_penalty(z_preds)) * 1000.0}
        
        weights = {k: (100.0 if is_stage1 and k == 'rec' else (10.0 if k in ['rec', 'assign', 'anchor'] else 1.0)) for k in raw_losses}
        if epoch >= int(max_epochs * 0.75):
            for i, k in enumerate(list(raw_losses.keys())[:17]): weights[k] = torch.exp(-lvars[i]) * loss_multipliers[i]
        
        if epoch < (max_epochs * 0.3): weights['l2'] = weights['lvr'] = weights['smooth'] = 0.0
        
        stage2_factor = (1.0 / (1 + np.exp(-12.0 * (min(1.0, (epoch - stage1_end) / max(1, int(max_epochs * 0.1))) - 0.5)))) if is_stage2 else 0.0

        head_losses_dict = {
            'ReconstructionLoss': self.loss_modules['ReconstructionLoss'](loss_rec * weights['rec'], loss_cons * weights['cons']),
            'StructuralLoss': self.loss_modules['StructuralLoss'](loss_assign * weights['assign'], loss_ortho * weights['ortho'], loss_conn * weights['conn'], loss_pruning * weights['pruning'], loss_sparsity * weights['sparsity'], loss_sep * weights['sep']),
            'PhysicalityLoss': self.loss_modules['PhysicalityLoss'](loss_align * weights['align'], loss_mi * weights['mi'], loss_anchor * weights['anchor'], loss_curv * weights['curv'], raw_losses['activity'] * weights.get('activity', 1.0), loss_l2 * weights['l2'], loss_lvr * weights['lvr'], loss_hinge * weights['hinge'], loss_smooth * weights['smooth']),
            'SymbolicConsistencyLoss': self.loss_modules['SymbolicConsistencyLoss'](loss_sym * weights['sym'], stage2_factor, self.symbolic_weight)
        }

        if epoch >= stage1_end:
            for i, (name, h_loss) in enumerate(head_losses_dict.items()):
                if i < 4: head_losses_dict[name] = h_loss * torch.exp(-lvars[i]) + lvars[i]

        do_grad_norm = (epoch > int(max_epochs * 0.1)) and not is_stage1
        if do_grad_norm:
            gn_weights = self.grad_norm_balancer.get_weights()
            head_names = ['ReconstructionLoss', 'StructuralLoss', 'PhysicalityLoss', 'SymbolicConsistencyLoss']
            head_losses_list = [head_losses_dict[h] for h in head_names]
            loss = sum(gn_weights[i] * head_losses_list[i] for i in range(len(head_names)))
            loss.backward(retain_graph=True)
            self.grad_norm_balancer.update(head_losses_list, list(self.model.encoder.parameters()), self.gn_optimizer)
            self.gn_optimizer.step(); self.gn_optimizer.zero_grad()
        elif self.use_pcgrad and (epoch > int(max_epochs * 0.1)) and not is_stage1:
            self._apply_pcgrad(head_losses_dict)
            loss = torch.stack(list(head_losses_dict.values())).sum()
        else:
            loss = torch.stack(list(head_losses_dict.values())).sum()
            loss = torch.clamp(loss + 0.1 * torch.sum(lvars**2), 1e-4, 1e5)
            if self.enable_gradient_accumulation: (loss / self.grad_acc_steps).backward()
            else: loss.backward()

        if self.enable_gradient_accumulation:
            if ((epoch + 1) % self.grad_acc_steps == 0) or (epoch == max_epochs - 1):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5 if epoch > int(max_epochs * 0.1) and stage2_factor >= 0.95 else 1.0)
                self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5 if epoch > int(max_epochs * 0.1) and stage2_factor >= 0.95 else 1.0)
            self.optimizer.step()

        self.loss_tracker.update({'total': loss.to(torch.float32), 'rec_raw': loss_rec, 'cons_raw': loss_cons, 'assign': loss_assign, 'align': loss_align, 'mi': loss_mi, 'sym': loss_sym, 'lvar_raw': loss_var, 'curv_raw': loss_curv, 'hinge_raw': loss_hinge, 'smooth_raw': loss_smooth, 'anchor_raw': loss_anchor, 'lvars_mean': lvars.mean()}, weights=weights)
        
        if epoch % 100 == 0:
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