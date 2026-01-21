import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
import numpy as np
from scipy.spatial import KDTree

def compute_stats(pos, vel):
    # pos, vel: [T, N, 2]
    pos_mean = pos.mean(axis=(0, 1))
    pos_std = pos.std(axis=(0, 1)) + 1e-6
    vel_mean = vel.mean(axis=(0, 1))
    vel_std = vel.std(axis=(0, 1)) + 1e-6
    return {'pos_mean': pos_mean, 'pos_std': pos_std, 
            'vel_mean': vel_mean, 'vel_std': vel_std}

def prepare_data(pos, vel, radius=1.1, stats=None, device='cpu'):
    # pos, vel: [T, N, 2]
    T, N, _ = pos.shape
    
    if stats is None:
        stats = compute_stats(pos, vel)
    
    pos_norm = (pos - stats['pos_mean']) / stats['pos_std']
    vel_norm = (vel - stats['vel_mean']) / stats['vel_std']
    
    dataset = []
    # Pre-convert to float32 for speed
    pos_norm = pos_norm.astype(np.float32)
    vel_norm = vel_norm.astype(np.float32)
    
    for t in range(T):
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
        corrs = np.array([[np.corrcoef(z_k[:, i], com_k[:, j])[0, 1] 
                          for j in range(2)] for i in range(z_k.shape[1])])
        avg_corrs.append(np.nan_to_num(corrs))
        
    return np.array(avg_corrs)

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

class Trainer:
    def __init__(self, model, lr=5e-4, device='cpu', stats=None, align_anneal_epochs=1000, warmup_epochs=400):
        self.model = model.to(device)
        self.device = device
        self.loss_tracker = LossTracker()
        self.s_history = []
        self.max_s_history = 10
        self.align_anneal_epochs = align_anneal_epochs
        self.warmup_epochs = warmup_epochs
        
        # MPS fix: torchdiffeq has issues with MPS (float64 defaults and stability)
        if str(device) == 'mps':
            self.model.ode_func.to('cpu')
            
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = torch.nn.MSELoss().to(device)
        self.stats = stats
        
    def train_step(self, data_list, dt, epoch=0, max_epochs=2000):
        self.optimizer.zero_grad()
        seq_len = len(data_list)
        
        # 1. Smooth Tau schedule: Exponential decay
        tau = max(0.1, np.exp(-epoch / 500.0))
        
        # 2. Decaying Teacher Forcing Ratio
        tf_ratio = max(0.0, 0.8 * (1.0 - epoch / (0.75 * max_epochs)))
        
        # 3. Alignment Annealing
        align_weight = max(0.0, 1.0 - epoch / self.align_anneal_epochs)
        
        # Warmup logic: Before warmup_epochs, we focus on reconstruction and pooling
        # only. After that, we introduce ODE consistency and alignment.
        is_warmup = epoch < self.warmup_epochs

        batch_0 = Batch.from_data_list([data_list[0]]).to(self.device)
        z_curr, s_0, losses_0, mu_0 = self.model.encode(batch_0.x, batch_0.edge_index, batch_0.batch, tau=tau)
        
        if torch.isnan(z_curr).any():
            z_curr = torch.nan_to_num(z_curr)

        loss_rec = 0
        loss_cons = 0
        loss_assign = losses_0['entropy'] + losses_0['diversity'] + 0.1 * losses_0['spatial']
        loss_pruning = losses_0['pruning']
        loss_sparsity = losses_0['sparsity']
        loss_sep = losses_0.get('separation', torch.tensor(0.0, device=self.device))
        loss_conn = self.model.get_connectivity_loss(s_0, batch_0.edge_index)
        loss_ortho = self.model.get_ortho_loss(s_0)
        
        s_prev = s_0
        mu_prev = mu_0
        z_preds = [z_curr]
        
        # Only do forward dynamics if not in deep warmup
        if not is_warmup:
            for t in range(1, seq_len):
                if np.random.random() < tf_ratio:
                    batch_t_prev = Batch.from_data_list([data_list[t-1]]).to(self.device)
                    z_curr_forced, _, _, _ = self.model.encode(batch_t_prev.x, batch_t_prev.edge_index, batch_t_prev.batch, tau=tau)
                    z_curr = torch.nan_to_num(z_curr_forced)
                
                t_span = torch.tensor([0, dt], device=self.device, dtype=torch.float32)
                z_next_seq = self.model.forward_dynamics(z_curr, t_span)
                z_curr = torch.nan_to_num(z_next_seq[1])
                z_preds.append(z_curr)
        else:
            # During warmup, just use encoded states
            for t in range(1, seq_len):
                batch_t = Batch.from_data_list([data_list[t]]).to(self.device)
                z_t, _, _, _ = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch, tau=tau)
                z_preds.append(z_t)
            
        z_preds = torch.stack(z_preds)
        loss_l2 = torch.mean(z_preds**2)
        
        z_vel = (z_preds[1:] - z_preds[:-1]) / dt
        loss_lvr = torch.mean((z_vel[1:] - z_vel[:-1])**2) if len(z_vel) > 1 else torch.tensor(0.0, device=self.device)
        
        s_stability = 0
        mu_stability = 0
        loss_align = 0
        
        mu_mean = torch.tensor(self.stats['pos_mean'], device=self.device, dtype=torch.float32) if self.stats else 0
        mu_std = torch.tensor(self.stats['pos_std'], device=self.device, dtype=torch.float32) if self.stats else 1
        vel_mean = torch.tensor(self.stats['vel_mean'], device=self.device, dtype=torch.float32) if self.stats else 0
        vel_std = torch.tensor(self.stats['vel_std'], device=self.device, dtype=torch.float32) if self.stats else 1

        for t in range(seq_len):
            batch_t = Batch.from_data_list([data_list[t]]).to(self.device)
            z_t_target, s_t, losses_t, mu_t = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch, tau=tau)
            
            if torch.isnan(z_t_target).any():
                z_t_target = torch.nan_to_num(z_t_target)
            
            # Use z_preds for reconstruction to force consistency
            recon_t = self.model.decode(z_preds[t], s_t, batch_t.batch)
            
            loss_rec += self.criterion(recon_t, batch_t.x)
            loss_assign += losses_t['entropy'] + losses_t['diversity'] + 0.1 * losses_t['spatial']
            loss_pruning += losses_t['pruning']
            loss_sparsity += losses_t['sparsity']
            loss_sep += losses_t.get('separation', torch.tensor(0.0, device=self.device))
            loss_conn += self.model.get_connectivity_loss(s_t, batch_t.edge_index)
            loss_ortho += self.model.get_ortho_loss(s_t)
            
            # CoM Position Alignment
            mu_t_norm = (mu_t - mu_mean) / (mu_std + 1e-6)
            d_sub = self.model.encoder.latent_dim // 2 if self.model.hamiltonian else 2
            d_align = min(d_sub, 2) 
            
            p_align = self.criterion(z_preds[t, :, :, :d_align], mu_t_norm[:, :, :d_align])
            loss_align += align_weight * p_align

            if t > 0:
                s_diff = self.criterion(s_t, s_prev)
                mu_diff = self.criterion(mu_t, mu_prev)
                loss_assign += s_diff + 5.0 * mu_diff 
                if not is_warmup:
                    loss_cons += self.criterion(z_preds[t], z_t_target)
                s_stability += s_diff.item()
                mu_stability += mu_diff.item()
            s_prev = s_t
            mu_prev = mu_t
        
        # Normalization
        loss_rec /= seq_len
        loss_cons /= (seq_len - 1) if not is_warmup else 1
        loss_assign /= seq_len
        loss_pruning /= seq_len
        loss_sparsity /= seq_len
        loss_ortho /= seq_len
        loss_align /= seq_len
        loss_sep /= seq_len
        loss_conn /= seq_len
        s_stability /= (seq_len - 1)
        
        self.s_history.append(s_stability)
        if len(self.s_history) > self.max_s_history:
            self.s_history.pop(0)
        
        # Adaptive Loss Weighting
        lvars = torch.clamp(self.model.log_vars, min=-2.0, max=10.0)
        
        weights = {
            'rec': torch.exp(-lvars[0]), 'cons': torch.exp(-lvars[1]),
            'assign': torch.exp(-lvars[2]), 'ortho': torch.exp(-lvars[3]),
            'l2': torch.exp(-lvars[4]), 'lvr': torch.exp(-lvars[5]),
            'align': torch.exp(-lvars[6]), 'pruning': torch.exp(-lvars[7]),
            'sep': torch.exp(-lvars[8]), 'conn': torch.exp(-lvars[9]),
            'sparsity': torch.exp(-lvars[10])
        }

        discovery_loss = (weights['rec'] * loss_rec + lvars[0]) + \
                         (weights['assign'] * loss_assign + lvars[2]) + \
                         (weights['ortho'] * loss_ortho + lvars[3]) + \
                         (weights['align'] * loss_align + lvars[6]) + \
                         (weights['pruning'] * loss_pruning + lvars[7]) + \
                         (weights['sep'] * loss_sep + lvars[8]) + \
                         (weights['conn'] * loss_conn + lvars[9]) + \
                         (weights['sparsity'] * loss_sparsity + lvars[10])

        if is_warmup:
            loss = discovery_loss
        else:
            loss = discovery_loss + (weights['cons'] * loss_cons + lvars[1]) + \
                   (weights['l2'] * loss_l2 + lvars[4]) + \
                   (weights['lvr'] * loss_lvr + lvars[5])
        
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return 0.0, 0.0, 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item(), loss_rec.item(), loss_cons.item()

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
