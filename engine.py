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
        # Fallback to computing from provided data if not given
        # Warning: This can lead to data leakage if used on test sets
        stats = compute_stats(pos, vel)
    
    pos_mean, pos_std = stats['pos_mean'], stats['pos_std']
    vel_mean, vel_std = stats['vel_mean'], stats['vel_std']
    
    pos_norm = (pos - pos_mean) / pos_std
    vel_norm = (vel - vel_mean) / vel_std
    
    dataset = []
    for t in range(T):
        curr_pos = pos[t]
        tree = KDTree(curr_pos)
        pairs = list(tree.query_pairs(radius))
        
        if len(pairs) > 0:
            idx1, idx2 = zip(*pairs)
            edges = np.array([idx1 + idx2, idx2 + idx1])
            edge_index = torch.tensor(edges, dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long, device=device)
        
        x = torch.cat([torch.tensor(pos_norm[t], dtype=torch.float, device=device), 
                       torch.tensor(vel_norm[t], dtype=torch.float, device=device)], dim=1)
        data = Data(x=x, edge_index=edge_index)
        dataset.append(data)
    return dataset, stats

class Trainer:
    def __init__(self, model, lr=1e-3, device='cpu', loss_weights=None, stats=None):
        self.model = model.to(device)
        self.device = device
        
        # MPS fix: torchdiffeq has issues with MPS (float64 defaults and stability)
        # We keep the ODE function on CPU while the rest of the model stays on MPS.
        if str(device) == 'mps':
            self.model.ode_func.to('cpu')
            
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        # Ensure criterion is on the same device as the model (majority of it)
        self.criterion = torch.nn.MSELoss().to(device)
        self.stats = stats
        
        # Pareto-optimal: Expose rigid hyperparameters
        default_weights = {
            'rec': 1.0,
            'cons': 5.0,
            'assign': 2.0,
            'latent_l2': 0.01  # Added to prevent state drift
        }
        if loss_weights is not None:
            default_weights.update(loss_weights)
        self.weights = default_weights

    def train_step(self, data_list, dt, epoch=0):
        self.optimizer.zero_grad()
        seq_len = len(data_list)
        
        # 1. Encode the first state
        batch_0 = Batch.from_data_list([data_list[0]]).to(self.device)
        z_0, s_0, entropy_0 = self.model.encode(batch_0.x, batch_0.edge_index, batch_0.batch)
        
        # 2. Predict full sequence in latent space
        t_span = torch.linspace(0, (seq_len - 1) * dt, seq_len, device=self.device, dtype=torch.float32)
        z_pred_seq = self.model.forward_dynamics(z_0, t_span)
        
        loss_rec = 0
        loss_cons = 0
        loss_assign = entropy_0 # Initialize with first entropy
        loss_l2 = torch.mean(z_pred_seq**2) # Penalize large latent values to ground the space
        
        s_prev = s_0
        
        for t in range(seq_len):
            batch_t = Batch.from_data_list([data_list[t]]).to(self.device)
            
            # Reconstruction loss at each step
            _, s_t, entropy_t = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch)
            recon_t = self.model.decode(z_pred_seq[t], s_t, batch_t.batch)
            loss_rec += self.criterion(recon_t, batch_t.x)
            
            # Assignment consistency loss: minimize change in super-node identity
            # AND diversity loss (entropy)
            loss_assign += entropy_t
            if t > 0:
                loss_assign += self.criterion(s_t, s_prev)
                # Consistency loss (encoding of real state vs predicted latent state)
                z_t_target, _, _ = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch)
                loss_cons += self.criterion(z_pred_seq[t], z_t_target)
            
            s_prev = s_t
        
        loss_rec /= seq_len
        loss_cons /= (seq_len - 1)
        loss_assign /= seq_len
        
        # Dynamic Loss Weighting: Faster annealing to ensure dynamics are learned early
        anneal = min(1.0, epoch / 800.0) if epoch > 50 else 0.1
        
        # Apply configurable weights
        # We increase the importance of consistency dynamically if it's lagging
        cons_weight = self.weights['cons']
        if loss_cons.item() > loss_rec.item() * 5: # Lowered threshold for boost
            cons_weight *= 2.0

        loss = (self.weights['rec'] * loss_rec + 
                anneal * (cons_weight * loss_cons + 
                         self.weights['latent_l2'] * loss_l2) +
                self.weights['assign'] * loss_assign)
        
        loss.backward()
        # Gradient Clipping to handle exploding gradients from ODE solver
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
