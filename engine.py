import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
import numpy as np
from scipy.spatial import KDTree

def prepare_data(pos, vel, radius=1.1, stats=None, device='cpu'):
    # pos, vel: [T, N, 2]
    T, N, _ = pos.shape
    
    if stats is None:
        # Standardize data based on the whole provided trajectory (default)
        # In production, stats should be fixed from training data
        pos_mean = pos.mean(axis=(0, 1))
        pos_std = pos.std(axis=(0, 1)) + 1e-6
        vel_mean = vel.mean(axis=(0, 1))
        vel_std = vel.std(axis=(0, 1)) + 1e-6
        stats = {'pos_mean': pos_mean, 'pos_std': pos_std, 
                 'vel_mean': vel_mean, 'vel_std': vel_std}
    else:
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
    def __init__(self, model, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, data_list, dt):
        self.optimizer.zero_grad()
        seq_len = len(data_list)
        
        # 1. Encode the first state
        batch_0 = Batch.from_data_list([data_list[0]]).to(self.device)
        z_0, s_0 = self.model.encode(batch_0.x, batch_0.edge_index, batch_0.batch)
        
        # 2. Predict full sequence in latent space
        t_span = torch.linspace(0, (seq_len - 1) * dt, seq_len, device=self.device, dtype=torch.float32)
        z_pred_seq = self.model.forward_dynamics(z_0, t_span)
        
        loss_rec = 0
        loss_cons = 0
        loss_assign = 0
        
        s_prev = s_0
        
        for t in range(seq_len):
            batch_t = Batch.from_data_list([data_list[t]]).to(self.device)
            
            # Reconstruction loss at each step
            # Use assignments from CURRENT state for predicting sequence reconstruction
            # to account for particle flow between super-nodes
            _, s_t = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch)
            recon_t = self.model.decode(z_pred_seq[t], s_t, batch_t.batch)
            loss_rec += self.criterion(recon_t, batch_t.x)
            
            # Assignment consistency loss: minimize change in super-node identity
            if t > 0:
                loss_assign += self.criterion(s_t, s_prev)
                # Consistency loss (encoding of real state vs predicted latent state)
                # Use the dynamic edge_index for encoding the target state
                z_t_target, _ = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch)
                loss_cons += self.criterion(z_pred_seq[t], z_t_target)
            
            s_prev = s_t
        
        loss_rec /= seq_len
        loss_cons /= (seq_len - 1)
        loss_assign /= (seq_len - 1)
        
        # Weight consistency and assignment stability
        loss = loss_rec + 5.0 * loss_cons + 2.0 * loss_assign
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_rec.item(), loss_cons.item()

if __name__ == "__main__":
    n_particles = 16
    spring_dist = 1.0
    sim = SpringMassSimulator(n_particles=n_particles, spring_dist=spring_dist, dynamic_radius=1.5)
    pos, vel = sim.generate_trajectory(steps=200)
    dataset, stats = prepare_data(pos, vel, radius=1.5)
    
    model = DiscoveryEngineModel(n_particles=n_particles, n_super_nodes=4)
    trainer = Trainer(model)
    
    for epoch in range(100):
        # Pick a random starting point in the trajectory
        idx = np.random.randint(0, len(dataset) - 1)
        batch_data = [dataset[idx], dataset[idx+1]]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f} (Rec: {rec:.6f}, Cons: {cons:.6f})")
