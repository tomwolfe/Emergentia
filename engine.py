import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
import numpy as np

def prepare_data(pos, vel, adj):
    # pos, vel: [T, N, 2]
    # adj: [N, N]
    T, N, _ = pos.shape
    edge_index = torch.tensor(np.argwhere(adj > 0).T, dtype=torch.long)
    
    dataset = []
    for t in range(T):
        # Feature: [x, y, vx, vy]
        x = torch.cat([torch.tensor(pos[t], dtype=torch.float), 
                       torch.tensor(vel[t], dtype=torch.float)], dim=1)
        data = Data(x=x, edge_index=edge_index)
        dataset.append(data)
    return dataset

class Trainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, data_list, dt):
        self.optimizer.zero_grad()
        
        # data_list: list of Data objects for a sequence of length T
        seq_len = len(data_list)
        
        # 1. Encode the first state
        batch_0 = Batch.from_data_list([data_list[0]])
        z_0 = self.model.encode(batch_0.x, batch_0.edge_index, batch_0.batch)
        
        # 2. Predict full sequence in latent space
        t_span = torch.linspace(0, (seq_len - 1) * dt, seq_len)
        z_pred_seq = self.model.forward_dynamics(z_0, t_span) # [seq_len, 1, n_super, latent]
        
        loss_rec = 0
        loss_cons = 0
        
        for t in range(seq_len):
            batch_t = Batch.from_data_list([data_list[t]])
            
            # Reconstruction loss at each step
            recon_t = self.model.decode(z_pred_seq[t])
            loss_rec += self.criterion(recon_t, batch_t.x.unsqueeze(0))
            
            # Consistency loss (encoding of real state vs predicted latent state)
            if t > 0:
                z_t_target = self.model.encode(batch_t.x, batch_t.edge_index, batch_t.batch)
                loss_cons += self.criterion(z_pred_seq[t], z_t_target)
        
        loss_rec /= seq_len
        loss_cons /= (seq_len - 1)
        
        # Weight consistency higher to force dynamics learning
        loss = loss_rec + 5.0 * loss_cons
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_rec.item(), loss_cons.item()

if __name__ == "__main__":
    n_particles = 16
    sim = SpringMassSimulator(n_particles=n_particles)
    pos, vel = sim.generate_trajectory(steps=200)
    dataset = prepare_data(pos, vel, sim.adj)
    
    model = DiscoveryEngineModel(n_particles=n_particles, n_super_nodes=4)
    trainer = Trainer(model)
    
    for epoch in range(100):
        # Pick a random starting point in the trajectory
        idx = np.random.randint(0, len(dataset) - 1)
        batch_data = [dataset[idx], dataset[idx+1]]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f} (Rec: {rec:.6f}, Cons: {cons:.6f})")
