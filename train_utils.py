import torch
import numpy as np

class ImprovedEarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def robust_energy(sim, pos, vel):
    """
    Robust energy calculation with a soft-floor for particle distances to prevent singularities.
    """
    ke = 0.5 * sim.m * np.sum(vel**2)
    pe = 0.0
    pairs = sim._compute_pairs(pos)
    if len(pairs) > 0:
        idx1, idx2 = zip(*pairs)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        diff = pos[idx2] - pos[idx1]
        if sim.box_size:
            for i in range(2):
                diff[:, i] -= sim.box_size[i] * np.round(diff[:, i] / sim.box_size[i])
        
        dist_sq = np.sum(diff**2, axis=1)
        sigma = getattr(sim, 'sigma', 1.0)
        dist_sq = np.maximum(dist_sq, (0.9 * sigma)**2)
        
        if hasattr(sim, 'k'):
            pe = 0.5 * sim.k * np.sum((np.sqrt(dist_sq) - sim.spring_dist)**2)
        else: # Lennard-Jones
            r6 = (sigma**2 / dist_sq)**3
            pe = 4.0 * sim.epsilon * np.sum(r6**2 - r6)
            
    return ke + pe

def get_device():
    if torch.cuda.is_available(): return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

class SparsityScheduler:
    def __init__(self, start_epoch, end_epoch, start_val, end_val):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_val = start_val
        self.end_val = end_val
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.start_epoch:
            val = self.start_val
        elif self.current_epoch > self.end_epoch:
            val = self.end_val
        else:
            ratio = (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            val = self.start_val + ratio * (self.end_val - self.start_val)
        self.current_epoch += 1
        return val
