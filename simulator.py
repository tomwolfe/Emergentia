from scipy.spatial import KDTree
import numpy as np
import torch

class SpringMassSimulator:
    def __init__(self, n_particles=64, k=10.0, m=1.0, dt=0.01, spring_dist=1.0, dynamic_radius=None):
        self.n_particles = n_particles
        self.k = k
        self.m = m
        self.dt = dt
        self.spring_dist = spring_dist
        self.dynamic_radius = dynamic_radius
        
        # Initialize particles in a grid
        side = int(np.ceil(np.sqrt(n_particles)))
        x = np.linspace(0, side * spring_dist, side)
        y = np.linspace(0, side * spring_dist, side)
        xv, yv = np.meshgrid(x, y)
        self.pos = np.stack([xv.flatten()[:n_particles], yv.flatten()[:n_particles]], axis=1)
        # Random initial velocity
        self.vel = np.random.randn(n_particles, 2) * 0.5
        
        # Create initial adjacency info
        self.radius = self.dynamic_radius if self.dynamic_radius else 1.1 * self.spring_dist
        if not self.dynamic_radius:
            self.fixed_pairs = self._compute_pairs(self.pos)

    def _compute_pairs(self, pos):
        tree = KDTree(pos)
        # query_pairs returns a set of (i, j) with i < j
        return list(tree.query_pairs(self.radius))

    def compute_forces(self, pos):
        forces = np.zeros_like(pos)
        pairs = self._compute_pairs(pos) if self.dynamic_radius else self.fixed_pairs
        
        if len(pairs) == 0:
            return forces

        # Vectorized force calculation for all pairs
        idx1, idx2 = zip(*pairs)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        
        diff = pos[idx2] - pos[idx1]
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        
        # Avoid division by zero and extremely small distances (repulsion)
        dist = np.maximum(dist, 0.01)
        
        f_mag = self.k * (dist - self.spring_dist)
        
        # Add a small repulsive force at very short distances to prevent overlap
        repulsion = 0.1 / (dist**2)
        f_mag -= repulsion
        
        force_vec = f_mag * (diff / dist)
        
        # Clamp forces to prevent explosion
        max_f = 100.0
        force_vec = np.clip(force_vec, -max_f, max_f)
        
        # Use np.add.at for scattering forces back to particles
        np.add.at(forces, idx1, force_vec)
        np.add.at(forces, idx2, -force_vec)
        
        return forces

    def step(self):
        # Semi-implicit Euler
        forces = self.compute_forces(self.pos)
        self.vel += (forces / self.m) * self.dt
        
        # Clamp velocity for stability
        max_v = 10.0
        self.vel = np.clip(self.vel, -max_v, max_v)
        
        self.pos += self.vel * self.dt
        
        # Check for NaNs
        if np.any(np.isnan(self.pos)):
            print("Warning: Simulation diverged (NaN). Resetting velocity.")
            self.vel = np.zeros_like(self.vel)
            self.pos = np.nan_to_num(self.pos)

        return self.pos.copy(), self.vel.copy()

    def generate_trajectory(self, steps=1000):
        trajectory_pos = []
        trajectory_vel = []
        for _ in range(steps):
            p, v = self.step()
            trajectory_pos.append(p)
            trajectory_vel.append(v)
        return np.array(trajectory_pos), np.array(trajectory_vel)

if __name__ == "__main__":
    sim = SpringMassSimulator(n_particles=16)
    pos, vel = sim.generate_trajectory(steps=100)
    print(f"Trajectory shape: {pos.shape}")
