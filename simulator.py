from scipy.spatial import KDTree
import numpy as np
import torch

class SpringMassSimulator:
    def __init__(self, n_particles=64, k=10.0, m=1.0, dt=0.01, spring_dist=1.0, dynamic_radius=None, box_size=None):
        self.n_particles = n_particles
        self.k = k
        self.m = m
        self.dt = dt
        self.spring_dist = spring_dist
        self.dynamic_radius = dynamic_radius
        self.box_size = box_size # tuple (L_x, L_y) or None
        
        # Initialize particles in a grid
        side = int(np.ceil(np.sqrt(n_particles)))
        x = np.linspace(0.1 * spring_dist, side * spring_dist, side)
        y = np.linspace(0.1 * spring_dist, side * spring_dist, side)
        xv, yv = np.meshgrid(x, y)
        self.pos = np.stack([xv.flatten()[:n_particles], yv.flatten()[:n_particles]], axis=1)
        # Random initial velocity
        self.vel = np.random.randn(n_particles, 2) * 0.5
        
        # Create initial adjacency info
        self.radius = self.dynamic_radius if self.dynamic_radius else 1.1 * self.spring_dist
        if not self.dynamic_radius:
            self.fixed_pairs = self._compute_pairs(self.pos)

    def _compute_pairs(self, pos):
        if self.box_size:
            # For PBC, we use a simple distance search with wrap-around
            # KDTree doesn't natively support PBC easily without manual tiling or custom distance
            # For 80/20 we'll use brute force or just ignore KDTree for PBC if n is small
            # but let's try a simple approach: tiling 3x3 if needed, or just brute force
            if self.n_particles <= 128:
                # Brute force is fine for small n
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                for i in range(2):
                    diff[:, :, i] -= self.box_size[i] * np.round(diff[:, :, i] / self.box_size[i])
                dist = np.linalg.norm(diff, axis=-1)
                idx1, idx2 = np.where((dist < self.radius) & (np.arange(self.n_particles)[:, None] < np.arange(self.n_particles)[None, :]))
                return list(zip(idx1, idx2))
            else:
                tree = KDTree(pos)
                return list(tree.query_pairs(self.radius))
        else:
            tree = KDTree(pos)
            return list(tree.query_pairs(self.radius))

    def compute_forces(self, pos):
        forces = np.zeros_like(pos)
        pairs = self._compute_pairs(pos) if (self.dynamic_radius or self.box_size) else self.fixed_pairs
        
        if len(pairs) == 0:
            return forces

        idx1, idx2 = zip(*pairs)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        
        diff = pos[idx2] - pos[idx1]
        
        # Apply Minimum Image Convention for PBC
        if self.box_size:
            for i in range(2):
                diff[:, i] -= self.box_size[i] * np.round(diff[:, i] / self.box_size[i])

        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist_clamped = np.maximum(dist, 0.2 * self.spring_dist)
        
        f_mag = self.k * (dist - self.spring_dist)
        repulsion = 0.5 * self.k * np.power(self.spring_dist / dist_clamped, 4)
        f_mag -= repulsion
        
        force_vec = f_mag * (diff / (dist + 1e-9))
        
        max_f = 500.0
        force_vec = np.clip(force_vec, -max_f, max_f)
        
        np.add.at(forces, idx1, force_vec)
        np.add.at(forces, idx2, -force_vec)
        
        return forces

    def step(self):
        # Velocity Verlet:
        forces_t = self.compute_forces(self.pos)
        v_half = self.vel + (forces_t / self.m) * (self.dt / 2.0)
        
        self.pos += v_half * self.dt
        
        # Apply PBC wrap-around
        if self.box_size:
            self.pos = self.pos % self.box_size
        
        forces_next = self.compute_forces(self.pos)
        self.vel = v_half + (forces_next / self.m) * (self.dt / 2.0)
        
        # Stability check & soft-clamping
        max_v = 20.0
        v_norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
        if np.any(v_norm > max_v):
            scale = np.where(v_norm > max_v, max_v / v_norm, 1.0)
            self.vel *= scale
        
        # Check for NaNs or massive values
        if np.any(np.isnan(self.pos)) or np.any(np.abs(self.pos) > 1e6):
            print("Warning: Simulation diverged. Re-centering and damping.")
            self.pos = np.nan_to_num(self.pos)
            self.pos = np.clip(self.pos, -100, 100)
            self.vel *= 0.1

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
