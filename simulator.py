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
        # Use a soft-core repulsion to prevent singularities
        dist_clamped = np.maximum(dist, 0.2 * self.spring_dist)
        
        f_mag = self.k * (dist - self.spring_dist)
        
        # Soft repulsive force at short distances to prevent overlap without explosion
        # LJ-like repulsion but smoother
        repulsion = 0.5 * self.k * np.power(self.spring_dist / dist_clamped, 4)
        f_mag -= repulsion
        
        force_vec = f_mag * (diff / dist)
        
        # Clamp forces to prevent explosion, but use a higher threshold than before
        max_f = 500.0
        force_vec = np.clip(force_vec, -max_f, max_f)
        
        # Use np.add.at for scattering forces back to particles
        np.add.at(forces, idx1, force_vec)
        np.add.at(forces, idx2, -force_vec)
        
        return forces

    def step(self):
        # Velocity Verlet:
        # 1. v(t + dt/2) = v(t) + (f(t)/m) * (dt/2)
        # 2. x(t + dt) = x(t) + v(t + dt/2) * dt
        # 3. f(t + dt) from x(t + dt)
        # 4. v(t + dt) = v(t + dt/2) + (f(t + dt)/m) * (dt/2)

        # Step 1
        forces_t = self.compute_forces(self.pos)
        v_half = self.vel + (forces_t / self.m) * (self.dt / 2.0)
        
        # Step 2
        self.pos += v_half * self.dt
        
        # Step 3
        forces_next = self.compute_forces(self.pos)
        
        # Step 4
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
