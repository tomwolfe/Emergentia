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
        # Start at 1.0 * spring_dist to avoid overlap issues with LJ potential
        x = np.linspace(1.0 * spring_dist, side * spring_dist, side)
        y = np.linspace(1.0 * spring_dist, side * spring_dist, side)
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
            if self.n_particles <= 64:
                # Optimized vectorized brute force for small N
                # pos: [N, 2]
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # [N, N, 2]
                L = np.array(self.box_size)
                # Minimum Image Convention
                diff -= L * np.round(diff / L)
                dist_sq = np.sum(diff**2, axis=-1)
                
                # Use upper triangle indices to avoid self-loops and double counting
                idx1, idx2 = np.triu_indices(self.n_particles, k=1)
                mask = dist_sq[idx1, idx2] < self.radius**2
                return list(zip(idx1[mask], idx2[mask]))
            else:
                # For larger N, use tiling + KDTree for efficiency with PBC
                # Tile the points in 3x3 to cover all possible periodic neighbors
                L_x, L_y = self.box_size
                offsets = np.array([
                    [-L_x, -L_y], [-L_x, 0], [-L_x, L_y],
                    [0, -L_y],    [0, 0],    [0, L_y],
                    [L_x, -L_y],  [L_x, 0],  [L_x, L_y]
                ])
                
                all_pos = []
                original_indices = []
                for off in offsets:
                    all_pos.append(pos + off)
                    original_indices.append(np.arange(self.n_particles))
                
                all_pos = np.concatenate(all_pos, axis=0)
                original_indices = np.concatenate(original_indices, axis=0)
                
                # Query tree: find neighbors of original points (the middle tile) in all tiles
                tree = KDTree(all_pos)
                # Middle tile is at index offset 4 (0-based: 0,1,2,3,4...)
                start_idx = 4 * self.n_particles
                end_idx = 5 * self.n_particles
                
                # Use query_ball_point for each point in the middle tile
                results = tree.query_ball_point(pos, self.radius)
                
                pairs = set()
                for i, neighbors in enumerate(results):
                    for n_idx in neighbors:
                        j = original_indices[n_idx]
                        if i < j:
                            pairs.add((i, j))
                        elif j < i:
                            pairs.add((j, i))
                return list(pairs)
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

    def energy(self, pos=None, vel=None):
        if pos is None: pos = self.pos
        if vel is None: vel = self.vel
        
        # Kinetic Energy: 0.5 * m * v^2
        ke = 0.5 * self.m * np.sum(vel**2)
        
        # Potential Energy: 0.5 * k * (d - d0)^2
        pe = 0.0
        pairs = self._compute_pairs(pos) if (self.dynamic_radius or self.box_size) else self.fixed_pairs
        if len(pairs) > 0:
            idx1 = np.array([p[0] for p in pairs])
            idx2 = np.array([p[1] for p in pairs])
            diff = pos[idx2] - pos[idx1]
            if self.box_size:
                for i in range(2):
                    diff[:, i] -= self.box_size[i] * np.round(diff[:, i] / self.box_size[i])
            dist = np.linalg.norm(diff, axis=1)
            pe = 0.5 * self.k * np.sum((dist - self.spring_dist)**2)
            
        return ke + pe

    def step(self, sub_steps=2):  # Reduced sub_steps for faster simulation
        dt_sub = self.dt / sub_steps
        for _ in range(sub_steps):
            # Velocity Verlet:
            forces_t = self.compute_forces(self.pos)
            v_half = self.vel + (forces_t / self.m) * (dt_sub / 2.0)

            self.pos += v_half * dt_sub

            # Apply PBC wrap-around
            if self.box_size:
                self.pos = self.pos % self.box_size

            forces_next = self.compute_forces(self.pos)
            self.vel = v_half + (forces_next / self.m) * (dt_sub / 2.0)

            # 1. Global Energy Clamping (Proactive)
            # Prevent runaway kinetic energy which leads to divergence
            ke = 0.5 * self.m * np.sum(self.vel**2)
            max_ke = 1000.0 * self.n_particles
            if ke > max_ke:
                self.vel *= np.sqrt(max_ke / ke)

            # 2. Individual Particle Velocity Clamping
            max_v = 20.0  # Reduced max velocity for stability
            v_norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
            if np.any(v_norm > max_v):
                scale = np.where(v_norm > max_v, max_v / v_norm, 1.0)
                self.vel *= scale

        # Check for NaNs or massive values after all sub-steps
        if np.any(np.isnan(self.pos)) or np.any(np.abs(self.pos) > 1e6):
            print("Warning: Simulation diverged. Applying smooth recovery.")
            # Smooth recovery: gradually damp velocities and apply position corrections
            # instead of abruptly zeroing velocities
            if self.box_size:
                # Apply PBC wrap-around to keep positions in bounds
                self.pos = self.pos % self.box_size
                # Gradually damp velocities instead of zeroing them completely
                damping_factor = 0.7  # Reduced damping for better dynamics
                self.vel *= damping_factor
            else:
                # For non-PBC systems, clip positions and damp velocities
                self.pos = np.nan_to_num(self.pos)
                self.pos = np.clip(self.pos, -5, 5)
                # Apply gradual damping instead of zeroing
                damping_factor = 0.5  # Reduced damping
                self.vel *= damping_factor

        return self.pos.copy(), self.vel.copy()

    def generate_trajectory(self, steps=1000):
        trajectory_pos = []
        trajectory_vel = []
        for _ in range(steps):
            p, v = self.step()
            trajectory_pos.append(p)
            trajectory_vel.append(v)
        return np.array(trajectory_pos), np.array(trajectory_vel)

class LennardJonesSimulator(SpringMassSimulator):
    """
    Simulates particles interacting via the Lennard-Jones potential:
    V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
    """
    def __init__(self, n_particles=64, epsilon=1.0, sigma=1.0, m=1.0, dt=0.002, dynamic_radius=None, box_size=None):
        # LJ needs smaller dt for stability due to 1/r^12 term
        super().__init__(n_particles=n_particles, m=m, dt=dt, spring_dist=sigma, dynamic_radius=dynamic_radius, box_size=box_size)
        self.epsilon = epsilon
        self.sigma = sigma
        self.radius = 2.5 * sigma if not dynamic_radius else dynamic_radius

    def compute_forces(self, pos):
        forces = np.zeros_like(pos)
        pairs = self._compute_pairs(pos)

        if len(pairs) == 0:
            return forces

        idx1, idx2 = zip(*pairs)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)

        diff = pos[idx2] - pos[idx1]
        if self.box_size:
            for i in range(2):
                diff[:, i] -= self.box_size[i] * np.round(diff[:, i] / self.box_size[i])

        dist_sq = np.sum(diff**2, axis=1, keepdims=True)
        # Avoid division by zero and extreme values: floor at 0.7 * sigma^2 instead of 0.5
        dist_sq = np.maximum(dist_sq, 0.7 * self.sigma**2)

        sr6 = (self.sigma**2 / dist_sq)**3
        # Prevent overflow by clamping sr6 before squaring
        sr6 = np.clip(sr6, -1e10, 1e10)
        sr12 = sr6**2

        # F = 24 * epsilon / r^2 * [2*(sigma/r)^12 - (sigma/r)^6] * vec_r
        f_mag = (24 * self.epsilon / dist_sq) * (2 * sr12 - sr6)

        force_vec = f_mag * diff

        # Stability clamping - reduced max force for better stability
        max_f = 500.0  # Reduced max force for better stability
        force_vec = np.clip(force_vec, -max_f, max_f)

        np.add.at(forces, idx1, -force_vec)
        np.add.at(forces, idx2, force_vec)

        return forces

    def energy(self, pos=None, vel=None):
        if pos is None: pos = self.pos
        if vel is None: vel = self.vel
        
        ke = 0.5 * self.m * np.sum(vel**2)
        pe = 0.0
        pairs = self._compute_pairs(pos)
        if len(pairs) > 0:
            idx1, idx2 = zip(*pairs)
            idx1 = np.array(idx1)
            idx2 = np.array(idx2)
            diff = pos[idx2] - pos[idx1]
            if self.box_size:
                for i in range(2):
                    diff[:, i] -= self.box_size[i] * np.round(diff[:, i] / self.box_size[i])
            dist_sq = np.sum(diff**2, axis=1)
            sr6 = (self.sigma**2 / dist_sq)**3
            # Prevent overflow by clamping sr6 before squaring
            sr6 = np.clip(sr6, -1e10, 1e10)
            sr12 = sr6**2
            pe = 4 * self.epsilon * np.sum(sr12 - sr6)
            
        return ke + pe
