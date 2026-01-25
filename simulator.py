from scipy.spatial import KDTree
import numpy as np
import torch

class SpringMassSimulator:
    def __init__(self, n_particles=64, k=10.0, m=1.0, dt=0.001, spring_dist=1.0, dynamic_radius=None, box_size=None):
        self.n_particles = n_particles
        self.k = k
        self.m = m
        self.dt = dt
        self.spring_dist = spring_dist
        self.dynamic_radius = dynamic_radius
        self.box_size = box_size # tuple (L_x, L_y) or None

        # Initialize particles in a grid
        side = int(np.ceil(np.sqrt(n_particles)))
        # Start at 1.122 * spring_dist to avoid overlap issues with LJ potential
        # 1.122 is approx the minimum of the LJ potential (2^(1/6))
        x = np.linspace(1.122 * spring_dist, side * 1.122 * spring_dist, side)
        y = np.linspace(1.122 * spring_dist, side * 1.122 * spring_dist, side)
        xv, yv = np.meshgrid(x, y)
        self.pos = np.stack([xv.flatten()[:n_particles], yv.flatten()[:n_particles]], axis=1)
        # Random initial velocity
        self.vel = np.random.randn(n_particles, 2) * 1.0

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
                # O(N) Cell-List implementation for larger N
                L = np.array(self.box_size)
                # Cell size must be at least the interaction radius
                cell_size = self.radius
                n_cells = np.floor(L / cell_size).astype(int)
                n_cells = np.maximum(n_cells, 1)
                actual_cell_size = L / n_cells
                
                # Assign particles to cells
                cell_indices = np.floor(pos / actual_cell_size).astype(int) % n_cells
                
                # Build cell dictionary: (cell_x, cell_y) -> [particle_indices]
                cells = {}
                for i, (cx, cy) in enumerate(cell_indices):
                    cell_key = (cx, cy)
                    if cell_key not in cells:
                        cells[cell_key] = []
                    cells[cell_key].append(i)
                
                pairs = set()
                # Iterate over filled cells
                for (cx, cy), p_indices in cells.items():
                    # Check neighbor cells (3x3 area, including periodic neighbors)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = (cx + dx) % n_cells[0], (cy + dy) % n_cells[1]
                            neighbor_key = (nx, ny)
                            
                            if neighbor_key in cells:
                                n_p_indices = cells[neighbor_key]
                                
                                for i in p_indices:
                                    for j in n_p_indices:
                                        if i < j:
                                            # Check distance with MIC
                                            diff = pos[i] - pos[j]
                                            diff -= L * np.round(diff / L)
                                            if np.sum(diff**2) < self.radius**2:
                                                pairs.add((i, j))
                                        elif j < i and neighbor_key != (cx, cy):
                                            # If different cells, we already handled i < j case above? 
                                            # Wait, the 3x3 loop will visit each pair twice if cells are different.
                                            # Using i < j consistently across all neighbor checks is safer.
                                            pass
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

        # Soft floor for particle distances to prevent infinite forces
        # Use a soft minimum distance to prevent numerical instabilities
        min_dist = 0.1 * self.spring_dist
        dist_smooth = np.sqrt(dist**2 + min_dist**2)

        f_mag = self.k * (dist_smooth - self.spring_dist)
        repulsion = 0.5 * self.k * np.power(self.spring_dist / dist_smooth, 4)
        f_mag -= repulsion

        force_vec = f_mag * (diff / (dist_smooth + 1e-9))

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

    def generate_trajectory(self, steps=1000, init_pos=None, init_vel=None):
        """
        Generates a trajectory for the spring-mass system.
        """
        if init_pos is not None:
            self.pos = init_pos.copy()
        if init_vel is not None:
            self.vel = init_vel.copy()
            
        pos_history = [self.pos.copy()]
        vel_history = [self.vel.copy()]
        for _ in range(steps):
            p, v = self.step()
            pos_history.append(p)
            vel_history.append(v)
        return np.array(pos_history), np.array(vel_history)

class LennardJonesSimulator(SpringMassSimulator):
    """
    Simulates particles interacting via the Lennard-Jones potential:
    V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
    """
    def __init__(self, n_particles=64, epsilon=1.0, sigma=1.0, m=1.0, dt=0.0005, dynamic_radius=None, box_size=None, sub_steps=10):
        # LJ needs smaller dt for stability due to 1/r^12 term
        super().__init__(n_particles=n_particles, m=m, dt=dt, spring_dist=sigma, dynamic_radius=dynamic_radius, box_size=box_size)
        self.epsilon = epsilon
        self.sigma = sigma
        self.radius = 2.5 * sigma if not dynamic_radius else dynamic_radius
        self.sub_steps = sub_steps

    def step(self, sub_steps=None):
        if sub_steps is None:
            sub_steps = self.sub_steps
        return super().step(sub_steps=sub_steps)

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
        
        # IMPROVED: Smooth Soft-Core LJ implementation for better stability
        # Instead of hard clipping dist_sq, we add a softening parameter alpha.
        # V(r) = 4*epsilon * [ (sigma^2 / (r^2 + alpha*sigma^2))^6 - (sigma^2 / (r^2 + alpha*sigma^2))^3 ]
        alpha = 0.001  # Reduced softening for higher physical fidelity
        dist_sq_soft = dist_sq + alpha * (self.sigma**2)

        sr6 = (self.sigma**2 / dist_sq_soft)**3
        # Prevent overflow by clamping sr6 before squaring
        sr6 = np.clip(sr6, -1e10, 1e10) # Increased range
        sr12 = sr6**2

        # Force magnitude with soft-core correction
        # F = -dV/dr * (vec_r / r) = 24 * epsilon * [2*sr12 - sr6] * (vec_r / (r^2 + alpha*sigma^2))
        f_mag = (24 * self.epsilon / dist_sq_soft) * (2 * sr12 - sr6)

        force_vec = f_mag * diff

        # Stability clamping - increased for better fidelity but still safe
        max_f = 200.0  # Increased from 100.0
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
            
            # Use same soft-core softening as in compute_forces
            alpha = 0.001
            dist_sq_soft = dist_sq + alpha * (self.sigma**2)
            
            sr6 = (self.sigma**2 / dist_sq_soft)**3
            # Prevent overflow by clamping sr6 before squaring
            sr6 = np.clip(sr6, -1e10, 1e10)
            sr12 = sr6**2
            pe = 4 * self.epsilon * np.sum(sr12 - sr6)
            
        return ke + pe
