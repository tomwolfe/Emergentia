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
        
        # Create initial adjacency matrix
        self.adj = self._compute_adj(self.pos)

    def _compute_adj(self, pos):
        adj = np.zeros((self.n_particles, self.n_particles))
        radius = self.dynamic_radius if self.dynamic_radius else 1.1 * self.spring_dist
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < radius:
                    adj[i, j] = adj[j, i] = 1.0
        return adj

    def compute_forces(self, pos):
        forces = np.zeros_like(pos)
        adj = self._compute_adj(pos) if self.dynamic_radius else self.adj
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if adj[i, j] > 0:
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        force = self.k * (dist - self.spring_dist) * (diff / dist)
                        forces[i] += force
        return forces

    def step(self):
        # Semi-implicit Euler
        forces = self.compute_forces(self.pos)
        self.vel += (forces / self.m) * self.dt
        self.pos += self.vel * self.dt
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
