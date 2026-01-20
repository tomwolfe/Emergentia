import numpy as np
import torch

class SpringMassSimulator:
    def __init__(self, n_particles=64, k=10.0, m=1.0, dt=0.01, spring_dist=1.0):
        self.n_particles = n_particles
        self.k = k
        self.m = m
        self.dt = dt
        self.spring_dist = spring_dist
        
        # Initialize particles in a grid
        side = int(np.ceil(np.sqrt(n_particles)))
        x = np.linspace(0, side * spring_dist, side)
        y = np.linspace(0, side * spring_dist, side)
        xv, yv = np.meshgrid(x, y)
        self.pos = np.stack([xv.flatten()[:n_particles], yv.flatten()[:n_particles]], axis=1)
        # Random initial velocity
        self.vel = np.random.randn(n_particles, 2) * 0.5
        
        # Create adjacency matrix (nearest neighbors in grid)
        self.adj = np.zeros((n_particles, n_particles))
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                dist = np.linalg.norm(self.pos[i] - self.pos[j])
                if dist < 1.1 * spring_dist:
                    self.adj[i, j] = self.adj[j, i] = 1.0

    def compute_forces(self, pos):
        forces = np.zeros_like(pos)
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if self.adj[i, j] > 0:
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
