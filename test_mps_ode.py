import torch
import torch.nn as nn
from torchdiffeq import odeint
import time

device = torch.device('mps')
print(f"Testing on {device}")

class SimpleFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 10), nn.Tanh())
    def forward(self, t, y):
        return self.net(y)

func = SimpleFunc().to(device)
y0 = torch.randn(100, 10).to(device)
t = torch.linspace(0, 1, 10).to(device)

print("Starting odeint on MPS...")
start = time.time()
try:
    with torch.no_grad():
        out = odeint(func, y0, t, method='rk4', options={'step_size': 0.1})
    print(f"Success! Time: {time.time() - start:.4f}s")
    print(f"Output shape: {out.shape}")
except Exception as e:
    print(f"Failed: {e}")

print("\nStarting odeint on CPU for comparison...")
func_cpu = SimpleFunc().to('cpu')
y0_cpu = y0.to('cpu')
t_cpu = t.to('cpu')
start = time.time()
out_cpu = odeint(func_cpu, y0_cpu, t_cpu, method='rk4', options={'step_size': 0.1})
print(f"Success! Time: {time.time() - start:.4f}s")
