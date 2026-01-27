import torch
import pytest
from emergentia.models import TrajectoryScaler

def test_scaling_reversibility():
    scaler = TrajectoryScaler()
    p = torch.randn(100, 4, 2) * 5.0
    f = torch.randn(100, 4, 2) * 10.0
    
    scaler.fit(p, f)
    p_s, f_s = scaler.transform(p, f)
    
    # Check if scales are within [0, 1] approximately
    assert torch.max(torch.abs(p_s)) <= 1.0001
    assert torch.max(torch.abs(f_s)) <= 1.0001
    
    f_inv = scaler.inverse_transform_f(f_s)
    assert torch.allclose(f, f_inv, atol=1e-5)

def test_scaling_zero_motion():
    scaler = TrajectoryScaler()
    p = torch.zeros(10, 2, 2)
    f = torch.zeros(10, 2, 2)
    
    scaler.fit(p, f)
    p_s, f_s = scaler.transform(p, f)
    
    assert torch.all(p_s == 0)
    assert torch.all(f_s == 0)
    assert scaler.p_scale == 1.0
    assert scaler.f_scale == 1.0

def test_extreme_scales():
    scaler = TrajectoryScaler()
    p = torch.randn(10, 2, 2) * 1e-10
    f = torch.randn(10, 2, 2) * 1e10
    
    scaler.fit(p, f)
    p_s, f_s = scaler.transform(p, f)
    
    assert torch.max(torch.abs(p_s)) <= 1.0001
    assert torch.max(torch.abs(f_s)) <= 1.0001
    
    f_inv = scaler.inverse_transform_f(f_s)
    assert torch.allclose(f, f_inv)
