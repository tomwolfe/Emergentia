import torch
import numpy as np
import sympy as sp

class PhysicalBasisRegistry:
    """
    Centralized registry for physical basis functions across different backends.
    Supported backends: 'torch', 'numpy', 'sympy'
    """
    
    _BASIS = {
        '1': {
            'torch': lambda d: torch.ones_like(d),
            'numpy': lambda d: np.ones_like(d),
            'sympy': lambda r: sp.Integer(1)
        },
        'r': {
            'torch': lambda d: d,
            'numpy': lambda d: d,
            'sympy': lambda r: r
        },
        '1/r': {
            'torch': lambda d: 1.0 / d,
            'numpy': lambda d: 1.0 / d,
            'sympy': lambda r: 1/r
        },
        '1/r^2': {
            'torch': lambda d: 1.0 / torch.pow(d, 2),
            'numpy': lambda d: 1.0 / np.power(d, 2),
            'sympy': lambda r: 1/r**2
        },
        '1/r^6': {
            'torch': lambda d: 1.0 / torch.pow(d, 6),
            'numpy': lambda d: 1.0 / np.power(d, 6),
            'sympy': lambda r: 1/r**6
        },
        '1/r^7': {
            'torch': lambda d: 1.0 / torch.pow(d, 7),
            'numpy': lambda d: 1.0 / np.power(d, 7),
            'sympy': lambda r: 1/r**7
        },
        '1/r^12': {
            'torch': lambda d: 1.0 / torch.pow(d, 12),
            'numpy': lambda d: 1.0 / np.power(d, 12),
            'sympy': lambda r: 1/r**12
        },
        '1/r^13': {
            'torch': lambda d: 1.0 / torch.pow(d, 13),
            'numpy': lambda d: 1.0 / np.power(d, 13),
            'sympy': lambda r: 1/r**13
        },
        'exp(-r)': {
            'torch': lambda d: torch.exp(-d),
            'numpy': lambda d: np.exp(-d),
            'sympy': lambda r: sp.exp(-r)
        }
    }

    @classmethod
    def get(cls, name, backend='torch'):
        if name not in cls._BASIS:
            raise ValueError(f"Unknown basis function: {name}")
        if backend not in cls._BASIS[name]:
            raise ValueError(f"Backend '{backend}' not supported for basis '{name}'")
        return cls._BASIS[name][backend]

    @classmethod
    def list_basis(cls):
        return list(cls._BASIS.keys())
    
    @classmethod
    def get_registry(cls, backend='torch'):
        return {name: cls.get(name, backend) for name in cls._BASIS.keys()}
