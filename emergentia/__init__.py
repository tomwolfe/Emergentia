"""Emergentia package. Lazy imports keep light submodules usable without heavy deps."""

_LAZY_ATTRS = {
    "PhysicsSim": (".simulator", "PhysicsSim"),
    "DiscoveryPipeline": (".engine", "DiscoveryPipeline"),
    "DiscoveryNet": (".models", "DiscoveryNet"),
    "TrajectoryScaler": (".models", "TrajectoryScaler"),
    "UnitChecker": (".unit_checker", "UnitChecker"),
    "is_dimensionally_consistent": (
        ".unit_checker",
        "is_dimensionally_consistent",
    ),
    "PhysicsPriorProvider": (".physics_priors", "PhysicsPriorProvider"),
    "AutoSmoother": (".preprocessing", "AutoSmoother"),
    "TrajectorySmoother": (".preprocessing", "TrajectorySmoother"),
    "GaussianSmoother": (".preprocessing", "GaussianSmoother"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr_name = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name, __name__), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
