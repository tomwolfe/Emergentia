"""Mutation-hardening for neighbor-list helpers (test-only, no prod-logic change)."""
import torch
from emergentia.simulator import (
    HarmonicPotential,
    PhysicsSim,
    _cells_adjacent_2d,
    _flat_cell_index,
)


def test_cells_adjacent_2d_boundary_exact():
    # Adjacent: differ by at most 1 in EACH dimension.
    assert _cells_adjacent_2d(0, 0, 0, 0) is True
    assert _cells_adjacent_2d(0, 0, 1, 0) is True
    assert _cells_adjacent_2d(0, 0, 0, 1) is True
    assert _cells_adjacent_2d(0, 0, 1, 1) is True
    assert _cells_adjacent_2d(3, 5, 2, 4) is True
    # Non-adjacent: differ by >= 2 in ANY dimension.
    # (2, 0) also kills `and`->`or` mutants (or would say True).
    assert _cells_adjacent_2d(0, 0, 2, 0) is False
    assert _cells_adjacent_2d(0, 0, 0, 2) is False
    assert _cells_adjacent_2d(0, 0, 2, 2) is False
    assert _cells_adjacent_2d(0, 0, -2, 0) is False


def test_flat_cell_index_exact():
    # Exact arithmetic kills +/-/stride mutants in cell flattening.
    assert int(_flat_cell_index(0, 0, 4)) == 0
    assert int(_flat_cell_index(0, 3, 4)) == 3
    assert int(_flat_cell_index(1, 0, 4)) == 4
    assert int(_flat_cell_index(2, 1, 4)) == 9
    assert int(_flat_cell_index(1, 2, 7)) == 9


def _make_nl_sim_1d(cutoff, seed=0):
    return PhysicsSim(
        n=2, dim=1, potential=HarmonicPotential(k=10.0, r0=1.0),
        seed=seed, device=torch.device("cpu"),
        use_neighbor_list=True, cutoff=cutoff,
    )


def test_neighbor_list_1d_matches_raw_when_cutoff_covers_all():
    sim = _make_nl_sim_1d(cutoff=10.0)
    pos = torch.tensor([[0.0], [1.2]])
    assert torch.allclose(
        sim._compute_forces_neighbor_list(pos),
        sim._compute_forces_raw(pos), atol=1e-6,
    )


def test_neighbor_list_1d_boundary_inclusive():
    sim = _make_nl_sim_1d(cutoff=2.0)
    pos = torch.tensor([[0.0], [2.0]])
    f_nl = sim._compute_forces_neighbor_list(pos)
    assert torch.allclose(f_nl, sim._compute_forces_raw(pos), atol=1e-6)
    assert f_nl.abs().max().item() > 1.0


def test_neighbor_list_1d_excludes_beyond_cutoff():
    sim = _make_nl_sim_1d(cutoff=2.0)
    pos = torch.tensor([[0.0], [2.5]])
    assert torch.allclose(
        sim._compute_forces_neighbor_list(pos),
        torch.zeros(2, 1), atol=1e-6,
    )
    assert sim._compute_forces_raw(pos).abs().max().item() > 1.0
