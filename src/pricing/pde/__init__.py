"""Finite-difference PDE engine: schemes, American exercise, studies, LSM."""

from pricing.pde.grid import Grid, build_grid
from pricing.pde.lsm import lsm_american
from pricing.pde.schemes import (
    SCHEMES,
    cfl_dt,
    fd_american,
    fd_price,
    fd_solution,
)
from pricing.pde.studies import (
    early_exercise_order,
    richardson_extrapolate,
    spatial_order,
    temporal_order,
)

__all__ = [
    "SCHEMES",
    "Grid",
    "build_grid",
    "cfl_dt",
    "early_exercise_order",
    "fd_american",
    "fd_price",
    "fd_solution",
    "lsm_american",
    "richardson_extrapolate",
    "spatial_order",
    "temporal_order",
]
