from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GridCoords:
    """Coordinate vectors for plotly and slicing."""
    x: np.ndarray  # shape (N,)
    y: np.ndarray  # shape (N,)
    z: np.ndarray  # shape (N,)
    extent: float
    N: int


def make_cartesian_grid(N: int, extent: float, *, dtype=np.float64):
    """
    Returns:
      X, Y, Z: 3D arrays shape (N,N,N)
      coords: GridCoords with 1D coordinate vectors
    """
    if N < 20:
        raise ValueError("Grid size N too small; use >= 50 for nicer visuals.")
    if extent <= 0:
        raise ValueError("extent must be positive.")

    x = np.linspace(-extent, extent, N, dtype=dtype)
    y = np.linspace(-extent, extent, N, dtype=dtype)
    z = np.linspace(-extent, extent, N, dtype=dtype)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    coords = GridCoords(x=x, y=y, z=z, extent=float(extent), N=int(N))
    return X, Y, Z, coords


def suggest_extent(n: int, Z: float, *, base_extent: float = 14.0) -> float:
    """Heuristic extent scaling: orbital size ~ n^2/Z (Bohr radii)."""
    Z = float(Z)
    if Z <= 0:
        Z = 1.0
    return float(base_extent * (n * n) / Z)
