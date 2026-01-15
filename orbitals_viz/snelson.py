from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def fibonacci_sphere(n: int) -> np.ndarray:
    """Even-ish points on a sphere using Fibonacci spiral. Returns (n,3) unit vectors."""
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.arange(n, dtype=np.float64)
    phi = (1 + 5**0.5) / 2  # golden ratio
    theta = 2 * np.pi * i / phi
    z = 1 - (2 * (i + 0.5) / n)
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def _orthonormal_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = n / np.linalg.norm(n)
    # pick a vector not parallel to n
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.85:
        a = np.array([0.0, 1.0, 0.0])
    u = a - np.dot(a, n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


def wavy_ring(
    center: np.ndarray,
    normal: np.ndarray,
    *,
    ring_radius: float = 1.0,
    wave_amplitude: float = 0.15,
    wave_number: int = 6,
    n_points: int = 360,
) -> np.ndarray:
    """Parametric 'wave ring' curve (n_points,3)."""
    u, v, n = _orthonormal_basis_from_normal(np.asarray(normal, dtype=np.float64))
    t = np.linspace(0.0, 2*np.pi, n_points, endpoint=True)
    base = (np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :])
    wobble = (wave_amplitude * np.sin(wave_number * t))[:, None] * n[None, :]
    pts = np.asarray(center, dtype=np.float64)[None, :] + ring_radius * base + wobble
    return pts


def make_snelson_figure(
    *,
    electrons: int = 1,
    shell_radius: float = 6.0,
    ring_radius: float = 1.8,
    wave_amplitude: float = 0.25,
    wave_number: int = 8,
    rings_points: int = 220,
    title: str = "Snelson-style wave-polyhedron (illustrative)",
    show_nucleus: bool = True,
):
    """
    IMPORTANT: This is an *illustrative* geometric model inspired by Snelson's ring/wave shell idea,
    not a standard quantum-mechanical orbital.

    We place 'electron rings' on a sphere using a Fibonacci distribution (works for any N),
    and draw each ring as a wavy closed curve.
    """
    dirs = fibonacci_sphere(int(electrons))
    centers = shell_radius * dirs

    traces = []
    for i in range(centers.shape[0]):
        c = centers[i]
        n = dirs[i]  # ring plane normal = radial direction (halo-like)
        pts = wavy_ring(
            c, n,
            ring_radius=ring_radius,
            wave_amplitude=wave_amplitude,
            wave_number=int(wave_number),
            n_points=int(rings_points),
        )
        traces.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="lines",
            line=dict(width=4),
            name=f"e-{i+1}",
            hoverinfo="skip",
        ))

    if show_nucleus:
        # simple small sphere marker
        traces.append(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers",
            marker=dict(size=6),
            name="nucleus",
            hovertemplate="ядро<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig
