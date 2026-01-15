from __future__ import annotations
import numpy as np
from scipy.special import sph_harm, genlaguerre, factorial


def cartesian_to_spherical(X, Y, Z):
    """
    Returns r, theta, phi in physics convention:
      r >= 0
      theta in [0, pi] (polar angle from +z)
      phi in [0, 2pi) (azimuth from +x toward +y)
    """
    r = np.sqrt(X * X + Y * Y + Z * Z)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.where(r > 0, Z / r, 1.0))
    phi = np.arctan2(Y, X)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return r, theta, phi


def radial_R_nl(n: int, l: int, r: np.ndarray, Z: float = 1.0, a0: float = 1.0):
    """
    Radial part R_{nl}(r) for hydrogen-like atom (atomic units by default: a0=1).
    Uses associated Laguerre polynomials.

    One common normalization:
      rho = 2 Z r / (n a0)
      R_{nl} = (2Z/(n a0))^(3/2) * sqrt((n-l-1)! / (2n (n+l)!))
               * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if l < 0 or l >= n:
        raise ValueError("l must satisfy 0 <= l < n")
    if Z <= 0:
        raise ValueError("Z must be > 0")

    rho = 2.0 * Z * r / (n * a0)

    k = n - l - 1
    alpha = 2 * l + 1
    L = genlaguerre(k, alpha)(rho)

    num = factorial(n - l - 1, exact=False)
    den = factorial(n + l, exact=False)

    pref = (2.0 * Z / (n * a0)) ** 1.5
    norm = np.sqrt(num / (2.0 * n * den))

    R = pref * norm * np.exp(-rho / 2.0) * (rho ** l) * L
    return R


def _real_spherical_harmonic(l: int, m: int, theta, phi):
    """
    Real-valued spherical harmonics (condon-shortley phase, matching scipy sph_harm).
    Useful for 'classic' real orbitals (px/py/pz etc).
    """
    if m == 0:
        return np.real(sph_harm(0, l, phi, theta))
    if m > 0:
        Yp = sph_harm(m, l, phi, theta)
        Ym = sph_harm(-m, l, phi, theta)
        return np.sqrt(2.0) * ((-1.0) ** m) * np.real(Yp)
    # m < 0
    mp = -m
    Yp = sph_harm(mp, l, phi, theta)
    return np.sqrt(2.0) * ((-1.0) ** mp) * np.imag(Yp)


def psi_hydrogen_spherical(
    n: int, l: int, m: int, r, theta, phi,
    *, nuclear_charge: float = 1.0, real_orbital: bool = True
):
    """
    Hydrogen-like wavefunction psi_{nlm}(r,theta,phi) = R_{nl}(r) Y_l^m(theta,phi).

    If real_orbital=True, returns a *real* orbital (real spherical harmonic basis).
    If False, returns complex basis (scipy.special.sph_harm).
    """
    R = radial_R_nl(n, l, r, Z=float(nuclear_charge))
    if real_orbital:
        Y = _real_spherical_harmonic(l, m, theta, phi)  # real
        return R * Y
    Yc = sph_harm(m, l, phi, theta)  # complex
    return R * Yc


def psi_hydrogen_cartesian(
    n: int, l: int, m: int, X, Y, Z,
    *, nuclear_charge: float = 1.0, real_orbital: bool = True
):
    r, theta, phi = cartesian_to_spherical(X, Y, Z)
    return psi_hydrogen_spherical(
        n, l, m, r, theta, phi, nuclear_charge=nuclear_charge, real_orbital=real_orbital
    )
