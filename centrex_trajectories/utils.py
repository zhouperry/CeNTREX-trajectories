import math
from typing import Sequence

import numba as nb
import numpy as np
import numpy.typing as npt

from .common_types import NDArray_or_Float

__all__: list[str] = []


def fit_stark_potential(
    electric_field: npt.NDArray[np.float64],
    stark_potential: npt.NDArray[np.float64],
    deg: int,
) -> npt.NDArray[np.float64]:
    """Fit a Stark potential with a polynomial, ensuring the linear term is zero.

    Args:
        electric_field (np.ndarray): Electric field values.
        stark_potential (np.ndarray): Corresponding Stark potentials.
        deg (int): Desired polynomial degree.

    Returns:
        np.ndarray: Polynomial coefficients.
    """
    degrees = np.arange(deg + 1)
    degrees = degrees[degrees != 1]
    fit = np.polynomial.Polynomial.fit(
        x=electric_field, y=stark_potential, deg=degrees, domain=[]
    )
    return fit.coef


def bounds_check_tolerance(
    x: float, xmin: float, xmax: float, rel_tol: float = 1e-9, abs_tol: float = 0.0
) -> bool:
    """Check if x is within [xmin, xmax] using relative or absolute tolerance.

    Args:
        x (float): Value to check.
        xmin (float): Lower bound.
        xmax (float): Upper bound.
        rel_tol (float, optional): Relative tolerance. Defaults to 1e-9.
        abs_tol (float, optional): Absolute tolerance. Defaults to 0.0.

    Returns:
        bool: True if x is within bounds (within tolerance), False otherwise.
    """
    return (
        xmin <= x <= xmax
        or math.isclose(x, xmin, rel_tol=rel_tol, abs_tol=abs_tol)
        or math.isclose(x, xmax, rel_tol=rel_tol, abs_tol=abs_tol)
    )


@nb.njit
def richards(
    z: NDArray_or_Float,
    zc: float,
    y0: float,
    Δy: float,
    Q: float,
    B: float,
    ν: float,
) -> NDArray_or_Float:
    """Compute the Richards (generalized logistic) growth curve.

    The Richards function is a flexible S-shaped curve that can model asymmetric transitions.

    Args:
        z (NDArray_or_Float): Independent variable (e.g., position or time).
        y0 (float): Lower asymptote (minimum plateau) of the curve.
        Δy (float): Total rise (difference between upper and lower plateau).
        Q (float): Horizontal shift parameter affecting the curve’s midpoint.
        B (float): Growth rate parameter controlling steepness.
        zc (float): Midpoint (inflection point) of the transition.
        ν (float): Shape parameter controlling the asymmetry of the two tails.

    Returns:
        NDArray_or_Float: The Richards-curve value(s) at `z`.
    """
    return y0 + Δy / ((1 + Q * np.exp(-B * (z - zc))) ** (1 / ν))


def double_richards_symmetric(
    z: NDArray_or_Float,
    centers: Sequence[float],
    y0: float,
    dy: float,
    Q: float,
    B: float,
    nu: float,
) -> NDArray_or_Float:
    """
    Compute a symmetric double-inflection Richards curve.

    This function generates two generalized logistic (Richards) transitions of
    equal shape and amplitude `dy`, one upward at `centers[0]` and one downward
    at `centers[1]`, producing a flat plateau of height `y0 + dy` between them
    and returning to `y0` outside the transition regions.

    Args:
        z (NDArray_or_Float): Independent variable(s) where to evaluate the curve.
        centers (Sequence[float]): [c1, c2] where c1 is the inflection
            point of the rising segment and c2 is the inflection point of the
            falling segment. Must satisfy c1 < c2 for proper behavior.
        y0 (float): Baseline value before rise and after fall.
        dy (float): Amplitude of both the rising and falling segments.
        Q (float): Horizontal-shift parameter (same for both transitions).
        B (float): Growth-rate/steepness parameter (same for both).
        nu (float): Shape/asymmetry parameter (same for both).

    Returns:
        NDArray_or_Float: Curve values at `z`, rising then falling.

    Examples:
        >>> import numpy as np
        >>> z = np.linspace(-1, 3, 100)
        >>> y = double_richards_symmetric(z, [0, 2], 0.0, 1.0, 1.0, 5.0, 1.0)
        >>> # This creates a smooth transition from 0 to 1 at z=0 and back to 0 at z=2
    """
    # Validate inputs to prevent common errors
    if len(centers) != 2:
        raise ValueError(f"Expected 2 centers, got {len(centers)}")

    c1, c2 = centers
    if c1 >= c2:
        raise ValueError(f"Centers must satisfy c1 < c2, got {c1} and {c2}")

    # Use the richards() function for consistency and to avoid duplication
    r1 = richards(z, c1, 0.0, 1.0, Q, B, nu)
    r2 = richards(z, c2, 0.0, 1.0, Q, -B, nu)  # Note the -B for falling transition

    # Combine with plateau in between
    return y0 + dy * (r1 + r2 - 1.0)
