import math
from typing import List

import numpy as np
import numpy.typing as npt

__all__: List[str] = []


def fit_stark_potential(
    electric_field: npt.NDArray[np.float64],
    stark_potential: npt.NDArray[np.float64],
    deg: int,
) -> npt.NDArray[np.float64]:
    """
    Fit a Stark Potential with a polynomial, ensuring the derivative at zero field is
    zero, e.g. the linear term of the potential is zero.

    Args:
        electric_field (npt.NDArray[np.float64]): electric field
        stark_potential (npt.NDArray[np.float64]): stark potential
        deg (int): polynomial degree

    Returns:
        npt.NDArray[np.float64]: polynomial coefficients
    """
    degrees = np.arange(deg + 1)
    degrees = degrees[degrees != 1]
    fit = np.polynomial.Polynomial.fit(
        x=electric_field, y=stark_potential, deg=degrees, domain=[]
    )
    return fit.coef

def bounds_check_tolerance(
    x: float,
    xmin: float,
    xmax: float,
    rel_tol: float = 1e-9,
    abs_tol: float = 0.0
) -> bool:
    """
    Check if x is within the bounds [xmin, xmax] with a relative or absolute tolerance.

    Args:
        x (float): Value to check.
        xmin (float): Minimum bound.
        xmax (float): Maximum bound.
        rel_tol (float, optional): Relative tolerance. Defaults to 1e-9.
        abs_tol (float, optional): Absolute tolerance. Defaults to 0.0.

    Returns:
        bool: True if x is within the bounds, False otherwise.
    """
    return (
        xmin <= x <= xmax or
        math.isclose(x, xmin, rel_tol=rel_tol, abs_tol=abs_tol) or
        math.isclose(x, xmax, rel_tol=rel_tol, abs_tol=abs_tol)
    )