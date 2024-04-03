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
