from typing import Callable, Protocol, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

NDArray_or_Float = TypeVar("NDArray_or_Float", float, npt.NDArray[np.float64])

ForceType = Callable[[float, float, float, float], Tuple[float, float, float]]


class OdeResultLike(Protocol):
    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
