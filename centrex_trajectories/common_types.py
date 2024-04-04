from typing import TypeVar

import numpy as np
import numpy.typing as npt

NDArray_or_Float = TypeVar("NDArray_or_Float", npt.NDArray[np.float_], float)
