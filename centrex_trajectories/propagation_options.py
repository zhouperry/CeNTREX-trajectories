from dataclasses import dataclass
from enum import Enum, auto
from typing import List

__all__: List[str] = ["PropagationType", "PropagationOptions"]


class PropagationType(Enum):
    """
    Enum to specify the propagation type

    Attributes
        ballistic (int): ballistic trajectory
        ode (int): ode trajectory
        linear (int): linear restoring force trajectory
    """

    ballistic = auto()
    ode = auto()
    linear = auto()


@dataclass
class PropagationOptions:
    """
    Dataclass to hold Propagation options

    Attributes
        n_cores (int): # cores used to solve ODE trajectories
        verbose (bool): enable verbose logging of joblib and others
    """

    n_cores: int = 6
    verbose: bool = True
