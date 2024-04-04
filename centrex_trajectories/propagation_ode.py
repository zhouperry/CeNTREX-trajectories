from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from .data_structures import Coordinates, Force, Velocities
from .propagation_options import PropagationOptions

__all__: List[str] = []


def z_stop_event_generator(
    z_stop: float,
) -> Callable[[float, npt.NDArray[np.float_]], float]:
    """
    Generate a terminal event function for solve_ivp that returns zero when z equals
    z_stop

    Args:
        z_stop (float): z coordinate at which to stop integration

    Returns:
        Callable: terminal event function
    """

    def event(
        t: float,
        y: npt.NDArray[np.float_],
    ) -> float:
        return y[2] - z_stop

    # stop integration when z_stop is reached
    event.terminal = True  # type: ignore
    return event


def collision_event_generator(
    collision_events: Sequence[Callable[[float, float, float], float]],
) -> Sequence[Callable[[float, npt.NDArray[np.float_]], float]]:
    events = []
    for collision_event in collision_events:

        def event(
            t: float,
            y: npt.NDArray[np.float_],
        ) -> float:
            return collision_event(y[0], y[1], y[2])

        event.terminal = True  # type: ignore
        events.append(event)
    return events


def solve_ode(*args) -> OptimizeResult:
    """
    Solve the trajectory propagation ODE for a single trajectory

    Returns:
        OptimizeResult: solution of the trajectory ODE
    """
    t, x, v, z_stop, mass, force, gravity, events = args
    t_span = [t, t + 2 * (z_stop - x.z) / v.vz]
    z_stop_event = z_stop_event_generator(z_stop)
    p = [x.x, x.y, x.z, v.vx, v.vy, v.vz]
    _ode_fun = partial(
        ode_fun, **{"mass": mass, "force_fn": force, "force_cst": gravity}
    )
    sol = solve_ivp(
        _ode_fun, t_span, p, events=events + [z_stop_event], rtol=1e-7, atol=1e-7
    )
    return sol


def ode_fun(
    t: float,
    d: List[float],
    mass: float,
    force_fn: Callable[[float, float, float, float], Tuple[float, float, float]],
    force_cst: Force,
) -> Tuple[float, float, float, float, float, float]:
    """
    General function describing the RHS of trajectory propagation

    Args:
        t (npt.NDArray[np.float64]): initial timestamps
        d (List[float]): list with x,y,z,vx,vy,vz
        mass (float): mass of particle
        force (Callable): function describing the force as a function of x,y,z
        gravity (Gravity): gravitational accelleration

    Returns:
        Union[List[npt.NDArray[np.float64]], List[float]]: RHS of trajectory propagation
                                                            ODE
    """
    x, y, z, vx, vy, vz = d
    fx, fy, fz = force_fn(t, x, y, z)
    ax = fx / mass
    ay = fy / mass
    az = fz / mass
    return (
        vx,
        vy,
        vz,
        ax + force_cst.fx,
        ay + force_cst.fy,
        az + force_cst.fz,
    )


def propagate_ODE_trajectories(
    t_start: npt.NDArray[np.float_],
    origin: Coordinates,
    velocities: Velocities,
    z_stop: float,
    mass: float,
    force_fun: Callable[[float, float, float, float], Tuple[float, float, float]],
    force_cst: Force = Force(0.0, -9.81, 0.0),
    events: Sequence[Callable[[float, float, float], float]] = [],
    z_save: Optional[
        Union[List[Union[float, int]], npt.NDArray[Union[np.float_, np.int_]]]
    ] = None,
    options: PropagationOptions = PropagationOptions(),
) -> List[OptimizeResult]:
    """
    propagate trajectories with an ODE solver

    Args:
        t_start (npt.NDArray[np.float64]): time at start of propagation
        origin (Coordinates): coordinates
        velocities (Velocities): velocities
        z_stop (float): z position to stop integration
        mass (float): mass of particle
        force_fun (Callable): function describing the force as a function of time and
                            position
        force_cst (Force, optional): force. Defaults to Force(0.0, -9.81, 0.0), gravity.
        z_save (Optional[
                    Union[
                        List[Union[float, int]],
                        npt.NDArray[Union[np.float64, np.int32]]
                        ]
                    ], optional): z positions to save coordinates and velocities .
                                    Defaults to None.
        options (PropagationOptions, optional): Propagation options for trajectories.
                                                Defaults to PropagationOptions().

    Returns:
        _type_: _description_
    """
    collision_events = collision_event_generator(events)

    solutions = Parallel(n_jobs=options.n_cores, verbose=int(options.verbose))(
        delayed(solve_ode)(
            t, x, v, z_stop, mass, force_fun, force_cst, collision_events
        )
        for t, x, v in zip(t_start, origin, velocities)
    )
    return solutions
