from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from .data_structures import Coordinates, Force, Velocities
from .particles import TlF
from .propagation_options import PropagationOptions

__all__: List[str] = []


def z_stop_event_generator(
    z_stop: float,
) -> Callable[[float, npt.NDArray[np.float64]], float]:
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
        y: npt.NDArray[np.float64],
    ) -> float:
        return y[2] - z_stop

    # stop integration when z_stop is reached
    event.terminal = True  # type: ignore
    return event


def collision_event_generator(
    collision_events: Sequence[Callable[[float, float, float], float]],
) -> Sequence[Callable[[float, npt.NDArray[np.float64]], float]]:
    def create_event(collision_event):
        def event(t: float, y: npt.NDArray[np.float64]) -> float:
            return collision_event(y[0], y[1], y[2])

        event.terminal = True  # type: ignore
        return event

    return [create_event(collision_event) for collision_event in collision_events]


def solve_ode(
    t: float,
    x: Coordinates,
    v: Velocities,
    z_stop: float,
    mass: float,
    force: Callable[[float, float, float, float], Tuple[float, float, float]],
    force_cst: Force,
    events: List[Callable[[float, npt.NDArray[np.float64]], float]],
) -> OptimizeResult:
    """
    Solve the trajectory propagation ODE for a single trajectory

    Returns:
        OptimizeResult: solution of the trajectory ODE
    """
    t_span = [t, t + 2 * (z_stop - x.z) / v.vz]
    if t_span[1] <= t_span[0]:
        raise ValueError(
            "Invalid `t_span`: Ensure `z_stop` is greater than the initial z position."
        )
    z_stop_event = z_stop_event_generator(z_stop)
    p = [x.x, x.y, x.z, v.vx, v.vy, v.vz]
    _ode_fun = partial(
        ode_fun, **{"mass": mass, "force_fn": force, "force_cst": force_cst}
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
        force_cst (Force): constant force

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
        ax + force_cst.fx / mass,
        ay + force_cst.fy / mass,
        az + force_cst.fz / mass,
    )


def propagate_ODE_trajectories(
    t_start: npt.NDArray[np.float64],
    origin: Coordinates,
    velocities: Velocities,
    z_stop: float,
    mass: float,
    force_fun: Callable[[float, float, float, float], Tuple[float, float, float]],
    force_cst: Force = Force(0.0, -9.81 * TlF().mass, 0.0),
    events: Sequence[Callable[[float, float, float], float]] = [],
    z_save: Optional[
        Union[List[Union[float, int]], npt.NDArray[Union[np.float64, np.int_]]]
    ] = None,
    options: PropagationOptions = PropagationOptions(),
)-> Tuple[
    List[OptimizeResult],  # 1. The original list of solution objects
    np.ndarray,            # 2. Bulk array of final timestamps (N,)
    np.ndarray,            # 3. Bulk array of final coordinates (N, 3)
    np.ndarray             # 4. Bulk array of final velocities (N, 3)
]:
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
        List[OptimizeResult]: A list of solutions for each trajectory, where each solution
                              contains the time, position, and velocity data for the particle.Returns:
        Tuple[
            List[OptimizeResult]: A list of solutions for each trajectory.
            np.ndarray: 1D NumPy array of final timestamps. (Shape N,)
            np.ndarray: 2D NumPy array of final coordinates. (Shape N, 3)
            np.ndarray: 2D NumPy array of final velocities. (Shape N, 3)
        ]
    """
    if len(t_start) != len(origin) or len(origin) != len(velocities):
        raise ValueError(
            "`t_start`, `origin`, and `velocities` must have the same length."
        )

    collision_events = collision_event_generator(events)

    solutions = Parallel(n_jobs=options.n_cores, verbose=int(options.verbose))(
        delayed(solve_ode)(
            t, x, v, z_stop, mass, force_fun, force_cst, collision_events
        )
        for t, x, v in zip(t_start, origin, velocities)
    )
    if solutions is None:
        raise ValueError("No trajectories.")

    # Vectorize the results processing by pre-allocating arrays
    n_solutions = len(solutions)

    # Pre-allocate NumPy arrays for the *final* state of each trajectory
    final_t_np = np.empty(n_solutions, dtype=np.float64)
    final_coords_np = np.empty((n_solutions, 3), dtype=np.float64)
    final_vels_np = np.empty((n_solutions, 3), dtype=np.float64)

    # Fill arrays (this loop is fast, it's over N solutions, not N*S points)
    for i, sol in enumerate(solutions):
        # Get the last state from the ODE solution
        # sol.t[-1] is the final time
        # sol.y[:,-1] is the final state vector [x, y, z, vx, vy, vz]
        final_t_np[i] = sol.t[-1]
        final_coords_np[i] = sol.y[0:3, -1] # Final [x, y, z]
        final_vels_np[i] = sol.y[3:6, -1] # Final [vx, vy, vz]
    # --- END NEW SECTION ---

    # Return both the original solutions AND the new bulk arrays
    return solutions, final_t_np, final_coords_np, final_vels_np