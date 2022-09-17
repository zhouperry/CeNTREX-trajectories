from collections.abc import Iterable
from typing import Optional, Tuple, List

import numpy as np
import numpy.typing as npt

from .data_structures import Coordinates, Gravity, Velocities
from .propagation_options import PropagationOptions

__all__: List[str] = []


def propagate_ballistic(
    t: npt.NDArray[np.float64],
    origin: Coordinates,
    velocities: Velocities,
    gravity: Gravity,
) -> Tuple[Coordinates, Velocities]:
    """
    Propagate trajectories starting at `origin` with `velocities` for a time `t`

    Args:
        t (NDArray[np.float64]): time to propagate per particle
        origin (Coordinates): origin of each particle
        velocities (Velocities): velocity of each particle
        gravity (Gravity): gravitational accelleration

    Returns:
        Tuple[Coordinates, Velocities]: coordinates and velocities after propagating for
                            time `t`
    """
    return (
        Coordinates(
            origin.x + velocities.vx * t + 1 / 2 * gravity.gx * t ** 2,
            origin.y + velocities.vy * t + 1 / 2 * gravity.gy * t ** 2,
            origin.z + velocities.vz * t + 1 / 2 * gravity.gz * t ** 2,
        ),
        Velocities(
            velocities.vx + gravity.gx * t,
            velocities.vy + gravity.gy * t,
            velocities.vz + gravity.gz * t,
        ),
    )


def calculate_time_ballistic(
    x: npt.NDArray[np.float64], v: npt.NDArray[np.float64], a: float = 0.0
) -> npt.NDArray[np.float64]:
    """
    Calculate the time it takes a ballistic trajectory to travel a distance x

    Args:
        x (npt.NDArray[np.float64]): distance
        v (npt.NDArray[np.float64]): velocity
        a (float, optional): acceleration. Defaults to 0.0.

    Returns:
        npt.NDArray[np.float64]: time to travel distance x
    """
    if np.alltrue(a == 0):
        return x / v
    elif np.alltrue(a != 0):
        return (np.sqrt(2 * a * x + v ** 2) - v) / a
    else:
        if isinstance(a, Iterable):
            return np.array([(np.sqrt(2 * ai * x + v ** 2) - v) / ai for ai in a])
        else:
            return (np.sqrt(2 * a * x + v ** 2) - v) / a


def propagate_ballistic_trajectories(
    t_start: npt.NDArray[np.float64],
    origin: Coordinates,
    velocities: Velocities,
    apertures: List,
    z_stop: float,
    gravity: Gravity = Gravity(0.0, -9.81, 0.0),
    z_save: Optional[list] = None,
    options: PropagationOptions = PropagationOptions(),
) -> Tuple[
    npt.NDArray[np.bool_], npt.NDArray[np.float64], Coordinates, Velocities, int
]:
    """
    Propagate balistic trajectories. Stores the initial and final timestamps,
    coordinates and velocities and at z positions specified in z_save.

    Args:
        t_start (npt.NDArray[np.float64]): start times
        origin (Coordinates): origin coordintes
        velocities (Velocities): initial velocities
        apertures (List): apertures
        z_stop (float): position at which to stop trajectory propagation
        gravity (Gravity, optional): Gravity. Defaults to Gravity(0.0, -9.81, 0.0).
        z_save (Optional[list], optional): z positions to save coordinates and
                                            velocities at. Defaults to None.
        options (PropagationOptions, optional): Options for propagation. Defaults to
                                                PropagationOptions().

    Returns:
        Tuple[
            npt.NDArray[np.bool_], npt.NDArray[np.float64], Coordinates, Velocities, int
            ]: tuple with a boolean array of trajectories that survive, indices array
                with indices that survive, coordinates, velocities and nr_colisions
    """
    # create the survival mask array
    survive = np.ones(origin.x.shape, dtype=bool)

    # iterate through apertures
    for aperture in apertures:
        # distance
        dz = aperture.z - origin.z
        # velocity
        vz = velocities.vz
        # gravity
        gz = gravity.gz
        t = calculate_time_ballistic(dz, vz, gz)
        if calculate_time_ballistic(z_stop - origin.z[0], velocities.vz[0], gz) <= t[0]:
            continue

        x, v = propagate_ballistic(t, origin, velocities, gravity)
        survive &= aperture.get_acceptance(x)
        survive &= ~np.isnan(t)

    nr_collisions = survive.size - survive.sum()

    t_accepted = t_start[survive]
    accepted_coords = origin.get_masked(survive)
    accepted_velocities = velocities.get_masked(survive)

    # save coordinates and velocities at z positions z_save if supplied
    if z_save is not None:
        for z in z_save:
            coords = accepted_coords.get_last()
            vels = accepted_velocities.get_last()
            t = (z - coords.z) / vels.vz
            x, v = propagate_ballistic(t, coords, vels, gravity)
            t_accepted = np.column_stack([t_accepted, t + t_start[survive]])
            accepted_coords.column_stack(x)
            accepted_velocities.column_stack(v)

    # stop timestamps, coordinates and velocities
    coords = accepted_coords.get_last()
    vels = accepted_velocities.get_last()
    t = (z_stop - coords.z) / vels.vz
    x, v = propagate_ballistic(t, coords, vels, gravity)

    t_accepted = np.column_stack([t_accepted, t + t_start[survive]])
    accepted_coords.column_stack(x)
    accepted_velocities.column_stack(v)
    return (survive, t_accepted, accepted_coords, accepted_velocities, nr_collisions)
