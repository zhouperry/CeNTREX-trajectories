from collections.abc import Iterable
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .data_structures import Coordinates, Gravity, Velocities, Force
from .propagation_options import PropagationOptions

from copy import deepcopy

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
        t = x / v
        if isinstance(t, np.ndarray):
            t[t < 0] = np.nan
        else:
            t = np.nan if t < 0 else t
        return t

    elif np.alltrue(a != 0):
        t1 = (np.sqrt(2 * a * x + v ** 2) - v) / a
        t2 = -(np.sqrt(2 * a * x + v ** 2) + v) / a
        t = np.zeros(t1.shape)
        m1 = t1 > 0
        m2 = t2 > 0
        t[m1] = t1[m1]
        t[m2] = t2[m2]
        t[~(m1 | m2)] = np.nan
        return t
    else:
        if isinstance(a, Iterable):
            t1 = np.array([(np.sqrt(2 * ai * x + v ** 2) - v) / ai for ai in a])
            t2 = -np.array([(np.sqrt(2 * ai * x + v ** 2) - v) / ai for ai in a])
            t = np.zeros(t1.shape)
            m1 = t1 > 0
            m2 = t2 > 0
            t[m1] = t1[m1]
            t[m2] = t2[m2]
            t[~(m1 | m2)] = np.nan
            return t
        else:
            t1 = (np.sqrt(2 * a * x + v ** 2) - v) / a
            t2 = -(np.sqrt(2 * a * x + v ** 2) + v) / a
            t = np.zeros(t1.shape)
            m1 = t1 > 0
            m2 = t2 > 0
            t[m1] = t1[m1]
            t[m2] = t2[m2]
            t[~(m1 | m2)] = np.nan
            return t


def propagate_ballistic_trajectories(
    t_start: npt.NDArray[np.float64],
    origin: Coordinates,
    velocities: Velocities,
    objects: List,
    z_stop: float,
    gravity: Gravity = Gravity(0.0, -9.81, 0.0),
    z_save: Optional[Union[List[float], npt.NDArray[np.float64]]] = None,
    save_collisions: bool = False,
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
    collisions = []
    indices = np.arange(origin.x.size)
    accepted_coords = deepcopy(origin)
    accepted_velocities = deepcopy(velocities)
    t_accepted = deepcopy(t_start)

    masks = []
    # iterate through apertures
    for obj in objects:
        # distance
        dz = obj.z_stop - accepted_coords.z
        # velocity
        vz = accepted_velocities.vz
        # gravity
        gz = gravity.gz
        t = calculate_time_ballistic(dz, vz, gz)
        if (
            calculate_time_ballistic(
                z_stop - accepted_coords.z[0], accepted_velocities.vz[0], gz
            )
            <= t[0]
        ):
            continue

        x, v = propagate_ballistic(t, accepted_coords, accepted_velocities, gravity)
        acceptance = obj.get_acceptance(
            accepted_coords, x, accepted_velocities, gravity
        )
        acceptance &= ~np.isnan(t)

        if save_collisions:
            if type(obj).__name__ == "PlateElectrodes":
                collisions.append(
                    obj.get_collisions(
                        accepted_coords, x, accepted_velocities, gravity
                    )[1:]
                )
            else:
                collisions.append(
                    (x.get_masked(~acceptance), v.get_masked(~acceptance))
                )

        accepted_coords = accepted_coords.get_masked(acceptance)
        accepted_velocities = accepted_velocities.get_masked(acceptance)
        t_accepted = t_accepted[acceptance]
        indices = indices[acceptance]

    survive = np.zeros(origin.x.size, dtype=bool)
    survive[indices] = True
    nr_collisions = survive.size - survive.sum()

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
    return (
        survive,
        t_accepted,
        accepted_coords,
        accepted_velocities,
        nr_collisions,
        collisions,
    )
