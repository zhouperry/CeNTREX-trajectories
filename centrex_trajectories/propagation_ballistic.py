from collections.abc import Iterable
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .data_structures import Acceleration, Coordinates, Velocities
from .propagation_options import PropagationOptions

__all__: List[str] = []


def propagate_ballistic(
    t: npt.NDArray[np.float_],
    origin: Coordinates,
    velocities: Velocities,
    acceleration: Acceleration,
) -> Tuple[Coordinates, Velocities]:
    """
    Propagate trajectories starting at `origin` with `velocities` for a time `t`

    Args:
        t (NDArray[np.float_]): time to propagate per particle
        origin (Coordinates): origin of each particle
        velocities (Velocities): velocity of each particle
        gravity (Gravity): gravitational accelleration

    Returns:
        Tuple[Coordinates, Velocities]: coordinates and velocities after propagating for
                            time `t`
    """
    return (
        Coordinates(
            origin.x + velocities.vx * t + 1 / 2 * acceleration.ax * t**2,
            origin.y + velocities.vy * t + 1 / 2 * acceleration.ay * t**2,
            origin.z + velocities.vz * t + 1 / 2 * acceleration.az * t**2,
        ),
        Velocities(
            velocities.vx + acceleration.ax * t,
            velocities.vy + acceleration.ay * t,
            velocities.vz + acceleration.az * t,
        ),
    )


def calculate_time_ballistic(
    dx: npt.NDArray[np.float_], v: npt.NDArray[np.float_], a: float = 0.0
) -> npt.NDArray[np.float_]:
    """
    Calculate the time it takes a ballistic trajectory to travel a distance x

    Args:
        dx (npt.NDArray[np.float64]): distance
        v (npt.NDArray[np.float64]): velocity
        a (float, optional): acceleration. Defaults to 0.0.

    Returns:
        npt.NDArray[np.float64]: time to travel distance x
    """
    if np.all(a == 0):
        t = dx / v
        if isinstance(t, np.ndarray):
            t[t < 0] = np.nan
        elif isinstance(t, (float, int)):
            t = np.nan if t < 0 else t
        return t

    elif np.all(a != 0):
        t1 = (np.sqrt(2 * a * dx + v**2) - v) / a
        t2 = -(np.sqrt(2 * a * dx + v**2) + v) / a
        t = np.zeros(t1.shape)
        m1 = t1 > 0
        m2 = t2 > 0
        t[m1] = t1[m1]
        t[m2] = t2[m2]
        t[~(m1 | m2)] = np.nan
        return t
    else:
        if isinstance(a, Iterable):
            t1 = np.array([(np.sqrt(2 * ai * dx + v**2) - v) / ai for ai in a])
            t2 = -np.array([(np.sqrt(2 * ai * dx + v**2) - v) / ai for ai in a])
            t = np.zeros(t1.shape)
            m1 = t1 > 0
            m2 = t2 > 0
            t[m1] = t1[m1]
            t[m2] = t2[m2]
            t[~(m1 | m2)] = np.nan
            return t
        else:
            t1 = (np.sqrt(2 * a * dx + v**2) - v) / a
            t2 = -(np.sqrt(2 * a * dx + v**2) + v) / a
            t = np.zeros(t1.shape)
            m1 = t1 > 0
            m2 = t2 > 0
            t[m1] = t1[m1]
            t[m2] = t2[m2]
            t[~(m1 | m2)] = np.nan
            return t


def propagate_ballistic_trajectories(
    t_start: npt.NDArray[np.float_],
    origin: Coordinates,
    velocities: Velocities,
    objects: List,
    z_stop: float,
    acceleration: Acceleration = Acceleration(0.0, -9.81, 0.0),
    z_save: Optional[Union[List[float], npt.NDArray[np.float_]]] = None,
    save_collisions: bool = False,
    options: PropagationOptions = PropagationOptions(),
) -> Tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.float_],
    Coordinates,
    Velocities,
    int,
    List[Tuple[Coordinates, Velocities]],
]:
    """
    Propagate balistic trajectories. Stores the initial and final timestamps,
    coordinates and velocities and at z positions specified in z_save.

    Args:
        t_start (npt.NDArray[np.float_]): start times
        origin (Coordinates): origin coordintes
        velocities (Velocities): initial velocities
        apertures (List): apertures
        z_stop (float): position at which to stop trajectory propagation
        gravity (Gravity, optional): Gravity. Defaults to Gravity(0.0, -9.81, 0.0).
        z_save (Optional[list], optional): z positions to save coordinates and
                                            velocities at. Defaults to None.
        save_collisions (bool): Defaults to False
        options (PropagationOptions, optional): Options for propagation. Defaults to
                                                PropagationOptions().

    Returns:
        Tuple[
            npt.NDArray[np.bool_], npt.NDArray[np.float_], Coordinates, Velocities, int,
            List[Tuple[Coordinates, Velocities]]
            ]: tuple with a boolean array of trajectories that survive, indices array
                with indices that survive, coordinates, velocities, nr_collisions and a
                list of tuples of coordinates and velocities for collisions with each
                succesive object
    """
    collisions = []
    indices = np.arange(origin.x.size)
    accepted_coords = deepcopy(origin)
    accepted_velocities = deepcopy(velocities)
    t_accepted = deepcopy(t_start)

    # iterate through apertures, saving coordinates and velocities at each object and
    # saving collisions if save_collisions == True
    for obj in objects:
        # distance
        dz = obj.z_stop - accepted_coords.z

        if not np.allclose(dz, 0):
            # velocity
            vz = accepted_velocities.vz
            # forward acceleration
            az = acceleration.az
            t = calculate_time_ballistic(dz, vz, az)

            x, v = propagate_ballistic(
                t, accepted_coords, accepted_velocities, acceleration
            )
        else:
            t = deepcopy(t_accepted)
            x, v = deepcopy(accepted_coords), deepcopy(accepted_velocities)
        try:
            acceptance = obj.get_acceptance(
                accepted_coords, x, accepted_velocities, acceleration
            )
        except AssertionError as error:
            raise AssertionError(f"{obj} -> {error.args[0]}")
        # why is this line here?
        acceptance &= ~np.isnan(t)

        if save_collisions:
            if hasattr(obj, "get_collisions"):
                collisions.append(
                    obj.get_collisions(
                        accepted_coords, x, accepted_velocities, acceleration
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
        for idz, z in enumerate(z_save):
            coords = accepted_coords.get_last()
            vels = accepted_velocities.get_last()
            dz = z - coords.z
            dt = calculate_time_ballistic(dz, vels.vz, acceleration.az)
            x, v = propagate_ballistic(dt, coords, vels, acceleration)
            if idz != 0:
                t_accepted = np.column_stack([t_accepted, t_accepted[:, -1] + dt])
            else:
                t_accepted = np.column_stack([t_accepted, t_accepted + dt])
            accepted_coords.column_stack(x)
            accepted_velocities.column_stack(v)

    # timestamps, coordinates and velocities at the end of the section
    coords = accepted_coords.get_last()
    vels = accepted_velocities.get_last()
    dt = calculate_time_ballistic(z_stop - coords.z, vels.vz, acceleration.az)
    x, v = propagate_ballistic(dt, coords, vels, acceleration)

    if z_save is not None and len(z_save) > 0:
        t_accepted = np.column_stack([t_accepted, t_accepted[:, -1] + dt])
    else:
        t_accepted = np.column_stack([t_accepted, t_accepted + dt])
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
