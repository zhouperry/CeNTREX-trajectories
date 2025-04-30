from collections.abc import Iterable
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .common_types import NDArray_or_Float
from .data_structures import Acceleration, Coordinates, Velocities
from .propagation_ballistic import calculate_time_ballistic
from .propagation_options import PropagationOptions

__all__: List[str] = []


def _C(ω: float, t: NDArray_or_Float) -> NDArray_or_Float:
    """Unified cosine helper (ordinary cos when ω ≠ 0, still fine when ω = 0)."""
    return np.cos(ω * t)


def _S(ω: float, t: NDArray_or_Float) -> NDArray_or_Float:
    """
    Unified sine/ω helper that is well-behaved as ω → 0.
    Returns  sin(ω t)/ω  for ω ≠ 0  and  t  for ω = 0.
    """
    return t if ω == 0 else np.sin(ω * t) / ω


def _pos(
    t: NDArray_or_Float, r0: float, v0: float, a: float, w: float
) -> NDArray_or_Float:
    """
    Position in one coordinate that may or may not have a spring (k).
    a  … constant acceleration component (F0 / m)
    """
    if w == 0:  # free, uniformly accelerated
        return r0 + v0 * t + 0.5 * a * t**2
    C = _C(w, t)
    S = _S(w, t)
    return r0 * C + v0 * S + a * (1.0 - C) / w**2


def _vel(
    t: NDArray_or_Float, r0: float, v0: float, a: float, w: float
) -> NDArray_or_Float:
    """
    Velocity in one coordinate — derivative of _pos().
    """
    if w == 0:  # free, uniformly accelerated
        return v0 + a * t
    return -w * r0 * np.sin(w * t) + v0 * _C(w, t) + a * _S(w, t)


def propagate_linear(
    t: NDArray_or_Float,
    origin: Coordinates,
    velocities: Velocities,
    acceleration: Acceleration,
    w: tuple[float, float, float],
    trap_center: tuple[float, float],
) -> tuple[Coordinates, Velocities]:
    wx, wy, wz = w
    assert wz == 0, "wz must be 0 for linear propagation"
    ax, ay, az = acceleration.ax, acceleration.ay, acceleration.az
    x_c, y_c = trap_center

    # Subtract trap center for spring displacement
    x = _pos(t, origin.x - x_c, velocities.vx, ax, wx) + x_c
    y = _pos(t, origin.y - y_c, velocities.vy, ay, wy) + y_c
    z = origin.z + velocities.vz * t + 0.5 * az * t**2

    vx = _vel(t, origin.x - x_c, velocities.vx, ax, wx)
    vy = _vel(t, origin.y - y_c, velocities.vy, ay, wy)
    vz = velocities.vz + az * t

    return (Coordinates(x, y, z), Velocities(vx, vy, vz))


def propagate_linear_trajectories(
    t_start: npt.NDArray[np.floating],
    origin: Coordinates,
    velocities: Velocities,
    objects: list,
    z_stop: float,
    acceleration: Acceleration = Acceleration(0.0, -9.81, 0.0),
    w: tuple[float, float, float] = (0.0, 0.0, 0.0),
    trap_center: tuple[float, float] = (0.0, 0.0),
    z_save: Optional[Union[List[float], npt.NDArray[np.floating]]] = None,
    save_collisions: bool = False,
    options: PropagationOptions = PropagationOptions(),
) -> Tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.floating],
    Coordinates,
    Velocities,
    int,
    List[Tuple[Coordinates, Velocities]],
]:
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

            x, v = propagate_linear(
                t, accepted_coords, accepted_velocities, acceleration, w, trap_center
            )
            hit, t_hit, x_hit, v_hit = obj.get_collisions_linear(
                accepted_coords, x, accepted_velocities, acceleration, w, trap_center
            )
            acceptance = ~hit
            if save_collisions:
                collisions.append((x_hit, v_hit))
        else:
            t = deepcopy(t_accepted)
            x, v = deepcopy(accepted_coords), deepcopy(accepted_velocities)
            try:
                acceptance = obj.get_acceptance(accepted_coords, x, v, acceleration)
            except AssertionError as error:
                raise AssertionError(f"{obj} -> {error.args[0]}")
            # why is this line here?
            acceptance &= ~np.isnan(t)
            if save_collisions:
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
            x, v = propagate_linear(dt, coords, vels, acceleration, w, trap_center)
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
    x, v = propagate_linear(dt, coords, vels, acceleration, w, trap_center)

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
