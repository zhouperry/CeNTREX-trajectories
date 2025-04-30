from collections.abc import Iterable
from copy import deepcopy
from typing import List, Optional, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

from .common_types import NDArray_or_Float
from .data_structures import Acceleration, Coordinates, Velocities
from .propagation_ballistic import calculate_time_ballistic
from .propagation_options import PropagationOptions

__all__: List[str] = []


def _C(ω: float, t: NDArray_or_Float) -> NDArray_or_Float:
    """
    Unified cosine helper function.

    Computes cos(ω * t) for ω ≠ 0 and handles the case where ω = 0 gracefully.

    Args:
        ω (float): Angular frequency.
        t (NDArray_or_Float): Time or array of time values.

    Returns:
        NDArray_or_Float: Computed cosine values.
    """
    return np.cos(ω * t)


def _S(ω: float, t: NDArray_or_Float) -> NDArray_or_Float:
    """
    Unified sine/ω helper function.

    Computes sin(ω * t) / ω for ω ≠ 0 and returns t for ω = 0.

    Args:
        ω (float): Angular frequency.
        t (NDArray_or_Float): Time or array of time values.

    Returns:
        NDArray_or_Float: Computed sine/ω values or t.
    """
    return t if ω == 0 else np.sin(ω * t) / ω


@overload
def _pos(
    t: float,
    r0: float,
    v0: float,
    a: float,
    w: float,
) -> float:
    ...

@overload
def _pos(
    t: npt.NDArray[np.floating],
    r0: Union[float, npt.NDArray[np.floating]],
    v0: Union[float, npt.NDArray[np.floating]],
    a: float,
    w: float,
) -> npt.NDArray[np.floating]:
    ...

def _pos(
    t: Union[float, npt.NDArray[np.floating]],
    r0: Union[float, npt.NDArray[np.floating]],
    v0: Union[float, npt.NDArray[np.floating]],
    a: float,
    w: float,
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Compute position in one coordinate with or without a spring force.

    Args:
        t (Union[float, npt.NDArray[np.floating]]): Time or array of time values.
        r0 (Union[float, npt.NDArray[np.floating]]): Initial position or array of positions.
        v0 (Union[float, npt.NDArray[np.floating]]): Initial velocity or array of velocities.
        a (float): Constant acceleration.
        w (float): Angular frequency of the spring.

    Returns:
        Same type as `t`: Position at time `t`.
    """
    if w == 0:  # free, uniformly accelerated
        return r0 + v0 * t + 0.5 * a * t**2
    C = _C(w, t)
    S = _S(w, t)
    return r0 * C + v0 * S + a * (1.0 - C) / w**2


@overload
def _vel(
    t: float,
    r0: float,
    v0: float,
    a: float,
    w: float,
) -> float:
    ...

@overload
def _vel(
    t: npt.NDArray[np.floating],
    r0: Union[float, npt.NDArray[np.floating]],
    v0: Union[float, npt.NDArray[np.floating]],
    a: float,
    w: float,
) -> npt.NDArray[np.floating]:
    ...

def _vel(
    t: Union[float, npt.NDArray[np.floating]],
    r0: Union[float, npt.NDArray[np.floating]],
    v0: Union[float, npt.NDArray[np.floating]],
    a: float,
    w: float,
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Compute velocity in one coordinate.

    This is the derivative of the position function `_pos`.

    Args:
        t (Union[float, npt.NDArray[np.floating]]): Time or array of time values.
        r0 (Union[float, npt.NDArray[np.floating]]): Initial position or array of positions.
        v0 (Union[float, npt.NDArray[np.floating]]): Initial velocity or array of velocities.
        a (float): Constant acceleration.
        w (float): Angular frequency of the spring.

    Returns:
        Same type as `t`: Velocity at time `t`.
    """
    if w == 0:  # free, uniformly accelerated
        return v0 + a * t
    return -w * r0 * np.sin(w * t) + v0 * _C(w, t) + a * _S(w, t)


def propagate_linear(
    t: Union[float, npt.NDArray[np.floating]],
    origin: Coordinates,
    velocities: Velocities,
    acceleration: Acceleration,
    w: Tuple[float, float, float],
    trap_center: Tuple[float, float],
) -> Tuple[Coordinates, Velocities]:
    """
    Propagate the position and velocity of an object linearly.

    Args:
        t (Union[float, npt.NDArray[np.floating]]): Time or array of time values.
        origin (Coordinates): Initial coordinates of the object.
        velocities (Velocities): Initial velocities of the object.
        acceleration (Acceleration): Constant acceleration acting on the object.
        w (Tuple[float, float, float]): Angular frequencies (wx, wy, wz).
        trap_center (Tuple[float, float]): Center of the spring trap.

    Returns:
        Tuple[Coordinates, Velocities]: Updated coordinates and velocities.
    """
    wx, wy, wz = w
    if wz != 0:
        raise ValueError("`wz` must be 0 for linear propagation")
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


def save_intermediate_states(
    z_save: Union[List[float], npt.NDArray[np.floating]],
    accepted_coords: Coordinates,
    accepted_velocities: Velocities,
    t_accepted: npt.NDArray[np.floating],
    acceleration: Acceleration,
    w: Tuple[float, float, float],
    trap_center: Tuple[float, float],
) -> npt.NDArray[np.floating]:
    """
    Save intermediate states of the trajectory at specified z positions.

    Args:
        z_save (Union[List[float], npt.NDArray[np.floating]]): Z positions to save intermediate states.
        accepted_coords (Coordinates): Current accepted coordinates.
        accepted_velocities (Velocities): Current accepted velocities.
        t_accepted (npt.NDArray[np.floating]): Current accepted time values.
        acceleration (Acceleration): Constant acceleration acting on the object.
        w (Tuple[float, float, float]): Angular frequencies (wx, wy, wz).
        trap_center (Tuple[float, float]): Center of the spring trap.

    Returns:
        npt.NDArray[np.floating]: Updated accepted time values.
    """
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
    return t_accepted


def propagate_linear_trajectories(
    t_start: npt.NDArray[np.floating],
    origin: Coordinates,
    velocities: Velocities,
    objects: List,
    z_stop: float,
    acceleration: Acceleration = Acceleration(0.0, -9.81, 0.0),
    w: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    trap_center: Tuple[float, float] = (0.0, 0.0),
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
    """
    Propagate trajectories linearly through a series of objects.

    Args:
        t_start (npt.NDArray[np.floating]): Initial time values.
        origin (Coordinates): Initial coordinates of the objects.
        velocities (Velocities): Initial velocities of the objects.
        objects (List): List of objects to propagate through.
        z_stop (float): Final z position to stop propagation.
        acceleration (Acceleration, optional): Constant acceleration. Defaults to gravity.
        w (Tuple[float, float, float], optional): Angular frequencies. Defaults to (0.0, 0.0, 0.0).
        trap_center (Tuple[float, float], optional): Center of the spring trap. Defaults to (0.0, 0.0).
        z_save (Optional[Union[List[float], npt.NDArray[np.floating]]], optional): Z positions to save intermediate states. Defaults to None.
        save_collisions (bool, optional): Whether to save collision data. Defaults to False.
        options (PropagationOptions, optional): Additional propagation options. Defaults to PropagationOptions().

    Returns:
        Tuple: A tuple containing:
            - survive (npt.NDArray[np.bool_]): Boolean array indicating surviving objects.
            - t_accepted (npt.NDArray[np.floating]): Accepted time values.
            - accepted_coords (Coordinates): Final accepted coordinates.
            - accepted_velocities (Velocities): Final accepted velocities.
            - nr_collisions (int): Number of collisions.
            - collisions (List[Tuple[Coordinates, Velocities]]): List of collision data.
    """
    if not isinstance(origin, Coordinates):
        raise TypeError("`origin` must be an instance of Coordinates.")
    if not isinstance(velocities, Velocities):
        raise TypeError("`velocities` must be an instance of Velocities.")
    if not isinstance(acceleration, Acceleration):
        raise TypeError("`acceleration` must be an instance of Acceleration.")

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
        t_accepted = save_intermediate_states(
            z_save, accepted_coords, accepted_velocities, t_accepted, acceleration, w, trap_center
        )

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
