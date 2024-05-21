from typing import List, Optional

import numpy as np

from .data_structures import Coordinates, Velocities

__all__: List[str] = [
    "generate_random_coordinates_normal_circle",
    "generate_random_coordinates_uniform_circle",
    "generate_random_velocities_normal",
]


def generate_random_coordinates_normal_circle(
    sigma: float,
    number: int,
    *,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    rng: Optional[np.random.Generator] = None,
) -> Coordinates:
    """
    Generate normally distributed samples over a circle of radius `radius` in the xy
    plane

    Args:
        sigma (float): sigma of the circle
        number (int): number of samples
        x (float, optional): central x coordinate. Defaults to 0.
        y (float, optional): central y coordinate. Defaults to 0.
        z (float, optional): central z coordinate. Defaults to 0.
        rng (Optional[np.random.Generator], optional): random number generator. Defaults
                                                        to None.

    Returns:
        Coordinates: randomly generated coordinates
    """
    if rng is None:
        rng = np.random.default_rng()

    return Coordinates(
        x=rng.normal(x, sigma, number),
        y=rng.normal(y, sigma, number),
        z=z * np.ones(number),
    )


def generate_random_coordinates_uniform_circle(
    radius: float,
    number: int,
    *,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    rng: Optional[np.random.Generator] = None,
) -> Coordinates:
    """
    Generate random coordinates uniformly over a circle

    Args:
        radius (float): radius of cicle
        number (int): number of samples
        x (float, optional): central x coordinate. Defaults to 0.
        y (float, optional): central y coordinate. Defaults to 0.
        z (float, optional): central z coordinate. Defaults to 0.
        rng (Optional[np.random.Generator], optional): random number generator. Defaults
                                                        to None.

    Returns:
        Coordinates: randomly generated coordinates
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = rng.uniform(0, 2 * np.pi, number)
    r = np.sqrt(rng.uniform(0, radius**2, number))
    return Coordinates(
        x=r * np.cos(theta) + x,
        y=r * np.sin(theta) + y,
        z=z * np.ones(number),
    )


def generate_random_velocities_normal(
    vx: float,
    vy: float,
    vz: float,
    sigma_vx: float,
    sigma_vy: float,
    sigma_vz: float,
    number: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Velocities:
    """
    Generate normally distributed velocity samples

    Args:
        vx (float): mean vx
        vy (float): mean vy
        vz (float): mean vz
        sigma_vx (float): sigma vx
        sigma_vy (float): sigma vy
        sigma_vz (float): sigma vz
        number (int): number of samples
        rng (Optional[np.random.Generator], optional): random number generator. Defaults
                                                        to None.

    Returns:
        Velocities: generated samples
    """
    if rng is None:
        rng = np.random.default_rng()
    return Velocities(
        vx=rng.normal(vx, sigma_vx, number),
        vy=rng.normal(vy, sigma_vy, number),
        vz=rng.normal(vz, sigma_vz, number),
    )
