# CeNTREX-trajectories
Code for simulating CeNTREX trajectories

## Sections
The beamline is split into sections specified with `Section`, which can be used as follows:
```Python
sections = [
    fourK = Section(
        name = "4K shield",
        objects = [CircularAperture(x=0, y=0, z=5e-3)],
        start = 0,
        stop = 10e-2,
        save_collisions = False,
        propagation_type=PropagationType.ballistic,
    )
]
```
This defines a section called `4K shield`, which runs from `z = 0 m -> 10e-2 m`. Collisions aren't
saved, the `propagation_type` is ballistic and it contains a single circular aperture centered around
the z axis with a radius of 5 mm.

## Collision objects
Currently two type of apertures are defined for collisions:
* `CircularAperture(x: float,y: float,r: float)`
* `RectangularAperture(x: float,y: float,wx: float,wy: float)`

Custom collision objects can be defined; apertures should inherit from `Aperture`, and each custom
collision object should have two functions:
* `check_in_bounds(start: float, stop: float)` which returns a boolean specifying whether the object fully resides inside the section
* `get_acceptance(coordinates: Coordinates)` which returns a boolean arrays specifiying which trajectories make it through the aperture

## Propagation types
There is support for ballistic and ODE solver trajectories, which is specified on a per section basis through `Section.propagation_type`
* `PropagationType.ballistic` assumes a constant velocity and constant gravitational acceleration
* `PropagationType.ode` needs a defined force function in the section and uses `scipy.integrate.solve_ivp` to calculate the trajectory

## Working example
```Python
import numpy as np
from CeNTREX_trajectories import (
    CircularAperture,
    Coordinates,
    Velocities,
    Gravity,
    PropagationType,
    Section,
    TlF,
    propagate_trajectories,
    PropagationOptions,
)

in_to_m = 0.0254

fourK = Section(
    name = "4K shield",
    objects = [CircularAperture(x=0, y=0, z=1.75 * in_to_m, r = 1/2 * in_to_m)],
    start = 0,
    stop = 2 * in_to_m,
    save_collisions = False,
    propagation_type = PropagationType.ballistic,
)
fourtyK = Section(
    name = "40K shield",
    objects = [
        CircularAperture(x=0, y=0, z=fourK.stop + 1.25 * in_to_m, r = 1/2 * in_to_m)
    ],
    start = fourK.stop,
    stop = fourK.stop + 1.5 * in_to_m,
    save_collisions = False,
    propagation_type = PropagationType.ballistic
)
bbexit = Section(
    name = "Beamsource Exit",
    objects = [CircularAperture(0, 0, fourtyK.stop + 2.5 * in_to_m, 2 * in_to_m)],
    start = fourtyK.stop,
    stop = fourtyK.stop + 3.25 * in_to_m,
    save_collisions = False,
    propagation_type = PropagationType.ballistic
)

sections = [fourK, fourtyK, bbexit]

n_trajectories = 100_000
coordinates_init = Coordinates(
    x = np.random.randn(n_trajectories) * 1.5e-3,
    y = np.random.randn(n_trajectories) * 1.5e-3,
    z = np.zeros(n_trajectories)
)
velocities_init = Velocities(
    vx = np.random.randn(n_trajectories) * 39.4,
    vy = np.random.randn(n_trajectories) * 39.4,
    vz = np.random.randn(n_trajectories) * 16 + 184
)

options = PropagationOptions(n_cores = 6, verbose=False)
gravity = Gravity(0, -9.81, 0)
particle = TlF()

section_data, trajectories = propagate_trajectories(
    sections, 
    coordinates_init, 
    velocities_init, 
    particle, 
    gravity=gravity, 
    options=options 
)
```