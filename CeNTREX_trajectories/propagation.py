import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .beamline_objects import ODESection, Section
from .data_structures import (
    Coordinates,
    Force,
    SectionData,
    Trajectories,
    Trajectory,
    Velocities,
)
from .particles import Particle
from .propagation_ballistic import propagate_ballistic_trajectories
from .propagation_ode import propagate_ODE_trajectories
from .propagation_options import PropagationOptions, PropagationType

__all__: List[str] = ["PropagationType", "propagate_trajectories", "PropagationOptions"]


def propagate_trajectories(
    sections: List[Union[Section, ODESection]],
    coordinates_init: Coordinates,
    velocities_init: Velocities,
    particle: Particle,
    t_start: Optional[npt.NDArray[np.float64]] = None,
    force: Force = Force(0.0, -9.81, 0.0),
    z_save: Optional[List] = None,
    options: PropagationOptions = PropagationOptions(),
) -> Tuple[List[SectionData], Trajectories]:
    """
    Propagate trajectories through sections starting at initial coordinates and initial
    velocities

    Args:
        sections (List): sections to propagate through
        coordinates_init (Coordinates): initial positions
        velocities_init (Velocities): initial velocities
        particle (Particle): particle to propagate
        t_start (Optional[npt.NDArray[np.float64]], optional): initial timestamps.
                                                                Defaults to None.
        force (Force, optional): Force. Defaults to Force(0.0, -9.81, 0.0), gravity.
        z_save (Optional[List], optional): z positions to save timestamps, coordinates
                                            and velocities. Defaults to None.
        options (PropagationOptions): Propagation options. Defaults to
                                                PropagationOptions().

    Returns:
        Tuple[List[SectionData], Trajectories]: return a list with the data per section
                                                stored as SectionData and the surviving
                                                trajectories
    """
    # initialize index array to keeps track of trajectory indices that make it through
    indices = np.arange(len(coordinates_init))

    # initializing classes holding the initial conditions
    coords_start = copy.deepcopy(coordinates_init)
    velocities_start = copy.deepcopy(velocities_init)
    if t_start is None:
        t_start = np.zeros(len(indices))
    else:
        t_start = copy.copy(t_start)

    # initialize 2D arrays for keeping track of the ballistic coordinates
    timestamps_tracked = t_start.copy()
    coordinates_tracked = copy.deepcopy(coordinates_init)
    velocities_tracked = copy.deepcopy(velocities_init)

    # list to store SectionData for each section
    section_data = []

    # class to hold trajectories
    trajectories = Trajectories()

    # propagate through sections
    for section in sections:
        if z_save is not None:
            # select z positions from z_save that are within the section
            z_save_section: Optional[List[float]] = [
                zs for zs in z_save if zs >= section.start and zs <= section.stop
            ]
        else:
            z_save_section = None
        # Initially when trajectories are propagated balistically they are stores in 2D
        # arrays, because particle takes the same number of steps when propagating
        # balistically. This is no longer true when propagating with an ODE solver, and
        # then storage is switched to Trajectories containing a single Trajectory for
        # each trajectory.
        # For performance this is only done after the first ODE section since the 2D
        # array storage and propagation method is much more performant

        # propagate ballistic if section is ballistic
        if section.propagation_type == PropagationType.ballistic:
            (
                mask,
                timestamp_list,
                coord_list,
                velocities_list,
                nr_collisions,
                collisions,
            ) = propagate_ballistic_trajectories(
                t_start,
                coords_start,
                velocities_start,
                section.objects,
                section.stop,
                force + section.force,
                z_save=z_save_section,
                save_collisions=section.save_collisions,
                options=options,
            )
            timestamps_tracked = timestamps_tracked[mask]
            coordinates_tracked = coordinates_tracked.get_masked(mask)
            velocities_tracked = velocities_tracked.get_masked(mask)
            indices = indices[mask]
            timestamps_tracked = np.column_stack([timestamps_tracked, timestamp_list])
            coordinates_tracked.column_stack(coord_list)
            velocities_tracked.column_stack(velocities_list)

            if len(trajectories) != 0:
                remove = [k for k in trajectories.keys() if k not in indices]
                trajectories.delete_trajectories(remove)

                for index, t, c, v in zip(
                    indices, timestamp_list, coord_list, velocities_list
                ):
                    trajectories.add_data(index, t, c, v)
            section_data.append(
                SectionData(section.name, collisions, nr_collisions, len(mask))
            )
            t_start = copy.copy(timestamp_list[:, -1])
            coords_start = coord_list.get_last()
            velocities_start = velocities_list.get_last()
        # propagate ODE if section is ODE
        elif section.propagation_type == PropagationType.ode:
            if len(trajectories) == 0:
                for index, t, c, v in zip(
                    indices, timestamps_tracked, coordinates_tracked, velocities_tracked
                ):
                    trajectories.add_data(index, t, c, v)
            timestamps_tracked = timestamps_tracked[:, -1]
            coordinates_tracked = coordinates_tracked.get_last()
            velocities_tracked = velocities_tracked.get_last()
            if isinstance(section, ODESection):
                force_fun = section.force
                force_cst = force
            else:

                def force_fun(
                    t: float, x: float, y: float, z: float
                ) -> Tuple[float, float, float]:
                    return (0.0, 0.0, 0.0)

                force_cst = force + section.force
            solutions = propagate_ODE_trajectories(
                t_start=t_start,
                origin=coords_start,
                velocities=velocities_start,
                z_stop=section.stop,
                mass=particle.mass,
                force_fun=force_fun,
                force_cst=force_cst,
                options=options,
            )
            coords = []
            velocities = []
            for sol, index in zip(solutions, indices):
                trajectories.add_data_ode(index, sol)
                coords.append([sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]])
                velocities.append([sol.y[3, -1], sol.y[4, -1], sol.y[5, -1]])
            coords_start = Coordinates(*np.array(coords).T)
            velocities_start = Velocities(*np.array(velocities).T)
            t_start = np.array(
                [trajectory.t[-1] for trajectory in trajectories.values()]
            )
            section_data.append(SectionData(section.name, [], 0, len(trajectories)))

    if len(trajectories) == 0:
        for index, t, c, v in zip(
            indices, timestamps_tracked, coordinates_tracked, velocities_tracked
        ):
            trajectories[index] = Trajectory(t, c, v, index)

    # remove coordinate entries in a trajectory
    for trajectory in trajectories.values():
        trajectory.remove_duplicate_entries()
    return section_data, trajectories
