"""
Trajectory Propagation Framework
=================================

This module provides the high-level interface for propagating molecular beam trajectories
through complex beamline configurations. It implements an adaptive multi-method approach
that switches between different propagation strategies based on the physics of each section.

Propagation Methods
-------------------
1. **Ballistic**: Constant acceleration (uniform fields, drift regions)
   - Uses analytical solutions: x(t) = x₀ + v₀t + ½at²
   - Efficient columnar array storage (all trajectories synchronized)
   - Collision detection via geometric intersection tests

2. **Linear (Harmonic)**: Linear restoring forces (ion guides, traps)
   - Uses analytical solutions: x(t) = A·sin(ωt) + B·cos(ωt)
   - Angular frequency ω = √(k/m) from spring constant
   - Efficient for transverse confinement with axial drift

3. **ODE**: Position-dependent forces (electrostatic lenses, gradients)
   - Uses adaptive step-size ODE integration (scipy.integrate.solve_ivp)
   - Switches to per-trajectory storage (different step counts)
   - Event detection for collisions with objects

Architecture
------------
The propagation engine uses a hybrid storage strategy for optimal performance:
- **2D Arrays**: Used for ballistic/linear sections (same steps for all trajectories)
- **Trajectories Dict**: Used during/after ODE sections (variable steps per trajectory)

Automatic conversion between storage formats occurs at section boundaries, allowing
efficient propagation through mixed beamline configurations.

Key Functions
-------------
- `propagate_trajectories`: Main entry point for trajectory propagation
- `do_ballistic`: Handle ballistic propagation through a section
- `do_linear`: Handle linear (harmonic) propagation through a section

Performance Notes
-----------------
- Columnar (2D array) storage is ~10x faster than per-trajectory storage for ballistic
- ODE integration is ~100x slower than ballistic, use sparingly
- Pre-filtering trajectories by aperture acceptance reduces unnecessary ODE integrations
- Numba-accelerated collision detection provides significant speedup

See Also
--------
- propagation_ballistic: Low-level ballistic propagation implementation
- propagation_linear: Low-level linear/harmonic propagation implementation
- propagation_ode: Low-level ODE integration implementation
- propagation_options: Configuration options for propagation behavior
"""

import copy
import math
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import cupy as cp
import gc

from .beamline_objects import Bore, LinearSection, ODESection, Section
from .common_types import ForceType
from .data_structures import (
    Acceleration,
    Coordinates,
    Force,
    SectionData,
    Trajectories,
    Trajectory,
    Velocities,
)
from .particles import Particle, TlF
from .propagation_ballistic import propagate_ballistic_trajectories
from .propagation_linear import propagate_linear_trajectories
from .propagation_ode import propagate_ODE_trajectories
from .propagation_ode_vectorized import ode_fun_batch, rk4_batch_fixed_steps
from .propagation_options import PropagationOptions, PropagationType
from typing import cast
__all__: List[str] = ["PropagationType", "propagate_trajectories", "PropagationOptions"]


def do_ballistic(
    indices: npt.NDArray[np.int_],
    timestamps_tracked: npt.NDArray[np.float64],
    coordinates_tracked: Coordinates,
    velocities_tracked: Velocities,
    section: Union[Section, ODESection],
    particle: Particle,
    force: Force,
    z_save_section: Optional[Union[List[float], npt.NDArray[np.float64]]],
    options: PropagationOptions,
) -> Tuple[
    npt.NDArray[np.float64],
    Coordinates,
    Velocities,
    npt.NDArray[np.int_],
    SectionData,
]:
    """
    Propagate trajectories ballistically through a section.

    This function handles ballistic (constant acceleration) propagation where particles
    move under the influence of constant forces. It's more performant than ODE-based
    propagation and is used for drift sections or sections with uniform fields.

    Args:
        indices (npt.NDArray[np.int_]):
            Array of trajectory indices for tracking which particles survive.
        timestamps_tracked (npt.NDArray[np.float64]):
            Array of tracked timestamps. Can be 1D (initial) or 2D (accumulated).
        coordinates_tracked (Coordinates):
            Current positions of all tracked trajectories.
        velocities_tracked (Velocities):
            Current velocities of all tracked trajectories.
        trajectories (Trajectories):
            Container for detailed trajectory data (used after ODE sections).
        section (Union[Section, ODESection]):
            The section through which to propagate, containing geometry and objects.
        particle (Particle):
            The particle species being propagated (contains mass, etc.).
        force (Force):
            External force applied during propagation (e.g., gravity).
        z_save_section (Optional[Union[List[float], npt.NDArray[np.float64]]]):
            Z positions within this section where intermediate states should be saved.
        options (PropagationOptions):
            Configuration options for propagation behavior.

    Returns:
        tuple:
            A tuple containing:
            - Updated timestamps_tracked (npt.NDArray[np.float64]).
            - Updated coordinates_tracked (Coordinates).
            - Updated velocities_tracked (Velocities).
            - Updated indices (npt.NDArray[np.int_]).
            - SectionData for the section.
    """
    # Calculate total constant force and resulting acceleration
    # Section force is added to external force (e.g., gravity + static fields)
    force_cst = force + section.force
    acceleration = Acceleration(
        force_cst.fx / particle.mass,
        force_cst.fy / particle.mass,
        force_cst.fz / particle.mass,
    )
    # Propagate trajectories ballistically through the section
    # Extract last timestamp (handling both 1D initial and 2D accumulated arrays)
    (
        mask,
        timestamp_list,
        coord_list,
        velocities_list,
        nr_collisions,
        collisions,
    ) = propagate_ballistic_trajectories(
        timestamps_tracked
        if timestamps_tracked.ndim == 1
        else timestamps_tracked[:, -1],
        coordinates_tracked.get_last(),
        velocities_tracked.get_last(),
        section.objects,
        section.stop,
        acceleration,
        z_save=z_save_section,
        save_collisions=section.save_collisions,
        options=options,
    )

    # Filter out trajectories that collided with objects
    # mask is True for trajectories that survived
    timestamps_tracked = timestamps_tracked[mask]
    coordinates_tracked = coordinates_tracked.get_masked(mask)
    velocities_tracked = velocities_tracked.get_masked(mask)
    indices = indices[mask]

    # append latest timestamps, coordinates and velocities to the 2D arrays
    timestamps_tracked = cp.column_stack([timestamps_tracked, timestamp_list])
    coordinates_tracked.column_stack(coord_list)
    velocities_tracked.column_stack(velocities_list)

    # Create section statistics
    section_data = SectionData(section.name, collisions, nr_collisions, len(mask))

    return (
        timestamps_tracked,
        coordinates_tracked,
        velocities_tracked,
        indices,
        section_data,
    )


def do_linear(
    indices: npt.NDArray[np.int_],
    timestamps_tracked: npt.NDArray[np.float64],
    coordinates_tracked: Coordinates,
    velocities_tracked: Velocities,
    section: LinearSection,
    particle: Particle,
    force: Force,
    z_save_section: Optional[Union[List[float], npt.NDArray[np.float64]]],
    options: PropagationOptions,
) -> Tuple[
    npt.NDArray[np.float64],
    Coordinates,
    Velocities,
    npt.NDArray[np.int_],
    SectionData,
]:
    """
    Propagate trajectories through a linear (harmonic) section.

    This function handles propagation through sections with linear restoring forces,
    such as harmonic traps or ion guides. The motion is analytically solvable with
    sinusoidal trajectories in the transverse directions and ballistic in z.

    Args:
        indices (npt.NDArray[np.int_]):
            Array of trajectory indices for tracking which particles survive.
        timestamps_tracked (npt.NDArray[np.float64]):
            Array of tracked timestamps. Can be 1D (initial) or 2D (accumulated).
        coordinates_tracked (Coordinates):
            Current positions of all tracked trajectories.
        velocities_tracked (Velocities):
            Current velocities of all tracked trajectories.
        section (LinearSection):
            The linear section with spring constants defining the restoring forces.
        particle (Particle):
            The particle species being propagated (contains mass, etc.).
        force (Force):
            External force applied during propagation (e.g., gravity).
        z_save_section (Optional[Union[List[float], npt.NDArray[np.float64]]]):
            Z positions within this section where intermediate states should be saved.
        options (PropagationOptions):
            Configuration options for propagation behavior.

    Returns:
        tuple:
            A tuple containing:
            - Updated timestamps_tracked (npt.NDArray[np.float64]).
            - Updated coordinates_tracked (Coordinates).
            - Updated velocities_tracked (Velocities).
            - Updated indices (npt.NDArray[np.int_]).
            - SectionData for the section.
    """
    # Calculate total constant force and acceleration (non-harmonic component)
    force_cst = force + section.force
    acceleration = Acceleration(
        force_cst.fx / particle.mass,
        force_cst.fy / particle.mass,
        force_cst.fz / particle.mass,
    )

    # Calculate angular frequencies for harmonic motion in each direction
    # ω = √(k/m) where k is spring constant and m is particle mass
    w = (
        math.sqrt(section.spring_constant[0] / particle.mass),
        math.sqrt(section.spring_constant[1] / particle.mass),
        math.sqrt(section.spring_constant[2] / particle.mass),
    )
    # Propagate trajectories through linear (harmonic) section
    # Extract last timestamp (handling both 1D initial and 2D accumulated arrays)
    (
        mask,
        timestamp_list,
        coord_list,
        velocities_list,
        nr_collisions,
        collisions,
    ) = propagate_linear_trajectories(
        t_start=timestamps_tracked
        if timestamps_tracked.ndim == 1
        else timestamps_tracked[:, -1],
        origin=coordinates_tracked.get_last(),
        velocities=velocities_tracked.get_last(),
        objects=section.objects,
        z_stop=section.stop,
        acceleration=acceleration,
        w=w,
        trap_center=(section.x, section.y),
        z_save=z_save_section,
        save_collisions=section.save_collisions,
        options=options,
    )

    # Filter out trajectories that collided with objects
    timestamps_tracked = timestamps_tracked[mask]
    coordinates_tracked = coordinates_tracked.get_masked(mask)
    velocities_tracked = velocities_tracked.get_masked(mask)
    indices = indices[mask]

    # append latest timestamps, coordinates and velocities to the 2D arrays
    timestamps_tracked = cp.column_stack([timestamps_tracked, timestamp_list])
    coordinates_tracked.column_stack(coord_list)
    velocities_tracked.column_stack(velocities_list)

    
    section_data = SectionData(section.name, collisions, nr_collisions, len(mask))

    return (
        timestamps_tracked,
        coordinates_tracked,
        velocities_tracked,
        indices,
        section_data,
    )


def propagate_trajectories(
    sections: List[Union[Section, ODESection, LinearSection]],
    coordinates_init: Coordinates,
    velocities_init: Velocities,
    particle: Particle,
    t_start: Optional[npt.NDArray[np.float64]] = None,
    force: Force = Force(0.0, -9.81 * TlF().mass, 0.0),
    z_save: Optional[List[float]] = None,
    options: PropagationOptions = PropagationOptions(),
) -> Tuple[List[SectionData], Trajectories]:
    """
    Propagate trajectories through a series of sections starting from initial
    coordinates and velocities.

    Args:
        sections (List[Union[Section, ODESection, LinearSection]]):
            List of sections to propagate through.
        coordinates_init (Coordinates):
            Initial positions of the particles.
        velocities_init (Velocities):
            Initial velocities of the particles.
        particle (Particle):
            The particle to propagate.
        t_start (Optional[npt.NDArray[np.float64]], optional):
            Initial timestamps. Defaults to None.
        force (Force, optional):
            External force applied during propagation. Defaults to gravity
            (Force(0.0, -9.81 * TlF().mass, 0.0)).
        z_save (Optional[List[float]], optional):
            Z positions at which to save timestamps, coordinates, and velocities.
            Defaults to None.
        options (PropagationOptions, optional):
            Options for propagation. Defaults to PropagationOptions().

    Returns:
        Tuple[List[SectionData], Trajectories]:
            A tuple containing:
            - A list of SectionData objects with data for each section.
            - The surviving trajectories as a Trajectories object.
    """
    # initialize index array to keeps track of trajectory indices that make it through
    indices = cp.arange(len(coordinates_init))

    # Initialize state arrays for all trajectories
    # These use efficient columnar storage for ballistic/linear propagation
    timestamps_tracked = (
        cp.asarray(t_start) if t_start is not None else cp.zeros(len(indices))
    )
    coordinates_tracked = Coordinates(
        cp.asarray(coordinates_init.x),
        cp.asarray(coordinates_init.y),
        cp.asarray(coordinates_init.z),
    )
    velocities_tracked = Velocities(
        cp.asarray(velocities_init.vx),
        cp.asarray(velocities_init.vy),
        cp.asarray(velocities_init.vz),
    )

    # Container for per-section statistics (collisions, survival rates, etc.)
    section_data = []

    # Container for detailed trajectory data (initially empty, populated after ODE sections)
    # Trajectories store variable-length data more efficiently than 2D arrays
    trajectories = Trajectories()
    ode_trajectories = Trajectories()
    is_cpu_state = False

    # propagate through sections
    for section in sections:
        # Determine which z_save positions fall within this section
        if z_save is not None:
            z_save_section: Optional[List[float]] = [
                zs for zs in z_save if zs >= section.start and zs <= section.stop
            ]
        else:
            z_save_section = None
        
        # Initially when trajectories are propagated ballistically they are stored in 2D
        # arrays, because particles take the same number of steps when propagating
        # ballistically. This is no longer true when propagating with an ODE solver, and
        # then storage is switched to Trajectories containing a single Trajectory for
        # each trajectory.
        # For performance this is only done after the first ODE section since the 2D
        # array storage and propagation method is much more performant.
        # After the ODE section the coordinates and velocities are transformed into
        # 2D arrays again, starting and the end of the ODE section. This allows for
        # use of the performant ballistic propagation method again after the ODE section
        is_gpu_section = (
                    section.propagation_type == PropagationType.ballistic
                    or section.propagation_type == PropagationType.linear
                )
        
        # propagate ballistic if section is ballistic
        if is_gpu_section:
            if is_cpu_state:
                # initialize 2D arrays for keeping track of the ballistic coordinates
                timestamps_tracked = cp.asarray(timestamps_tracked)
                coordinates_tracked = Coordinates(
                    cp.asarray(coordinates_tracked.x),
                    cp.asarray(coordinates_tracked.y),
                    cp.asarray(coordinates_tracked.z),
                )
                velocities_tracked = Velocities(
                    cp.asarray(velocities_tracked.vx),
                    cp.asarray(velocities_tracked.vy),
                    cp.asarray(velocities_tracked.vz),
                )
                indices = cp.asarray(indices)
                is_cpu_state = False

            if section.propagation_type == PropagationType.ballistic:
                (
                    timestamps_tracked,
                    coordinates_tracked,
                    velocities_tracked,
                    indices,
                    sec_dat,
                ) = do_ballistic(
                    indices=indices,
                    timestamps_tracked=timestamps_tracked,
                    coordinates_tracked=coordinates_tracked,
                    velocities_tracked=velocities_tracked,
                    section=section,
                    particle=particle,
                    force=force,
                    z_save_section=z_save_section,
                    options=options,
                )
                section_data.append(sec_dat)
            else:
                (
                    timestamps_tracked,
                    coordinates_tracked,
                    velocities_tracked,
                    indices,
                    sec_dat,
                ) = do_linear(
                    indices=indices,
                    timestamps_tracked=timestamps_tracked,
                    coordinates_tracked=coordinates_tracked,
                    velocities_tracked=velocities_tracked,
                    section=cast(LinearSection, section),
                    particle=particle,
                    force=force,
                    z_save_section=z_save_section,
                    options=options,
                )
                section_data.append(sec_dat)

        # propagate ODE if section is ODE
        elif section.propagation_type == PropagationType.ode:
            # Check if particles need to be propagated ballistically to reach ODE section start
            # This can happen if there's a gap between sections or after initialization
            # We do this check on whatever device we are currently on (likely GPU)
            # Note: coordinates_tracked.get_last().z is a cupy array if is_cpu_state is False
            # np.any/np.allclose work on cupy arrays too (via array interface or cupy implementation)
            # but to be safe and consistent, we use the module corresponding to the state
            xp = np if is_cpu_state else cp
            
            last_z = coordinates_tracked.get_last().z
            if xp.any(last_z < section.start) and not xp.allclose(last_z, section.start):
                # Bridge the gap with ballistic propagation
                # We do this BEFORE converting to CPU if we are on GPU, to use fast GPU ballistic
                (
                    timestamps_tracked,
                    coordinates_tracked,
                    velocities_tracked,
                    indices,
                    sec_dat,
                ) = do_ballistic(
                    indices=indices,
                    timestamps_tracked=timestamps_tracked,
                    coordinates_tracked=coordinates_tracked,
                    velocities_tracked=velocities_tracked,
                    section=Section(
                        name="_",
                        objects=[],
                        start=float(coordinates_tracked.get_last().z[0]),  # just give it one start coord
                        stop=section.start,
                        save_collisions=False,
                    ),
                    particle=particle,
                    force=force,
                    z_save_section=z_save_section,
                    options=options,
                )
            
            # Now convert to CPU for ODE integration
            if not is_cpu_state:
                timestamps_tracked = cp.asnumpy(timestamps_tracked)
                coords_np_tuple = coordinates_tracked.get_numpy()
                vels_np_tuple = velocities_tracked.get_numpy()
                indices = cp.asnumpy(indices)

                coordinates_tracked = Coordinates(*coords_np_tuple)
                velocities_tracked = Velocities(*vels_np_tuple)
                is_cpu_state = True

            # Set up force function for ODE integration
            if isinstance(section, ODESection):
                # Use optimized scalar force function if available (e.g., from electrostatic lens)
                force_fun = cast(
                    ForceType, getattr(section, "force_fast_scalar", section.force)
                )  # type: ignore[assignment]
                force_cst = force
            else:
                # No section-specific force, only external force
                def force_fun(
                    t: float,
                    x: float,
                    y: float,
                    z: float,
                ) -> Tuple[float, float, float]:
                    return (0.0, 0.0, 0.0)

                force_cst = force + section.force

            nr_trajectories = len(indices) # indices is now numpy array

            # PRE-FILTER: Check transverse acceptance before ODE integration
            # ============================================================
            acceleration = Acceleration(
                force_cst.fx / particle.mass,
                force_cst.fy / particle.mass,
                force_cst.fz / particle.mass,
            )
            masks = [
                obj.get_acceptance(
                    coordinates_tracked.get_last(),
                    coordinates_tracked.get_last(),
                    velocities_tracked.get_last(),
                    acceleration,
                )
                for obj in section.objects
            ]

            nr_collisions = 0
            collisions = []

            # Apply acceptance filters from all objects in this section
            if len(masks) > 0:
                # Combine all acceptance masks (trajectory must pass all apertures)
                mask = np.bitwise_and.reduce(masks)

                # Count and optionally save collision data
                nr_collisions += (~mask).sum()
                if section.save_collisions:
                    collisions.append(
                        (
                            coordinates_tracked.get_last()[~mask],
                            velocities_tracked.get_last()[~mask],
                        )
                    )

                # Filter out trajectories that failed acceptance check
                timestamps_tracked = timestamps_tracked[mask]
                coordinates_tracked = coordinates_tracked.get_masked(mask)
                velocities_tracked = velocities_tracked.get_masked(mask)
                indices = indices[mask]

            (
                solutions,
                final_t_np,      # NEW from modified solver
                final_coords_np, # NEW from modified solver
                final_vels_np    # NEW from modified solver
            ) = propagate_ODE_trajectories(
                t_start=timestamps_tracked[:, -1]
                if timestamps_tracked.ndim > 1
                else timestamps_tracked, # Already 1D
                origin=coordinates_tracked.get_last(),
                velocities=velocities_tracked.get_last(),
                z_stop=section.stop,
                mass=particle.mass,
                force_fun=force_fun,
                force_cst=force_cst,
                events=[obj.collision_event_function for obj in section.objects],
                options=options,
            )

            for sol, index in zip(solutions, indices):
                ode_trajectories.add_data_ode(index, sol)

            # Use the new bulk arrays for history tracking (FAST)
            timestamps_tracked = np.column_stack([timestamps_tracked, final_t_np])

            coordinates_tracked.column_stack(
                Coordinates(final_coords_np[:,0], final_coords_np[:,1], final_coords_np[:,2])
            )
            velocities_tracked.column_stack(
                Velocities(final_vels_np[:,0], final_vels_np[:,1], final_vels_np[:,2])
            )
            
            # POST-FILTER: Check for trajectories that terminated early (collisions during ODE)
            mask = np.ones(len(solutions), dtype=bool)
            for idx, (sol, index) in enumerate(zip(solutions, indices)):
                # If final z position doesn't match section stop, trajectory hit an object
                if not math.isclose(sol.y[2, -1], section.stop):
                    mask[idx] = False

            # Handle trajectories that collided during ODE integration
            if (~mask).sum() > 0:
                nr_collisions += (~mask).sum()
                if section.save_collisions:
                    collisions.append(
                        (
                            coordinates_tracked.get_last()[~mask],  # type: ignore[index]
                            velocities_tracked.get_last()[~mask],  # type: ignore[index]
                        )
                    )

                # Remove collided trajectories from tracking
                coordinates_tracked = coordinates_tracked.get_masked(mask)
                velocities_tracked = velocities_tracked.get_masked(mask)
                indices = indices[mask]
                timestamps_tracked = timestamps_tracked[mask]

            # Record section statistics
            section_data.append(
                SectionData(section.name, collisions, nr_collisions, nr_trajectories)
            )

        # Vectorized ODE propagation
        elif section.propagation_type == PropagationType.ode_vectorized:
             # Similar logic to ODE: ensure CPU state
            xp = np if is_cpu_state else cp
            last_z = coordinates_tracked.get_last().z
            if xp.any(last_z < section.start) and not xp.allclose(last_z, section.start):
                (
                    timestamps_tracked,
                    coordinates_tracked,
                    velocities_tracked,
                    indices,
                    sec_dat,
                ) = do_ballistic(
                    indices=indices,
                    timestamps_tracked=timestamps_tracked,
                    coordinates_tracked=coordinates_tracked,
                    velocities_tracked=velocities_tracked,
                    section=Section(
                        name="_",
                        objects=[],
                        start=float(coordinates_tracked.get_last().z[0]),
                        stop=section.start,
                        save_collisions=False,
                    ),
                    particle=particle,
                    force=force,
                    z_save_section=z_save_section,
                    options=options,
                )
            
            # Convert to CPU
            if not is_cpu_state:
                timestamps_tracked = cp.asnumpy(timestamps_tracked)
                coords_np_tuple = coordinates_tracked.get_numpy()
                vels_np_tuple = velocities_tracked.get_numpy()
                indices = cp.asnumpy(indices)

                coordinates_tracked = Coordinates(*coords_np_tuple)
                velocities_tracked = Velocities(*vels_np_tuple)
                is_cpu_state = True

            coord_last = coordinates_tracked.get_last()
            vel_last = velocities_tracked.get_last()
            tstart = (
                timestamps_tracked[:, -1]
                if timestamps_tracked.ndim > 1
                else timestamps_tracked
            )
            if options.section_options.get(section.name) is None:
                raise ValueError("Options for vectorized ODE solver required")
            
            t, d, survived = rk4_batch_fixed_steps(
                f=ode_fun_batch,
                t0=tstart,
                z_stop=section.stop,
                d0=np.column_stack(
                    (
                        coord_last.x,
                        coord_last.y,
                        coord_last.z,
                        vel_last.vx,
                        vel_last.vy,
                        vel_last.vz,
                    )
                ),
                n_steps=options.section_options[section.name].n_steps,
                n_save=options.section_options[section.name].n_save,
                mass=particle.mass,
                force_fn=section.force,
                force_cst=force,
                check_bounds_functions=[
                    obj.check_in_bounds_vectorized for obj in section.objects
                ],
            )

            timestamps_tracked = timestamps_tracked[survived]
            coordinates_tracked = coordinates_tracked.get_masked(survived)
            velocities_tracked = velocities_tracked.get_masked(survived)
            indices = indices[survived]

            for idx in range(t.shape[0]):
                timestamps_tracked = np.column_stack(
                    [timestamps_tracked, t[idx, survived]]
                )
                coordinates_tracked.column_stack(
                    Coordinates(
                        d[idx, survived, 0], d[idx, survived, 1], z=d[idx, survived, 2]
                    )
                )
                velocities_tracked.column_stack(
                    Velocities(
                        d[idx, survived, 3], d[idx, survived, 4], d[idx, survived, 5]
                    )
                )

            # Create section statistics
            section_data.append(SectionData(section.name, [], (~survived).sum(), len(t)))

        gc.collect()  # Collect garbage to free memory after each section
    
    if isinstance(timestamps_tracked, cp.ndarray):
        timestamps_tracked = cp.asnumpy(timestamps_tracked)
        coordinates_tracked = Coordinates(*coordinates_tracked.get_numpy())
        velocities_tracked = Velocities(*velocities_tracked.get_numpy())
        indices = cp.asnumpy(indices)

    
    # Use the 'add_data_bulk' method you created
    trajectories.add_data_bulk(
        indices,
        timestamps_tracked,
        coordinates_tracked,
        velocities_tracked
    )

    # ** MERGE DETAILED ODE STEPS **
    # If an ODE section ran, 'ode_trajectories' has detailed
    # intermediate steps. We merge this data into our final object.
    if len(ode_trajectories) > 0:
        for index, trajectory in trajectories.items():
            if index in ode_trajectories:
                # Replace the simple history (from bulk add)
                # with the detailed history (from ODE)
                detailed_traj = ode_trajectories[index]
                trajectory.t = detailed_traj.t
                trajectory.coordinates = detailed_traj.coordinates
                trajectory.velocities = detailed_traj.velocities

    # remove duplicate entries
    for trajectory in trajectories.values():
        trajectory.remove_duplicate_entries()

    return section_data, trajectories
