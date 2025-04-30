from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt

from .common_types import NDArray_or_Float
from .data_structures import Acceleration, Coordinates, Force, Velocities
from .particles import Particle
from .propagation_ballistic import calculate_time_ballistic, propagate_ballistic
from .propagation_options import PropagationType
from .utils import bounds_check_tolerance

__all__ = [
    "Section",
    "ElectrostaticQuadrupoleLens",
    "MagnetostaticHexapoleLens",
    "CircularAperture",
    "RectangularAperture",
    "RectangularApertureFinite",
    "PlateElectrodes",
    "Bore",
]

path = Path(__file__).resolve().parent
with open(path / "saved_data" / "stark_poly.pkl", "rb") as f:
    stark_poly: npt.NDArray[np.float64] = pickle.load(f)
    stark_potential_default = np.polynomial.Polynomial(stark_poly)
    stark_potential_default_derivative = stark_potential_default.deriv()


@dataclass
class Section:
    """
    Generic Section dataclass

    Attributes:
        name (str): name of section
        objects (List): objects inside the section which particles can collide with
        start (float): start of section in z [m]
        stop (float): end of section in z [m]
        save_collisions (bool): save the coordinates and velocities of collisions in
                                this section
        propagation_type (PropagationType): propagation type to use for integration
        force (Optional(Force): force to use, a constant force in addition to the force
                                that acts on the entire beamline,e.g. gravity. Could be
                                used for constant deflection from an electric field.

    """

    name: str
    objects: List[BeamlineObject]
    start: float
    stop: float
    save_collisions: bool
    propagation_type: PropagationType = PropagationType.ballistic
    force: Optional[Force] = None

    def __post_init__(self):
        """
        Check if all objects reside fully inside the section, runs upon initialization.
        """
        for o in self.objects:
            assert o.check_in_bounds(
                self.start, self.stop
            ), f"{o.name} not inside {self.name}"


@dataclass
class LinearSection(Section):
    x: float = 0.0
    y: float = 0.0
    spring_constant: tuple[float, ...] = (0.0, 0.0, 0.0)
    propagation_type: PropagationType = PropagationType.linear
    force: Optional[Force] = None


class ODESection:
    propagation_type: PropagationType = PropagationType.ode

    def __init__(
        self,
        name: str,
        objects: List[BeamlineObject],
        start: float,
        stop: float,
        save_collisions: bool,
    ):
        self.name = name
        self.objects = objects
        self.start = start
        self.stop = stop
        self.save_collisions = save_collisions

    def _check_objects(self):
        """
        Check if all objects reside fully inside the section, runs upon initializatin.
        """
        for o in self.objects:
            assert o.check_in_bounds(
                self.start, self.stop
            ), f"{o.name} not inside {self.name}"

    @overload
    def force(
        self, t: float, x: float, y: float, z: float
    ) -> Tuple[float, float, float]: ...

    @overload
    def force(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]: ...

    def force(self, t, x, y, z):
        raise NotImplementedError


class ElectrostaticQuadrupoleLens(ODESection):
    """
    Electrostatic Quadrupole Lens class
    """

    def __init__(
        self,
        name: str,
        objects: List[Any],
        start: float,
        stop: float,
        V: float,
        R: float,
        x: float = 0,
        y: float = 0,
        save_collisions: bool = False,
        stark_potential: None | npt.NDArray[np.float64] = None,
    ) -> None:
        """
        Electrostatic Quadrupole Lens Section

        Args:
            name (str): name of electrostatic quadrupole lens
            objects (List): objects inside the section which particles can collide with
            start (float): start of section in z [m]
            stop (float): stop of section in z [m]
            V (float): Voltage on electrodes [Volts]
            R (float): radius of lens bore [m]
            save_collisions (Optional[bool], optional): Save the coordinates and
                                                    velocities of collisions in this
                                                    section. Defaults to False.
            stark_potential (Optional[np.ndarray]): polynomial coefficients of the stark
                                                    potential as a function of electric
                                                    field
        """
        super().__init__(name, objects, start, stop, save_collisions)
        self.V = V
        self.R = R
        self.x0 = x
        self.y0 = y
        self._check_objects()
        self._initialize_potentials(stark_potential)

    def _initialize_potentials(
        self, stark_potential: None | npt.NDArray[np.float64] = None
    ) -> None:
        """
        Generate the radial derivative of the Stark potential for the force calculation
        """
        if stark_potential is None:
            self._stark_potential = stark_potential_default
        else:
            self._stark_potential = np.polynomial.Polynomial(stark_potential)

        self._stark_potential_derivative = self._stark_potential.deriv()

    def x_transformed(self, x: NDArray_or_Float) -> NDArray_or_Float:
        return x - self.x0

    def y_transformed(self, y: NDArray_or_Float) -> NDArray_or_Float:
        return y - self.y0

    def electric_field(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Calculate the electric field of the lens at x,y,z

        Args:
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            Union[NDArray[np.float64], float]: electric field at x,y,z in V/m
        """
        _x = self.x_transformed(x)
        _y = self.y_transformed(y)
        return 2 * self.V * np.sqrt(_x**2 + _y**2) / (self.R) ** 2

    def electric_field_derivative_r(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Derivative of the electric field in r: dE/dr (r = sqrt(x^2+y^2))

        Args:
            x (Union[npt.NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[npt.NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[npt.NDArray[np.float64], float]): z coordinates(s) [m]

        Returns:
            Union[npt.NDArray[np.float64], float]: derivative of electric field in r V/m^2
        """
        if isinstance(x, np.ndarray):
            return 2 * self.V / self.R**2 * np.ones(x.shape)
        else:
            return 2 * self.V / self.R**2

    @overload
    def force(
        self, t: float, x: float, y: float, z: float
    ) -> Tuple[float, float, float]: ...

    @overload
    def force(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]: ...

    def force(self, t, x, y, z):
        """
        Calculate the force at x,y,z

        Args:
            t (float): time [s]
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            List: force in x, y and z
        """
        _x = self.x_transformed(x)
        _y = self.y_transformed(y)
        r = np.sqrt(_x**2 + _y**2)
        if r == 0:
            dx = 0.0
            dy = 0.0
        else:
            dx = _x / r
            dy = _y / r
        stark = -self.stark_potential_derivative(
            x, y, z
        ) * self.electric_field_derivative_r(x, y, z)
        return (stark * dx, stark * dy, 0)

    def stark_potential_derivative(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Calculate the derivative (dStark/dE) of the stark potential at x,y,z

        Args:
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            Union[np.ndarray, float]: derivative of stark potential dV/dE
        """
        E = self.electric_field(x, y, z)
        return self._stark_potential_derivative(E)

    def stark_potential(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Calculate the the stark potential at x,y,z

        Args:
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            Union[NDArray[np.float64], float]: stark potential
        """
        electric_field = self.electric_field(x, y, z)
        return self._stark_potential(electric_field)

    def stark_potential_E(self, electric_field: NDArray_or_Float) -> NDArray_or_Float:
        """
        Calculate the stark potential as a function of electric field in V/m

        Args:
            electric_field (Union[np.ndarray, float]): electric field in V/m

        Returns:
            Union[np.ndarray, float]: stark potential
        """
        return self._stark_potential(electric_field)

    def stark_potential_E_derivative(
        self,
        electric_field: NDArray_or_Float,
    ) -> NDArray_or_Float:
        """
        Calculate the derivative (dStark/dE) of the stark potential as a function of E

        Args:
            electric_field (Union[np.ndarray, float]): electric field in V/m

        Returns:
            Union[np.ndarray, float]: stark potential derivative dV/dE
        """
        return self._stark_potential_derivative(electric_field)


class MagnetostaticHexapoleLens(ODESection):
    """
    Magnetic Hexapole Lens class

    The magnetic field can be expressed as:
        Br = Bar * cos(3ϕ) * (r/Rin)**2
        Bϕ = -Bar * sin(3ϕ) * (r/Rin)**2
        Bz = 0
    From Manipulating beams of paramagnetic atoms and molecules using
    inhomogeneous magnetic fields (Progress in Nuclear Magnetic Resonance Spectroscopy,
    Volumes 120-121, https://doi.org/10.1016/j.pnmrs.2020.08.002)
    where Bar is given by:
        B0 * (n/(n-1)) * cos(π/M)**n * sin(2π/M)/(nπ/M) * (1-(Rin/Rout)**(n-1))
        if n > 1 or
        B0 * sin(2π/M)/(2π/M) * ln(Rout/Rin)
        if n = 1
    Here n is a measure for how many poles (Hexapole has n=3), M is the number of
    magnet sections, Rin is the inner radius, Rout the outer radius and B0 is a constant
    scaling the magnetic field.
    Expression from Eq 5 a/b from Application of permanent magnets in accelerators
    and electron storage rings (Journal of Applied Physics 57, 3605 (1985)).
    https://doi.org/10.1063/1.335021

    Converting to cartesian coordinates:
        r = sqrt(x**2 + y**2)
        ϕ = arctan2(y,x)
        sin(ϕ) = y/r
        cos(ϕ) = x/r
        sin(3ϕ) = 3*sin(ϕ)*cos(ϕ)**2 - sin(ϕ)**3
        cos(3ϕ) = cos(ϕ)**3 - 3*sin(ϕ)**2 * cos(ϕ)
        Bx = cos(ϕ)*Br - sin(ϕ)*Bϕ
        By = sin(ϕ)*Br + cos(ϕ)*Bϕ
        Bz = Bz

    Using sympy:
    ```Python
    import sympy as smp
    r, ϕ, Bar, Rin, x, y, μx, μy = smp.symbols("r ϕ Bar Rin x y μx μy")

    # substitutions
    subsine = [(smp.sin(ϕ), y/r), (smp.cos(ϕ), x/r)]
    subr = [(r, smp.sqrt(x**2 + y**2))]
    subsine3 = [
        (smp.sin(3*ϕ), 3*smp.sin(ϕ)*smp.cos(ϕ)**2 - smp.sin(ϕ)**3),
        (smp.cos(3*ϕ), smp.cos(ϕ)**3 - 3*smp.sin(ϕ)**2*smp.cos(ϕ))
    ]

    Br = Bar * smp.cos(3*ϕ) * (r/Rin)**2
    Bϕ = -Bar * smp.sin(3*ϕ) * (r/Rin)**2
    Bx = smp.cos(ϕ) * Br - smp.sin(ϕ)*Bϕ
    By = smp.sin(ϕ) * Br + smp.cos(ϕ)*Bϕ

    Bx = Bx.subs(subsine3 + subsine + subr).simplify()
    By = By.subs(subsine3 + subsine + subr).simplify()

    dBx = smp.sqrt(smp.diff(Bx,x)**2 + smp.diff(By,x)**2).simplify()
    dBy = smp.sqrt(smp.diff(Bx,y)**2 + smp.diff(By,y)**2).simplify()

    fx = (x/r * dBx).subs(subr).simplify() * -μx
    fy = (y/r * dBy).subs(subr).simplify() * -μy

    ```
    """

    def __init__(
        self,
        name: str,
        objects: List[Any],
        start: float,
        stop: float,
        particle: Particle,
        Rin: float,
        Rout: float,
        M: int,  # nr of blocks in the magnet
        n: int = 3,  # n = 2 -> quadrupole; n = 3 -> hexapole
        save_collisions: Optional[bool] = False,
    ) -> None:
        self.name = name
        self.objects = objects
        self.start = start
        self.stop = stop
        self.particle = particle

        self.Rin = Rin
        self.Rout = Rout
        self.n = n
        self.M = M
        self._magnetic_field_aperture_radius()

    @overload
    def force(
        self, t: float, x: float, y: float, z: float
    ) -> Tuple[float, float, float]: ...

    @overload
    def force(
        self,
        t: float,
        x: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        z: npt.NDArray[np.floating],
    ) -> Tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]: ...

    def force(
        self,
        t,
        x,
        y,
        z,
    ):
        """
        Calculate the force at x,y,z

        The magnetic force on the dipole is given by F = ∇(μ⋅B).

        Args:
            t (float): time [s]
            x (Union[NDArray[np.float64], float]): x coordinate(s) [m]
            y (Union[NDArray[np.float64], float]): y coordinate(s) [m]
            z (Union[NDArray[np.float64], float]): z coordinate(s) [m]

        Returns:
            List: force in x, y and z
        """
        # μ points in the direction of B for a sufficiently large field (e.g. in the
        # hexapole), so μ points along x/r, y/r, 0

        fx = -2 * x * self.Bar / self.Rin**2 * self.particle.magnetic_moment
        fy = -2 * y * self.Bar / self.Rin**2 * self.particle.magnetic_moment
        fz = fx * 0.0
        return (fx, fy, fz)

    def _magnetic_field_aperture_radius(self) -> None:
        """
        Expression from Eq 5 a/b from Application of permanent magnets in accelerators
        and electron storage rings (Journal of Applied Physics 57, 3605 (1985)).
        https://doi.org/10.1063/1.335021
        """
        Br = 1
        n, M = self.n, self.M
        Rin, Rout = self.Rin, self.Rout
        if self.n > 1:
            self.Bar: float = (
                Br
                * (n / (n - 1))
                * np.cos(np.pi / M) ** n
                * (np.sin(n * np.pi / M) / (n * np.pi / M))
                * (1 - (Rin / Rout) ** (n - 1))
            )
        elif self.n == 1:
            self.Bar = Br * np.sin(np.pi / M) / (2 * np.pi / M) * np.log(Rout / Rin)

    def magnetic_field_cylindrical(
        self,
        r: NDArray_or_Float,
        ϕ: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> List[NDArray_or_Float]:
        Br = self.Bar * np.cos(3 * ϕ) * (r / self.Rin) ** 2
        Bϕ = -self.Bar * np.sin(3 * ϕ) * (r / self.Rin) ** 2
        Bz = Br * 0.0
        return [Br, Bϕ, Bz]

    def magnetic_field_cartesian(
        self,
        x: NDArray_or_Float,
        y: NDArray_or_Float,
        z: NDArray_or_Float,
    ) -> List[NDArray_or_Float]:
        r = np.sqrt(x**2 + y**2)
        ϕ = np.arctan2(y, x)
        Br, Bϕ, Bz = self.magnetic_field_cylindrical(r, ϕ, z)
        Bx = np.cos(ϕ) * Br - np.sin(ϕ) * Bϕ
        By = np.sin(ϕ) * Br + np.cos(ϕ) * Bϕ
        Bz = Bx * 0.0
        return [Bx, By, Bz]


@dataclass
class BeamlineObject:
    x: float
    y: float
    z: float

    def check_in_bounds(self, start: float, stop: float) -> bool:
        """
        Check if the aperture is within the specified bounds

        Args:
            start (float): start coordinate in z [m]
            stop (float): stop coordinate in z [m]

        Returns:
            bool: True if within bounds, else False
        """
        return (self.z >= start) & (self.z <= stop)

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        raise NotImplementedError

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        raise NotImplementedError


@dataclass
class CircularAperture(BeamlineObject):
    """
    Circular aperture

    Attributes:
        x (float): x coordinate [m]
        y (float): y coordinate [m]
        z (float): z coordinate [m]
        r (float): radius of the aperture [m]
    """

    r: float

    @property
    def z_stop(self):
        return self.z

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        """
        check if the supplied coordinates are within the aperture

        Args:
            coords (Coordinates): coordinates to check

        Returns:
            np.ndarray: boolean array where True indicates coordinates are within the
            aperture
        """
        assert np.allclose(
            stop.z, self.z
        ), "supplied coordinates not at location of aperture"
        return (stop.x - self.x) ** 2 + (stop.y - self.y) ** 2 <= self.r**2

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        return (x - self.x) + (y - self.y) + (z - self.z)


@dataclass
class RectangularAperture(BeamlineObject):
    """
    Rectangular aperture

    Attributes:
        x (float): x coordinate [m]
        y (float): y coordinate [m]
        z (float): z coordinate [m]
        wx (float): width of the aperture [m]
        wy (float): height of the aperture [m]
    """

    wx: float
    wy: float

    @property
    def z_stop(self):
        return self.z

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        """
        check if the supplied coordinates are within the aperture

        Args:
            coords (Coordinates): coordinates to check

        Returns:
            np.ndarray: boolean array where True indicates coordinates are within the
            aperture
        """
        assert np.allclose(
            stop.z, self.z
        ), "supplied coordinates not at location of aperture"
        return (np.abs((stop.x - self.x)) <= self.wx / 2) & (
            np.abs((stop.y - self.y)) <= self.wy / 2
        )

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        x_factor = 0 if (x >= self.x - self.wx / 2 and x <= self.x + self.wx / 2) else 1
        y_factor = 0 if (y >= self.y - self.wy / 2 and y <= self.y + self.wy / 2) else 1
        return z - self.z + x_factor + y_factor


@dataclass
class RectangularApertureFinite(BeamlineObject):
    """
    Rectangular aperture

    Attributes:
        x (float): x coordinate [m]
        y (float): y coordinate [m]
        z (float): z coordinate [m]
        wx (float): width of the aperture [m]
        wy (float): height of the aperture [m]
        wxp (float): width of the plate the aperture is in [m]
        wyp (float): height of the plate the aperure is in [m]
    """

    wx: float
    wy: float
    wxp: float
    wyp: float

    @property
    def z_stop(self):
        return self.z

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        """
        check if the supplied coordinates are within the aperture

        Args:
            start (Coordinates): start coordinates
            stop (Coordinates): stop coordinates
            vels (Velocities): velocities of the particles
            force (Force): force acting on the particles

        Returns:
            np.ndarray: boolean array where True indicates coordinates are within the
            aperture
        """
        assert np.allclose(
            stop.z, self.z
        ), "supplied coordinates not at location of aperture"
        inside_aperture = (np.abs((stop.x - self.x)) <= self.wx / 2) & (
            np.abs((stop.y - self.y)) <= self.wy / 2
        )
        outside_plate = (np.abs((stop.x - self.x)) >= self.wxp / 2) & (
            np.abs((stop.y - self.y)) >= self.wyp / 2
        )
        return inside_aperture | outside_plate

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        inside_aperture_x = (x >= self.x - self.wx / 2) and (x <= self.x + self.wx / 2)
        outside_plate_x = (x >= self.x + self.wxp / 2) or (x <= self.x - self.wxp / 2)
        inside_aperture_y = (y >= self.y - self.wy / 2) and (y <= self.y + self.wy / 2)
        outside_plate_y = (y >= self.y + self.wyp / 2) or (y <= self.y - self.wyp / 2)
        x_factor = 0 if inside_aperture_x or outside_plate_x else 1
        y_factor = 0 if inside_aperture_y or outside_plate_y else 1
        return z - self.z + x_factor + y_factor


@dataclass
class PlateElectrodes(BeamlineObject):
    x: float
    y: float
    z: float
    length: float
    width: float
    separation: float

    @property
    def z_stop(self):
        return self.z + self.length

    def check_in_bounds(self, start: float, stop: float) -> bool:
        return self.z >= start and self.z + self.length <= stop

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        m, _, _ = self.get_collisions(start, stop, vels, acceleration)
        return ~m

    def get_collisions(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> Tuple[npt.NDArray[np.bool_], Coordinates, Velocities]:
        t = np.zeros(start.x.shape)

        dx_upper = (self.x + self.separation / 2) - start.x
        dy_upper = (self.x - self.separation / 2) - start.x

        m_inside = (start.x > (self.x - self.separation / 2)) & (
            start.x < (self.x + self.separation / 2)
        )
        m_below = start.x < (self.x - self.separation / 2)
        m_above = start.x > (self.x + self.separation / 2)
        m_vpos = vels.vx > 0
        m_vneg = vels.vx < 0

        m = m_inside & m_vpos
        t[m] = calculate_time_ballistic(dx_upper[m], vels.vx[m], acceleration.ax)

        m = m_inside & m_vneg
        t[m] = calculate_time_ballistic(dy_upper[m], vels.vx[m], acceleration.ax)

        m = m_below & m_vpos
        t[m] = calculate_time_ballistic(dy_upper[m], vels.vx[m], acceleration.ax)

        m = m_above & m_vneg
        t[m] = calculate_time_ballistic(dx_upper[m], vels.vx[m], acceleration.ax)

        x, v = propagate_ballistic(t, start, vels, acceleration)
        m = x.z <= (self.z + self.length)
        m &= x.z >= self.z
        m &= np.abs(x.y - self.y) <= self.width / 2
        return m, x.get_masked(m), v.get_masked(m)

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        # for now assume electrode plates in y direction

        # z_factor for checking if z coordinates are within electrodes
        z_factor = 0 if (z >= self.z and z <= self.z + self.length) else 1

        # y_factor for checking if z coordinates are within electrodes
        x_factor = (
            0
            if (x >= self.x - self.separation / 2 and x <= self.x + self.separation / 2)
            else 1
        )
        return (x - self.x) + x_factor + z_factor


@dataclass
class Bore(BeamlineObject):
    x: float
    y: float
    z: float
    length: float
    radius: float

    def __post_init__(self):
        self.z_stop = self.z + self.length

    def check_in_bounds(self, start: float, stop: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
        return (
            (math.isclose(self.z, start, rel_tol=rel_tol, abs_tol=abs_tol) or self.z > start) and
            (math.isclose(self.z + self.length, stop, rel_tol=rel_tol, abs_tol=abs_tol) or self.z + self.length < stop)
        )

    def get_acceptance(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
    ) -> npt.NDArray[np.bool_]:
        mask_z = (stop.z >= self.z) & (stop.z <= self.z + self.length)
        mask_r = ((stop.x - self.x) ** 2 + (stop.y - self.y) ** 2) > self.radius**2
        accept_array = np.ones(stop.x.shape, dtype=np.bool_)
        accept_array[mask_z & mask_r] = False
        return accept_array

    def collision_event_function(self, x: float, y: float, z: float) -> float:
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

        # z_factor for checking if z coordinates are within electrodes
        z_factor = int(~bounds_check_tolerance(z, self.z, self.z + self.length))

        return (r - self.radius) + z_factor

    def get_collisions_linear(
        self,
        start: Coordinates,
        stop: Coordinates,
        vels: Velocities,
        acceleration: Acceleration,
        w: tuple[float, float, float],
        trap_center: tuple[float, float],
    ) -> (
        Tuple[np.ndarray[Any, np.dtype[Any]]]
        | Tuple[np.ndarray[Any, np.dtype[Any]] | Coordinates | Velocities]
    ):
        assert w[2] == 0
        assert w[0] == w[1], "isotropic case requires w=(ω,ω,0) with ω>0"

        # --- bulk arrays ----------------------------------------------------
        x0, y0, z0    = map(np.asarray, (start.x, start.y, start.z))
        vx0, vy0, vz0 = map(np.asarray, (vels.vx,  vels.vy,  vels.vz))
        z_goal        = np.asarray(stop.z) if np.ndim(stop.z) else np.full_like(z0, stop.z)

        N  = x0.size
        ω  = w[0]
        if not np.allclose(w[0], w[1]) or ω <= 0 or w[2] != 0:
            raise ValueError("Need isotropic trap: w = (ω,ω,0) with ω>0.")

        # --- geometry -------------------------------------------------------
        cx = np.asarray(self.x) if np.ndim(self.x) else np.full(N, self.x)
        cy = np.asarray(self.y) if np.ndim(self.y) else np.full(N, self.y)

        sx, sy = trap_center
        sx = np.asarray(sx) if np.ndim(sx) else np.full(N, sx)
        sy = np.asarray(sy) if np.ndim(sy) else np.full(N, sy)

        # static shift caused by constant force
        ox = sx + acceleration.ax / ω**2
        oy = sy + acceleration.ay / ω**2

        # move to cylinder-centred coords
        Cx = ox - cx
        Cy = oy - cy
        Ax = x0 - ox
        Ay = y0 - oy
        Bx = vx0 / ω
        By = vy0 / ω

        R2 = self.radius ** 2
        az = acceleration.az

        # ====================================================================
        # 1)    time window (z-motion)
        # ====================================================================
        dz = z_goal - z0
        if abs(az) < 1e-14:                               # ballistic
            with np.errstate(divide="ignore", invalid="ignore"):
                t_end = dz / vz0
        else:                                             # quadratic
            a, b, c = 0.5*az, vz0, -dz
            disc    = b*b - 4*a*c
            disc[disc < 0] = 0
            rt      = np.sqrt(disc)
            t1, t2  = (-b - rt)/(2*a), (-b + rt)/(2*a)
            t_end   = np.where(dz >= 0, np.maximum(t1, t2),
                                        np.minimum(t1, t2))
        valid = (t_end > 0) & np.isfinite(t_end)

        # ------------------------------------------------ result arrays -----
        hit = np.zeros(N, bool)
        t_hit = np.full(N, np.nan)
        x_hit = np.full(N, np.nan);  y_hit = np.full(N, np.nan);  z_hit = np.full(N, np.nan)
        vx_hit = np.full(N, np.nan); vy_hit = np.full(N, np.nan); vz_hit = np.full(N, np.nan)

        if not valid.any():                       # nothing moves forward
            return hit, t_hit, Coordinates(x_hit, y_hit, z_hit), Velocities(vx_hit, vy_hit, vz_hit)

        survivors = np.where(valid)[0]

        # ====================================================================
        # 2)    analytic first hit
        # ====================================================================
        for i in survivors:
            # ---  quick analytic upper bound ---------------------------------
            centre_dist = math.hypot(Cx[i], Cy[i])
            amp2        = Ax[i]**2 + Bx[i]**2 + Ay[i]**2 + By[i]**2
            if centre_dist + math.sqrt(amp2) < self.radius - 1e-9:
                continue                       # ellipse never touches the wall

            # ---  quartic coefficients  (unchanged) ---------------------------
            a =  Bx[i]*Cy[i] + By[i]*Cx[i] + Ax[i]*By[i] + Ay[i]*Bx[i]
            b =  Ax[i]*Cx[i] + Ay[i]*Cy[i] + 0.5*(Ax[i]**2 + Ay[i]**2
                                                - Bx[i]**2 - By[i]**2)\
                + Cx[i]**2 + Cy[i]**2 - R2
            c =  Ax[i]*By[i] + Ay[i]*Bx[i]
            d = 0.5*(Ax[i]**2 + Ay[i]**2 - Bx[i]**2 - By[i]**2)

            A4, A3 = d + b, 2*(c + a)
            A2 = 2*b + 6*R2 - 6*(Cx[i]**2 + Cy[i]**2)
            A1, A0 = 2*(c - a), d - b

            u = np.roots([A4, A3, A2, A1, A0]).real
            u = u[np.abs(np.imag(u)) < 1e-8]        # keep nearly-real roots
            if u.size == 0:
                continue

            θ = (2*np.arctan(u) + 2*np.pi) % (2*np.pi)
            t_cand = θ / ω
            t_cand = t_cand[(t_cand > 0) & (t_cand <= t_end[i])]
            if t_cand.size == 0:
                continue

            t0 = float(t_cand.min())
            Cθ, Sθ = math.cos(ω*t0), math.sin(ω*t0)

            # position and radial check (0.1 µm tolerance)
            x_rel = Cx[i] + Ax[i]*Cθ + Bx[i]*Sθ
            y_rel = Cy[i] + Ay[i]*Cθ + By[i]*Sθ
            r_now = math.hypot(x_rel, y_rel)
            if abs(r_now - self.radius) > 1e-7:      # 0.1 µm ≈ 4 × 10⁻⁶ inch
                continue

            # ---  fill outputs  ------------------------------------------------
            hit[i]   = True
            t_hit[i] = t0
            x_hit[i] = cx[i] + x_rel
            y_hit[i] = cy[i] + y_rel
            z_hit[i] = z0[i] + vz0[i]*t0 + 0.5*az*t0**2

            vx_hit[i] = -ω*Ax[i]*Sθ + ω*Bx[i]*Cθ
            vy_hit[i] = -ω*Ay[i]*Sθ + ω*By[i]*Cθ
            vz_hit[i] = vz0[i] + az*t0

        return (
            hit,
            t_hit,
            Coordinates(x_hit, y_hit, z_hit),
            Velocities(vx_hit, vy_hit, vz_hit),
        )
