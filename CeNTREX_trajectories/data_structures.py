from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Union, List, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

__all__: List[str] = ["Gravity", "Force", "Velocities", "Coordinates", "Trajectories"]


@dataclass(frozen=True)
class SectionData:
    """
    Specifies a section of the beamline

    Attributes
        name (str): name of the section
        saved_collisions (list): saved coordinates and trajectories of collisions
        nr_collisions (int): number of collisions in section
        nr_trajectories (int): number of trajectories entering the section
        survived (int): number of trajectories surviving the section
        throughput (float): survival rate of section
    """

    name: str
    saved_collisions: List[Any]
    nr_collisions: int
    nr_trajectories: int
    survived: int = field(init=False)
    throughput: float = field(init=False)

    def __post_init__(self):
        super().__setattr__("survived", self.nr_trajectories - self.nr_collisions)
        super().__setattr__("throughput", self.survived / self.nr_trajectories)


@dataclass
class Force:
    """
    Force

    Attributes:
        fx (float): force in x [N or kg*m/s^2]
        fy (float): force in y [N or kg*m/s^2]
        fz (float): force in z [N or kg*m/s^2]
    """

    fx: float
    fy: float
    fz: float


@dataclass
class Gravity:
    """
    Gravity

    Attributes:
        gx (float): gravitational acceleration in x [m/s^2]
        gy (float): gravitational acceleration in y [m/s^2]
        gz (float): gravitational acceleration in z [m/s^2]
    """

    gx: float
    gy: float
    gz: float


@dataclass
class Velocity:
    """
    Velocity

    Attributes:
        vx (float): velocity in x [m/s]
        vy (float): velocity in y [m/s]
        vz (float): velocity in z [m/s]
    """

    vx: float
    vy: float
    vz: float


@dataclass
class Velocities:
    """
    Velocities

    Attributes:
        vx (NDArray[np.float64]): velocities in x [m/s]
        vy (NDArray[np.float64]): velocities in y [m/s]
        vz (NDArray[np.float64]): velocities in z [m/s]
    """

    vx: npt.NDArray[np.float64]
    vy: npt.NDArray[np.float64]
    vz: npt.NDArray[np.float64]

    def get_masked(self, mask: npt.NDArray[np.bool_]) -> Velocities:
        """
        return the masked velocities

        Args:
            mask (NDArray[np.bool_]): boolean array to select velocities to return

        Returns:
            Velocities: masked velocities
        """
        return Velocities(self.vx[mask], self.vy[mask], self.vz[mask])

    def __getitem__(self, i: int) -> Union[Velocity, Velocities]:
        """
        Get either a single Velocity or Velocities, depending on whether the stored
        velocities are a 1D or 2D array

        Args:
            i (int): index

        Returns:
            Union[Velocity, Velocities]: Velocity or Velocities corresponding to index i
        """
        if len(self.vx.shape) > 1:
            return Velocities(self.vx[i, :], self.vy[i, :], self.vz[i, :])
        else:
            return Velocity(self.vx[i], self.vy[i], self.vz[i])

    def __len__(self):
        return len(self.vx)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if len(self.vx.shape) > 1:
            _i = self._i
            if _i < len(self.vx):
                c = Velocities(self.vx[_i, :], self.vy[_i, :], self.vz[_i, :])
                self._i += 1
                return c
            else:
                raise StopIteration
        else:
            _i = self._i
            if _i < self.vx.size:
                c = Velocity(self.vx[_i], self.vy[_i], self.vz[_i])
                self._i += 1
                return c
            else:
                raise StopIteration

    def append(self, velocities: Velocities) -> None:
        """
        Append velocities

        Args:
            velocities (Velocities): velocities to append
        """
        self.vx = np.append(self.vx, velocities.vx)
        self.vy = np.append(self.vy, velocities.vy)
        self.vz = np.append(self.vz, velocities.vz)

    def column_stack(self, velocities: Velocities) -> None:
        """
        Column stack velocities

        Args:
            velocities (Velocities): velocities to column stack
        """
        self.vx = np.column_stack([self.vx, velocities.vx])
        self.vy = np.column_stack([self.vy, velocities.vy])
        self.vz = np.column_stack([self.vz, velocities.vz])

    def append_from_ode(
        self,
        sol: OptimizeResult,
        save_start: bool = True,
        v_indices: npt.NDArray[np.int32] = np.ndarray([3, 4, 5]),
    ) -> None:
        """
        Append to velocities from an ODE solution

        Args:
            sol (OptimizeResult): ODE solution for a trajectory under variable force
                                    from solve_ivp
            save_start (bool, optional): save start value from sol. Defaults to True.
        """
        if save_start:
            slice = np.s_[:]
        else:
            slice = np.s_[1:]
        vx_i, vy_i, vz_i = v_indices
        self.vx = np.append(self.vx, sol.y[vx_i, slice])
        self.vy = np.append(self.vy, sol.y[vy_i, slice])
        self.vz = np.append(self.vz, sol.y[vz_i, slice])

    def get_last(self) -> Velocities:
        """
        Get the last velocity entry

        Returns:
            Velocities: last velocities in arrays
        """
        if not len(self.vx.shape) > 1:
            return self
        return Velocities(self.vx[:, -1], self.vy[:, -1], self.vz[:, -1])


@dataclass
class Coordinate:
    """
    Coordinate

    Attributes
        x (float): coordinate in x [m]
        y (float): coordinate in y [m]
        z (float): coordinate in z [m]
    """

    x: float
    y: float
    z: float


@dataclass
class Coordinates:
    """
    Coordinates

    Attributes:
        x (NDArray[np.float64]): coordinates in x [m]
        y (NDArray[np.float64]): coordinates in y [m]
        z (NDArray[np.float64]): coordinates in z [m]
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]

    def get_masked(self, mask: npt.NDArray[np.bool_]) -> Coordinates:
        """
        return the masked Coordinates

        Args:
            mask (NDArray[np.bool_]): boolean array to select coordinates to return

        Returns:
            Coordinates: masked coordinates
        """
        return Coordinates(self.x[mask], self.y[mask], self.z[mask])

    def __getitem__(self, i: int) -> Union[Coordinate, Coordinates]:
        """
        Get either a single Coordinate or Coordinates, depending on whether the stored
        Coordinates are a 1D or 2D array

        Args:
            i (int): index

        Returns:
            Union[Coordinate, Coordinates]: Coordinate or Coordinates corresponding to
                                            index i
        """
        if len(self.x.shape) > 1:
            return Coordinates(self.x[i, :], self.y[i, :], self.z[i, :])
        else:
            return Coordinate(self.x[i], self.y[i], self.z[i])

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if len(self.x.shape) > 1:
            _i = self._i
            if _i < len(self.z):
                c = Coordinates(self.x[_i, :], self.y[_i, :], self.z[_i, :])
                self._i += 1
                return c
            else:
                raise StopIteration
        else:
            _i = self._i
            if _i < self.x.size:
                c = Coordinate(self.x[_i], self.y[_i], self.z[_i])
                self._i += 1
                return c
            else:
                raise StopIteration

    def append(self, coordinates: Coordinates) -> None:
        """
        Append coordinates

        Args:
            velocities (Velocities): velocities to append
        """
        self.x = np.append(self.x, coordinates.x)
        self.y = np.append(self.y, coordinates.y)
        self.z = np.append(self.z, coordinates.z)

    def column_stack(self, coordinates: Coordinates) -> None:
        """
        Column stack coordinates

        Args:
            coordinates (Coordinates): coordinates to column stack
        """
        self.x = np.column_stack([self.x, coordinates.x])
        self.y = np.column_stack([self.y, coordinates.y])
        self.z = np.column_stack([self.z, coordinates.z])

    def append_from_ode(
        self,
        sol: OptimizeResult,
        save_start: bool = True,
        x_indices: npt.NDArray[np.int32] = np.ndarray([0, 1, 2]),
    ) -> None:
        """
        Append to coordinates from an ODE solution

        Args:
            sol (OptimizeResult): ODE solution for a trajectory under variable force
                                    from solve_ivp
            save_start (bool, optional): save start value from sol. Defaults to True.
        """
        if save_start:
            slice = np.s_[:]
        else:
            slice = np.s_[1:]
        x_i, y_i, z_i = x_indices
        self.x = np.append(self.x, sol.y[x_i, slice])
        self.y = np.append(self.y, sol.y[y_i, slice])
        self.z = np.append(self.z, sol.y[z_i, slice])

    def get_last(self) -> Coordinates:
        """
        Get the last coordinate entry

        Returns:
            Coordinates: last coordinates in arrays
        """
        if not len(self.x.shape) > 1:
            return self
        return Coordinates(self.x[:, -1], self.y[:, -1], self.z[:, -1])


@dataclass
class Trajectory:
    """
    Trajectory holds the timestamps, coordinates and velocities for a single particle

    Attributes:
        t (ndarray[float]) timestamps [s]
        coordinates (Coordinates): coordinates
        velocities (Velocities): velocities
        x (ndarray[float]): coordinates in x [m]
        y (ndarray[float]): coordinates in y [m]
        z (ndarray[float]): coordinates in z [m]
        vx (ndarray[float]): velocities in x [m/s]
        vy (ndarray[float]): velocities in y [m/s]
        vz (ndarray[float]): velocities in z [m/s]
        index (int): index of particle, from initial distribution
    """

    t: npt.NDArray[np.float64]
    coordinates: Coordinates
    velocities: Velocities
    index: int

    def __getitem__(self, i: int) -> tuple:
        """
        Get the timestamp, coordinates and velocities at index i

        Args:
            i (int): index

        Returns:
            tuple: timestamp, coordinates, velocities
        """
        return self.t[i], self.coordinates[i], self.velocities[i]

    def __len__(self):
        return len(self.coordinates)

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """
        x coordinates [m]

        Returns:
            NDArray[np.float64]: x coordinates [m]
        """
        return self.coordinates.x

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """
        y coordinates [m]

        Returns:
            NDArray[np.float64]: y coordinates [m]
        """
        return self.coordinates.y

    @property
    def z(self) -> npt.NDArray[np.float64]:
        """
        z coordinates [z]

        Returns:
            NDArray[np.float64]: z coordinates [m]
        """
        return self.coordinates.z

    @property
    def vx(self) -> npt.NDArray[np.float64]:
        """
        x velocities [m/s]

        Returns:
            NDArray[np.float64]: x velocities [m/s]
        """
        return self.velocities.vx

    @property
    def vy(self) -> npt.NDArray[np.float64]:
        """
        y velocities [m/s]

        Returns:
            NDArray[np.float64]: y velocities [m/s]
        """
        return self.velocities.vy

    @property
    def vz(self) -> npt.NDArray[np.float64]:
        """
        z velocities [m/s]

        Returns:
            NDArray[np.float64]: z velocities [m/s]
        """
        return self.velocities.vz

    def append(
        self,
        t: npt.NDArray[np.float64],
        coordinates: Coordinates,
        velocities: Velocities,
    ) -> None:
        """
        append timestamps, coordinates and velocities to the trajectory

        Args:
            t (NDArray[np.float64]): timestamps
            coordinates (Coordinates): coordinates
            velocities (Velocities): velocities
        """
        self.t = np.append(self.t, t)
        self.coordinates.append(coordinates)
        self.velocities.append(velocities)

    def append_from_ode(self, sol: OptimizeResult, save_start: bool = True) -> None:
        """
        Append to trajectory from an OdeResult.

        Args:
            sol (OptimizeResult): OdeResult from solve_ivp for a trajectory with
                                    varying force
            save_start (bool, optional): save start value from sol. Defaults to True.
        """
        if save_start:
            self.t = np.append(self.t, sol.t)
        else:
            self.t = np.append(self.t, sol.t[1:])
        self.coordinates.append_from_ode(sol, save_start)
        self.velocities.append_from_ode(sol, save_start)

    def remove_duplicate_entries(self) -> None:
        """
        Remove duplicate entries from the trajectory
        """
        self.t, indices = np.unique(self.t, return_index=True)
        mask = np.zeros(len(self.coordinates), dtype=bool)
        mask[indices] = True
        self.coordinates = self.coordinates.get_masked(mask)
        self.velocities = self.velocities.get_masked(mask)


class Trajectories(MutableMapping):
    """
    Holds multiple Trajectory objects
    """

    def __init__(self, *args, **kwargs):
        self._storage = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def __repr__(self):
        return f"Trajectories(n={self.__len__()})"

    def __delitem__(self, key):
        del self._storage[key]

    def __setitem__(self, key, value):
        assert isinstance(value, Trajectory)
        self._storage[key] = value

    def delete_trajectories(
        self, indices: Union[List[int], npt.NDArray[np.int32]]
    ) -> None:
        """
        Delete trajectories corresponding to indices

        Args:
            indices (np.ndarray[int]): indices of trajectories to remove
        """
        for index in indices:
            del self._storage[index]

    def add_data(
        self,
        index: int,
        t: npt.NDArray[np.float64],
        coordinates: Coordinates,
        velocities: Velocities,
    ) -> None:
        """
        Add data to Trajectory `index`, create trajectory if not present

        Args:
            index (int): index
            t (NDArray[np.float64]): timestamps [s]
            coordinates (Coordinates): coordinates
            velocities (Velocities): velocities
        """
        if index not in self._storage:
            self.__setitem__(index, Trajectory(t, coordinates, velocities, index))
        else:
            self._storage[index].append(t, coordinates, velocities)

    def add_data_ode(self, index: int, sol: OptimizeResult) -> None:
        """
        Add data from an OdeResult to Trajectory `index`

        Args:
            index (int): index
            sol (OptimizeResult): OdeResult from solve_ivp for a trajectory with
                                    varying force
        """
        if index not in self._storage:
            self.__setitem__(
                index,
                Trajectory(
                    sol.t,
                    Coordinates(sol.y[0, :], sol.y[1, :], sol.y[2, :]),
                    Velocities(sol.y[3, :], sol.y[4, :], sol.y[5, :]),
                    index,
                ),
            )
        else:
            self._storage[index].append_from_ode(sol, save_start=False)
