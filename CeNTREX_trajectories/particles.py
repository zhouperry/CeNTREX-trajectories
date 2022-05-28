import math
from abc import ABC
from dataclasses import dataclass

__all__ = ["Particle", "TlF", "BaF"]


@dataclass
class Particle(ABC):
    mass: float
    magnetic_moment: float = math.nan


@dataclass
class TlF(Particle):
    mass: float = (204.38 + 19.00) * 1.67e-27  # mass in kg


@dataclass
class BaF(Particle):
    mass: float = (137 + 19) * 1.67e-27  # mass in kg
    magnetic_moment: float = 9.274e-24  # magnetic moment in J/T
