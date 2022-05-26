from abc import ABC
from dataclasses import dataclass

__all__ = ["TlF"]


@dataclass
class Particle(ABC):
    mass: float


@dataclass
class TlF(Particle):
    mass: float = (204.38 + 19.00) * 1.67e-27  # mass in kg
