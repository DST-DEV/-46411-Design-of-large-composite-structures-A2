import numpy as np

from abc import ABC, abstractmethod

class Material2D(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def Q(self) -> np.ndarray:
        """Plane stress stiffness matrix (3x3)"""

    @property
    def S(self) -> np.ndarray:
        """Plane stress compliance matrix (3x3)"""
