from dataclasses import dataclass, field

import numpy as np

from classical_laminate_theory.materials.base import Material2D

@dataclass(frozen=True, slots=True)
class EquivalentLamina(Material2D):
    """
    Linear elastic "equivalent" lamina under plane stress conditions.

    This class represents a unidirectional or multiaxial ply as an equivalent
    single orthotropic lamina. The maerial properties are defined in the
    equivalent local material coordinate system (1–2), where:
    - 1-direction : x-direction of the multiaxial ply
    - 2-direction : y-direction of the multiaxial ply

    Parameters
    ----------
    E1 : float
        Young’s modulus in ply x-direction [Pa].
    E2 : float
        Young’s modulus in ply y-direction [Pa].
    G12 : float
        In-plane shear modulus [Pa].
    nu12 : float
        Major Poisson’s ratio (strain in direction 2 due to stress in
        direction 1).
    nu23 : float
        Minor Poisson’s ratio.

    Attributes
    ----------
    Q : ndarray of shape (3, 3)
        Reduced in-plane stiffness matrix in the "local" (1–2) coordinate
        system using engineering shear strain γ_12.

    S : ndarray of shape (3, 3)
        Reduced in-plane compliance matrix in the "local" (1–2) coordinate
        system.

    Notes
    -----
    - First of: I know this is a weird way to represent a multiaxial lamina,
      but it's what we have to work with since the DTU 10 MW RWT only specifies
      these equivalent material properties
    - Matrices are defined using engineering shear strain γ_12.
    - The material object is immutable.
    - Valid only for plane stress applications in Classical Laminate Theory.
    - No temperature, damage, or nonlinear effects are included.
    """
    E1: float = field(doc="Young’s modulus in ply x-direction",
                      metadata={"unit": "Pa"})
    E2: float = field(doc="Young’s modulus in ply y-direction",
                      metadata={"unit": "Pa"})
    G12: float = field(doc="In-plane shear modulus",
                       metadata={"unit": "Pa"})
    nu12: float = field(doc="Major Poisson’s ratio")
    nu21: float = field(doc="Minor Poisson’s ratio")
    Q: np.ndarray = field(init=False, repr=False, doc="Stiffness matrix")
    S: np.ndarray = field(init=False, repr=False, doc="Compliance matrix")
    s_hat_1t: float = field(default=0,
                            doc="Tensile strength in material direction",
                            metadata={"unit": "Pa"})
    s_hat_1c: float = field(default=0,
                            doc="Compressive strength in material direction 1",
                            metadata={"unit": "Pa"})
    s_hat_2t: float = field(default=0,
                            doc="Tensile strength in material direction 2",
                            metadata={"unit": "Pa"})
    s_hat_2c: float = field(default=0,
                            doc="Compressive strength in material direction 1",
                            metadata={"unit": "Pa"})
    t_hat_12: float = field(default=0,
                            doc="In-plane shear strength",
                            metadata={"unit": "Pa"})

    def __post_init__(self):
        S = self._compute_compliance()
        object.__setattr__(self, "S", S)

        Q = np.linalg.inv(S)
        object.__setattr__(self, "Q", Q)

    def _compute_compliance(self) -> np.ndarray:
        """Compute the plane stress compliance matrix (3x3)"""
        S = np.array([
            [1/self.E1, -self.nu21/self.E2, 0],
            [-self.nu12/self.E1, 1/self.E2, 0],
            [0, 0, 1/self.G12]
        ])
        return S

    def strength_as_dict(self):
        return {"s_hat_1t": self.s_hat_1t, "s_hat_1c": self.s_hat_1c,
                "s_hat_2t": self.s_hat_2t, "s_hat_2c": self.s_hat_2c,
                "t_hat_12": self.t_hat_12}

if __name__ == "__main__":
    triax = EquivalentLamina(E1=21.79e9, E2=14.67e9, G12=9.413e9,
                             nu12=.478, nu21=.3218)
