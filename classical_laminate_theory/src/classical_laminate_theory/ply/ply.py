from dataclasses import dataclass, field

import numpy as np

from classical_laminate_theory.materials.base import Material2D

@dataclass(frozen=True, slots=True)
class Ply:
    """
    Single lamina.

    The ply precomputes and stores all constitutive and transformation
    tensors required for Classical Laminate Theory (CLT), including
    stiffness, compliance, and stress/strain transformation matrices
    in both local (1–2) and global (x–y) coordinates.

    Parameters
    ----------
    material : Material2D
        Constitutive material model providing local plane-stress stiffness and
        compliance matrices.
    theta : float
        Fiber orientation angle in degrees.
        The angle is measured from the global x-axis to the local material
        1-direction using the right-hand rule (counterclockwise positive).
    thickness : float
        Ply thickness (must be positive).

    Attributes
    ----------
    Q_12 : ndarray of shape (3, 3)
        Reduced stiffness matrix in local material coordinates (1–2).
    Q_xy : ndarray of shape (3, 3)
        Reduced stiffness matrix transformed to global coordinates (x–y).
    S_12 : ndarray of shape (3, 3)
        Reduced compliance matrix in local material coordinates.
    S_xy : ndarray of shape (3, 3)
        Reduced compliance matrix in global coordinates.
    T_eps : ndarray of shape (3, 3)
        Strain transformation matrix from local (1–2) to global (x–y)
        coordinates using engineering strain notation [εx, εy, γxy].
    T_eps_inv : ndarray of shape (3, 3)
        Inverse strain transformation matrix (global to local).
    T_sig : ndarray of shape (3, 3)
        Stress transformation matrix from local (1–2) to global (x–y)
        coordinates.
    T_sig_inv : ndarray of shape (3, 3)
        Inverse stress transformation matrix (global to local).

    Notes
    -----
    - Plane stress assumption is enforced (σ3 = τ13 = τ23 = 0).
    - Engineering shear strain notation (γ_xy) is used consistently.
    - All transformation and constitutive matrices are computed once
      during initialization and stored as immutable attributes.
    - The ply contains no load or state information. It represents
      only geometric orientation and constitutive behavior.

    See Also
    --------
    Laminate : Assembly of multiple plies into a laminate.
    Material2D : Base class defining plane-stress constitutive behavior.
    """

    material: Material2D
    theta: float
    theta_rad: float = field(init=False, repr=False)
    thickness: float
    Q_12: np.ndarray = field(init=False, repr=False,
                             doc="Stiffness matrix in local (lamina) "
                                 "coordinates")
    Q_xy: np.ndarray = field(init=False, repr=False,
                             doc="Stiffness matrix in global coordinates")
    S_12: np.ndarray = field(init=False, repr=False,
                             doc="Compliance matrix in local (lamina) "
                                 "coordinates")
    S_xy: np.ndarray = field(init=False, repr=False,
                             doc="Compliance matrix in global coordinates")
    T_eps: np.ndarray = field(init=False, repr=False,
                              doc="Strain transformation matrix from lamina "
                                  "to global coordinates")
    T_eps_inv: np.ndarray = field(init=False, repr=False,
                                  doc="Strain transformation matrix from "
                                      "global to lamina coordinates")
    T_sig: np.ndarray = field(init=False, repr=False,
                              doc="Stress transformation matrix from lamina "
                                  "to global coordinates")
    T_sig_inv: np.ndarray = field(init=False, repr=False,
                                  doc="Stress transformation matrix from "
                                      "global to lamina coordinates")

    def __post_init__(self):
        object.__setattr__(self, "theta_rad", np.deg2rad(self.theta))

        T_sig = self._compute_stress_transformation()
        T_sig_inv = np.linalg.inv(T_sig)

        T_eps = self._compute_strain_transformation()
        T_eps_inv = np.linalg.inv(T_eps)

        Q_12 = self.material.Q
        Q_xy = T_sig @ Q_12 @ T_eps_inv

        S_12 = self.material.S
        S_xy = T_eps @ S_12 @ T_sig_inv

        object.__setattr__(self, "Q_12", Q_12)
        object.__setattr__(self, "Q_xy", Q_xy)

        object.__setattr__(self, "S_12", S_12)
        object.__setattr__(self, "S_xy", S_xy)

        object.__setattr__(self, "T_eps", T_eps)
        object.__setattr__(self, "T_eps_inv", T_eps_inv)
        object.__setattr__(self, "T_sig", T_sig)
        object.__setattr__(self, "T_sig_inv", T_sig_inv)

    def _compute_strain_transformation(self) -> np.ndarray:
        """Compute transformation for strains from local (lamina) coordinates
        to global coordinates."""
        c = np.cos(self.theta_rad)
        s = np.sin(self.theta_rad)
        T_eps = np.array([[c**2, s**2, -s*c],
                          [s**2, c**2, s*c],
                          [2*s*c, -2*s*c, c**2 - s**2]])
        return T_eps

    def _compute_stress_transformation(self) -> np.ndarray:
        """Compute transformation for stresses from local (lamina) coordinates
        to global coordinates."""
        c = np.cos(self.theta_rad)
        s = np.sin(self.theta_rad)
        T_sig = np.array([[c**2, s**2, -2*s*c],
                          [s**2, c**2, 2*s*c],
                          [s*c, -s*c, c**2 - s**2]])
        return T_sig

if __name__ == "__main__":
    from classical_laminate_theory.materials.orthotropic import OrthotropicLamina
    lamina = OrthotropicLamina(E1=2e9, E2=3e9, G12=1.5e9, nu12=.3)

    ply = Ply(material=lamina, theta=45, thickness=2e-3)
