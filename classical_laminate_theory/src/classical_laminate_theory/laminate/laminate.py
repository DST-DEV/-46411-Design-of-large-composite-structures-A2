import numpy as np

from classical_laminate_theory.ply.ply import Ply

class Laminate:
    """
    Classical laminate defined by an ordered stack of plies.

    The laminate represents a flat, plane-stress composite plate
    analyzed under Classical Laminate Theory (CLT) assumptions:

    - Each ply is homogeneous and orthotropic.
    - Perfect bonding between plies.
    - Plane stress condition.
    - Kirchhoff-Love kinematics (no transverse shear deformation).

    The laminate is defined by an ordered sequence of `Ply` objects.
    The mid-plane is located at z = 0. The through-thickness
    coordinates are computed automatically from the ply thicknesses.

    The class provides access to:
        - Ply stack (immutable)
        - Through-thickness z-coordinates
        - ABD stiffness matrix

    The laminate object is geometric and constitutive only. it does not store
    load cases or solution state.
    """

    def __init__(self, plies: list[Ply]):
        """
        Parameters
        ----------
        plies : list[Ply]
            Ordered sequence of plies defining the stacking sequence.
            The first ply in the list corresponds to the bottom ply
            (lowest z-coordinate). The last ply corresponds to the top ply.

        Notes
        -----
        - The ply list is converted to an immutable tuple.
        - The laminate mid-plane is located at z = 0.
        - Through-thickness coordinates are computed automatically from the ply
          thicknesses.
        """
        self._plies = tuple(plies)
        self._z = self._compute_z_coords()
        self._ABD_matrix = self._calculate_ABD_matrix()

    @property
    def plies(self):
        """Ordered tuple of the plies of the laminate."""
        return self._plies

    @property
    def z(self):
        """
        Tuple of (z_bottom, z_top) coordinates for each ply.

        Coordinates are measured from the laminate mid-plane.
        """
        return self._z

    @property
    def ABD_matrix(self):
        """6x6 laminate stiffness matrix."""
        return self._ABD_matrix

    def _compute_z_coords(self):
        total_thickness = sum(p.thickness for p in self.plies)
        z_bot = -total_thickness / 2
        z = []
        for ply in self.plies:
            z_top = z_bot + ply.thickness
            z.append((z_bot, z_top))
            z_bot = z_top

        return tuple(z)


    def _calculate_ABD_matrix(self):
        """
        Calculate the 6x6 laminate stiffness matrix.

        Returns
        -------
        ndarray, shape (6, 6)
            Block matrix composed of:
                A : extensional stiffness (3x3)
                B : bending-extension coupling stiffness (3x3)
                D : bending stiffness (3x3)
        """
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for ply, (z_k, z_k1) in zip(self.plies, self.z):
            Qb = ply.Q_xy
            A += Qb * (z_k1 - z_k)
            B += 0.5 * Qb * (z_k1**2 - z_k**2)
            D += (1/3) * Qb * (z_k1**3 - z_k**3)

        ABD = np.block([
            [A, -B],
            [-B, D]
        ])
        return np.round(ABD, 3)

    @staticmethod
    def strain_global(z: int | float | np.ndarray, eps0: np.ndarray,
                      kappa: np.ndarray) -> np.ndarray:
        """
        Calculate the strain at a given z-position in global coordinates.

        Calculates the strain components epsilon_x, epsilon_y, and gamma_xy as
        a numpy array with the first axis representing the strain components
        and the second axis the z-positions (in case z is a vector)

        Parameters
        ----------
        z : int | float | np.ndarray of shape (n,)
            Z-position.
        eps0 : np.ndarray of shape (3,)
            Midplane strain of the laminate.
        kappa : np.ndarray of shape (3,)
            Midplane curvature of the laminate.

        Returns
        -------
        np.ndarray of shape (3,) or (3,n)
            Strain at the specified z-position in the global coordinate system.
        """
        if not np.isscalar(z):
            if z.ndim>1 and np.sum(z.shape) != np.multiply(*z.shape):
                raise TypeError("Only scalar values and vectors supported for "
                                "z-position")
            eps0 = np.reshape(eps0, (-1, 1))
            kappa = np.reshape(kappa, (-1, 1))
            z = z.flatten()
        return eps0 + z * kappa

    def strain_local(self, ply_index: int, eps0: np.ndarray,
                     kappa: np.ndarray) -> np.ndarray:
        """
        Calculate the strain in a ply in the local ply coordinates.

        Parameters
        ----------
        ply_index : int
            Index of the ply in the laminate.
        eps0 : np.ndarray of shape (3,)
            Midplane strain of the laminate.
        kappa : np.ndarray of shape (3,)
            Midplane curvature of the laminate.

        Returns
        -------
        np.ndarray of shape (3,)
            Strain of the specified ply in the local ply coordinate system.
        """
        ply = self.plies[ply_index]
        z = self.z[ply_index]
        z_mid = (z[0] + z[1])/2
        eps_global = self.strain_global(z=z_mid, eps0=eps0, kappa=kappa)

        return ply.T_eps_inv @ eps_global

    def strain_local_all(self, eps0: np.ndarray,
                         kappa: np.ndarray) -> np.ndarray:
        """
        Calculate the strains of all plies in the respective local coordinates.

        Parameters
        ----------
        eps0 : np.ndarray of shape (3,)
            Midplane strain of the laminate.
        kappa : np.ndarray of shape (3,)
            Midplane curvature of the laminate.

        Returns
        -------
        np.ndarray of shape (3,n)
            Strain of all n plies in the respective ply coordinate system.
        """
        if len(self.plies) == 1:
            return np.atleast_2d(self.strain_local(ply_index=0, eps0=eps0,
                                                   kappa=kappa))

        z_mid = np.sum(self.z, axis=1)/2
        eps_global = self.strain_global(z=z_mid, eps0=eps0, kappa=kappa)

        T_eps_inv = np.stack([ply.T_eps_inv for ply in self.plies], axis=-1)

        return np.einsum('ijn,jn->in', T_eps_inv, eps_global)


    def stress_local(self, ply_index: int, eps0: np.ndarray,
                     kappa: np.ndarray) -> np.ndarray:
        """
        Calculate the stress in a ply in the local ply coordinates.

        Parameters
        ----------
        ply_index : int
            Index of the ply in the laminate.
        eps0 : np.ndarray of shape (3,)
            Midplane strain of the laminate.
        kappa : np.ndarray of shape (3,)
            Midplane curvature of the laminate.

        Returns
        -------
        np.ndarray of shape (3,)
            Stress of the specified ply in the local ply coordinate system.
        """
        eps_local = self.strain_local(ply_index=ply_index, eps0=eps0,
                                      kappa=kappa)
        return self.plies[ply_index].Q_12 @ eps_local

    def stress_local_all(self, eps0: np.ndarray,
                         kappa: np.ndarray) -> np.ndarray:
        """
        Calculate the stress of all plies in the respective local coordinates.

        Parameters
        ----------
        eps0 : np.ndarray of shape (3,)
            Midplane strain of the laminate.
        kappa : np.ndarray of shape (3,)
            Midplane curvature of the laminate.

        Returns
        -------
        np.ndarray of shape (3,n)
            Stresses of all n plies in the respective ply coordinate system.
        """
        if len(self.plies) == 1:
            return np.atleast_2d(self.stress_local(ply_index=0, eps0=eps0,
                                                   kappa=kappa))
        eps_local = self.strain_local_all(eps0=eps0, kappa=kappa)
        Q_12_all = np.stack([ply.Q_12 for ply in self.plies], axis=-1)
        return np.einsum('ijn,jn->in', Q_12_all, eps_local)

if __name__ == "__main__":
    from classical_laminate_theory.materials.orthotropic import OrthotropicLamina
    glass_lamina = OrthotropicLamina(E1=40e9, E2=9.8e9, G12=2.8e9, nu12=.3)

    angles = [0, 45, -45, 90]
    plies = [Ply(material=glass_lamina, theta=angle, thickness=.15e-3)
             for angle in angles]

    laminate = Laminate(plies)

    ABD_matrix = laminate.ABD_matrix
