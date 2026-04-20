import numpy as np

from classical_laminate_theory.failure.base import FailureCriterion

class TsaiWu(FailureCriterion):
    """
    Tsai–Wu polynomial failure criterion for an orthotropic lamina
    under plane stress conditions.

    Failure is predicted when the failure index FI >= 1

    Parameters
    ----------
    s_hat_1t : float
        Tensile strength in material direction 1.
    s_hat_1c : float
        Compressive strength in material direction 1.
    s_hat_2t : float
        Tensile strength in material direction 2.
    s_hat_2c : float
        Compressive strength in material direction 2.
    t_hat_12 : float
        In-plane shear strength.

    Notes
    -----
    - Stresses must be provided in the local ply coordinate system (1–2).
    - Plane stress assumption (σ3 = τ13 = τ23 = 0).
    - The interaction parameter F12 is approximated using the common
      quadratic interaction assumption.
    - The Tsai–Wu criterion captures tension–compression asymmetry
      through the linear stress terms.
    """

    @staticmethod
    def failure_index(s_hat_1t: int | float | np.number,
                      s_hat_1c: int | float | np.number,
                      s_hat_2t: int | float | np.number,
                      s_hat_2c: int | float | np.number,
                      t_hat_12: int | float | np.number,
                      stress: np.ndarray) -> float:
        """
        Compute the Tsai–Wu failure index for a given stress state.

        Parameters
        ----------
        s_hat_1t : float
            Tensile strength in material direction 1.
        s_hat_1c : float
            Compressive strength in material direction 1.
        s_hat_2t : float
            Tensile strength in material direction 2.
        s_hat_2c : float
            Compressive strength in material direction 2.
        t_hat_12 : float
            In-plane shear strength.
        stress : ndarray of shape (3,)
            Local ply stress vector: [σ1, σ2, τ12]

        Returns
        -------
        float
            Failure index. Failure is predicted if the returned value >= 1.
        """
        s1, s2, t12 = stress

        f1 = 1/s_hat_1t - 1/s_hat_1c
        f2 = 1/s_hat_2t - 1/s_hat_2c
        f11 = 1/(s_hat_1t*s_hat_1c)
        f22= 1/(s_hat_2t*s_hat_2c)
        f66 = 1/t_hat_12**f2
        f12 = -.5*np.sqrt(f11*f22)

        return f1*s1 + f2*s2 + f11*s1**2 + f22*s2**2 + f66*t12**2 + 2*f12*s1*s2


    @staticmethod
    def failure_envelope(s_hat_1t: int | float | np.number,
                         s_hat_1c: int | float | np.number,
                         s_hat_2t: int | float | np.number,
                         s_hat_2c: int | float | np.number,
                         t_hat_12: int | float | np.number,
                         n_points: int = 400, t12: float = 0.0
                         ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the closed Tsai–Wu failure envelope in the (σ1, σ2) plane
        for a fixed shear stress τ12.

        The envelope satisfies:

            FI(σ1, σ2, τ12) = 1

        Parameters
        ----------
        s_hat_1t : float
            Tensile strength in material direction 1.
        s_hat_1c : float
            Compressive strength in material direction 1.
        s_hat_2t : float
            Tensile strength in material direction 2.
        s_hat_2c : float
            Compressive strength in material direction 2.
        t_hat_12 : float
            In-plane shear strength.
        n_points : int, optional
            Number of sampling points along the σ1 axis.
        t12 : float, optional
            Fixed shear stress τ12. Default is 0.

        Returns
        -------
        s1_curve : ndarray of shape (2*n_points + 1,)
            σ1 coordinates of the closed failure boundary.
        s2_curve : ndarray of shape (2*n_points + 1,)
            σ2 coordinates of the closed failure boundary.

        Notes
        -----
        - Stresses are in the local (1–2) coordinate system.
        - The Tsai–Wu criterion produces a quadratic failure surface.
        - Only real-valued portions of the envelope are returned.
        """

        # Strength coefficients
        f1 = 1/s_hat_1t - 1/s_hat_1c
        f2 = 1/s_hat_2t - 1/s_hat_2c
        f11 = 1/(s_hat_1t*s_hat_1c)
        f22 = 1/(s_hat_2t*s_hat_2c)
        f66 = 1/t_hat_12**2
        f12 = -0.5 * np.sqrt(f11 * f22)

        # Create σ1 range
        s1 = np.linspace(-s_hat_1c, s_hat_1t, n_points)

        # Quadratic coefficients in σ2
        a = f22
        b = f2 + 2*f12*s1
        c = f11*s1**2 + f1*s1 + f66*t12**2 - 1.0

        # Discriminant
        disc = b**2 - 4*a*c
        disc = np.where(disc >= 0, disc, np.nan)

        sqrt_disc = np.sqrt(disc)

        s2_upper = (-b + sqrt_disc) / (2*a)
        s2_lower = (-b - sqrt_disc) / (2*a)

        # Remove invalid points consistently
        valid_pos = ~np.isnan(s2_upper)
        valid_neg = ~np.isnan(s2_lower)

        s1_upper = s1[valid_pos]
        s2_upper = s2_upper[valid_pos]

        s1_lower = s1[valid_neg]
        s2_lower = s2_lower[valid_neg]

        # Construct closed curve
        s1_curve = np.append(np.concatenate([s1_upper, s1_lower[::-1]]),
                             s1_upper[0])
        s2_curve = np.append(np.concatenate([s2_upper, s2_lower[::-1]]),
                             s2_upper[0])

        return s1_curve, s2_curve

if __name__ == "__main__":

    strength = dict(s_hat_1t=1080e6, s_hat_1c=620e6,
                    s_hat_2c=128e6, s_hat_2t=39e6, t_hat_12=89e6)

    s1 = [1.079e9, 1.079e9, -618e6, -622e6]
    s2 = [38e6, -120e6, 38e6, -120e6]
    t6 = [88e6, 10e6, 0, 0]

    for i, (s1_i, s2_i, t6_i) in enumerate(zip(s1, s2, t6)):
        if TsaiWu.failure_index(stress=(s1_i, s2_i, t6_i),
                                **strength) < 1:
            print (f"Load case {i}: no failure")
        else:
            print (f"Load case {i}: failure")
