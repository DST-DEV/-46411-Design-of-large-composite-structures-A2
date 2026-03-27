import numpy as np

from classical_laminate_theory.failure.base import FailureCriterion

class TsaiHill(FailureCriterion):
    """
    Tsai–Hill quadratic failure criterion for an orthotropic lamina
    under plane stress conditions.

    Failure is predicted when the failure index FI >= 1


    Notes
    -----
    - Stresses must be provided in the local ply coordinate system (1–2).
    - Plane stress assumption (σ3 = τ13 = τ23 = 0).
    - The criterion does not distinguish between different failure modes
      (fiber vs matrix); it provides a single interaction index.
    """

    @staticmethod
    def failure_index(s_hat_1t: int | float | np.number,
                      s_hat_1c: int | float | np.number,
                      s_hat_2t: int | float | np.number,
                      s_hat_2c: int | float | np.number,
                      t_hat_12: int | float | np.number,
                      stress: np.ndarray) -> float:
        """
        Compute the Tsai–Hill failure index for a given stress state.

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
        s_hat_1 = s_hat_1t if s1 >= 0 else s_hat_1c
        s_hat_2 = s_hat_2t if s2 >= 0 else s_hat_2c

        return (s1/s_hat_1)**2 - (s1*s2)/(s_hat_1**2) + (s2/s_hat_2)**2 \
            + (t12/t_hat_12)**2

    @staticmethod
    def failure_envelope(s_hat_1t: int | float | np.number,
                         s_hat_1c: int | float | np.number,
                         s_hat_2t: int | float | np.number,
                         s_hat_2c: int | float | np.number,
                         t_hat_12: int | float | np.number,
                         n_points: int = 400, t12: float = 0.0
                         ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the closed Tsai–Hill failure envelope in the (σ1, σ2) plane
        for a fixed shear stress τ12.

        The returned arrays define a closed curve satisfying:

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
        - Stresses are assumed in the local (1–2) coordinate system.
        - The envelope accounts for different tensile and compressive
          strengths in both material directions.
        - Portions without real solutions are automatically excluded.
        """
        s1 = np.linspace(-s_hat_1c, s_hat_1t, n_points)

        # Select X strength depending on sign of σ1
        s_hat_1 = np.where(s1 >= 0, s_hat_1t, s_hat_1c)

        shear_term = (t12 / t_hat_12) ** 2

        def solve_branch(s_hat_2):
            """Solve σ2 branches via quadratic formula."""
            a = 1.0 / s_hat_2**2
            b = -s1 / s_hat_1**2
            c = (s1 / s_hat_1)**2 + shear_term - 1.0

            disc = b**2 - 4*a*c
            disc = np.where(disc >= 0, disc, np.nan)

            sqrt_disc = np.sqrt(disc)

            s2_pos = (-b + sqrt_disc) / (2*a)
            s2_neg = (-b - sqrt_disc) / (2*a)

            return s2_pos, s2_neg

        # Solve assuming σ2 ≥ 0
        s2_t_pos, s2_t_neg = solve_branch(s_hat_2t)

        # Solve assuming σ2 < 0
        s2_c_pos, s2_c_neg = solve_branch(s_hat_2c)

        # Merge based on sign consistency
        s2_upper = np.where(s2_t_pos >= 0, s2_t_pos, s2_c_pos)
        s2_lower = np.where(s2_t_neg < 0, s2_t_neg, s2_c_neg)

        # Remove invalid (NaN) points consistently
        valid_upper = ~np.isnan(s2_upper)
        valid_lower = ~np.isnan(s2_lower)

        s1_upper = s1[valid_upper]
        s2_upper = s2_upper[valid_upper]

        s1_lower = s1[valid_lower]
        s2_lower = s2_lower[valid_lower]

        # Construct closed curve:
        # forward along upper branch
        # backward along lower branch
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
        if TsaiHill.failure_index(stress=(s1_i, s2_i, t6_i),
                                   **strength) < 1:
            print (f"Load case {i}: no failure")
        else:
            print (f"Load case {i}: failure")
