import numpy as np

from classical_laminate_theory.laminate.laminate import Laminate

def solve_midplane_strain(laminate: Laminate,
                          NM: np.ndarray) -> tuple[np.ndarray]:
    """
    Calculate the midplane strains in a laminate.

    Calculate the midplane strains in a laminate from the ABD-Matrix and the
    external loading vector NM.

    Parameters
    ----------
    laminate : Laminate
        The laminate to calculate the midplane stress for.
    NM : np.ndarray of shape (6,)
        Applied load vector [Nx, Ny, Nz, Mx, My, Mz].

    Returns
    -------
    eps0 : np.ndarray of shape (3,)
        Midplane strain of the laminate.
    kappa : np.ndarray of shape (3,)
        Midplane curvature of the laminate.
    """
    ABD = laminate.ABD_matrix
    strain0 = np.linalg.solve(ABD, NM)
    eps0, kappa = strain0[:3], strain0[3:]
    return eps0, kappa

if __name__ == "__main__":
    from classical_laminate_theory.materials.orthotropic import OrthotropicLamina
    from classical_laminate_theory.ply.ply import Ply
    glass_lamina = OrthotropicLamina(E1=40e9, E2=9.8e9, G12=2.8e9, nu12=.3)

    angles = np.deg2rad([0, 45, -45, 90])
    plies = [Ply(material=glass_lamina, theta=angle, thickness=15e-3)
             for angle in angles]

    laminate = Laminate(plies)

    NM = np.array([100, 200, 300, 50, 100, 80])
    eps0, kappa = solve_midplane_strain(laminate, NM)

    eps_ply0 = laminate.strain_local(ply_index=0, eps0=eps0, kappa=kappa)
    eps_plies = laminate.strain_local_all(eps0=eps0, kappa=kappa)

    sigma_ply0 = laminate.stress_local(ply_index=0, eps0=eps0, kappa=kappa)
    sigma_plies = laminate.stress_local_all(eps0=eps0, kappa=kappa)
