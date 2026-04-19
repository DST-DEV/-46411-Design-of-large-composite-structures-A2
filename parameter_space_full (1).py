from pathlib import Path
from typing import List
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr
import yaml

import classical_laminate_theory as clt
import scivis

# ===================================================================
# %% PARAMETERS
# ===================================================================
L = 4.0  # Beam span [m]
W = 4.2  # Panel width [m]

# Cross beam geometry
B_CROSS = 0.3  # Cross beam width [m]
L_CROSS = 4.7  # Cross beam length [m]

# Layup definition
# Final design skin layup: [0^5/+45/-45]_S.
# It is written explicitly as a symmetric stacking sequence to avoid parser
# ambiguity in the superscript notation.
LAYUP_STR_PANEL = "[0^5/+45/-45]_S"
LAYUP_PANEL = [0, 0, 0, 0, 0, 45, -45, -45, 45, 0, 0, 0, 0, 0]
N_PLIES_PANEL = len(LAYUP_PANEL)

LAYUP_STR_CROSS = "[0^5/+45/-45]_S"
LAYUP_CROSS = [0, 0, 0, 0, 0, 45, -45, -45, 45, 0, 0, 0, 0, 0]
N_PLIES_CROSS = len(LAYUP_CROSS)

# Panel skin thickness search bounds [m]
T_PLY_MIN , T_PLY_MAX = 2e-4, 4e-4  # Ply thickness limits [m]
T_PLY_AVG = (T_PLY_MAX + T_PLY_MIN) / 2
T_S_MIN_PANEL, T_S_MAX_PANEL = T_PLY_AVG*N_PLIES_PANEL, 20e-3  # T_PLY_AVG*N_LAYERS - 25 mm per face sheet
T_S_MIN_CROSS, T_S_MAX_CROSS = T_PLY_AVG*N_PLIES_CROSS, 35e-3  # T_PLY_AVG*N_LAYERS - 25 mm per face sheet

N_T_S_PANEL = 400
N_T_S_CROSS = 400

# Core thickness search bounds [m]
T_C_MIN_PANEL, T_C_MAX_PANEL = 30e-3, 100e-3   # 5 - 150 mm
T_C_MIN_CROSS, T_C_MAX_CROSS = 15e-3, 180e-3   # 5 - 150 mm

N_T_C_PANEL = 400
N_T_C_CROSS = 400

DELTA_MAX = .040  # Maximum allowable deflection (panel + cross beam) [m]
DELTA_MAX_RATIOS = np.arange(.2, .81, .1)
THICKNESS_MAX = .25  # Maximum allowable thickness (panel + cross beam) [m]

# ===================================================================
# %% PARAMETER VARIATION
# ===================================================================
# Final design selected from the trade-off study:
# Foam: H130
# Skin layup: [0^5/+45/-45]_S
# Panel: t_f = 11.7 mm, t_c = 55.5 mm
# Cross-beam: t_f = 22.5 mm, t_c = 107.0 mm
SELECTED_CASE_PANEL = {
    "t_s": 11.7e-3,
    "t_c": 55.5e-3,
    "foam": "H130",
    "load_case": "LC1",
}

SELECTED_CASE_CROSS = {
    "t_s": 22.5e-3,
    "t_c": 107.0e-3,
    "foam": "H130",
    "load_case": "LC1",
}
# ===================================================================
# %% MATERIAL DATABASE
# ===================================================================

@dataclass
class FoamGrade:
    name:            str
    sigma_hat_t:  float   # Tensile strength [Pa]
    sigma_hat_c:  float   # Compressive strength [Pa]
    tau_hat_12:  float   # Shear strength [Pa]
    E_c:  float   # Compressive modulus [Pa]
    E_t:  float   # Tensile modulus [Pa]
    G:  float   # Shear modulus [Pa]
    gamma_12:  float   # Shear strain [-]
    density:  float   # kg/m3

# Divinycell H series
with open(Path(__file__).parent
          / "_data" / "Divinycell_H_properties.yml", "r") as file:
    foam_mat = yaml.safe_load(file)

FOAM_NAMES = list(foam_mat["density"].keys())
DIVINYCELL_H: List[FoamGrade] = [
    FoamGrade(name = foam_name,
              sigma_hat_c = foam_mat["compressive_strength"][foam_name],
              sigma_hat_t = foam_mat["tensile_strength"][foam_name],
              tau_hat_12 = foam_mat["shear_strength"][foam_name],
              E_c = foam_mat["compressive_modulus"][foam_name],
              E_t = foam_mat["tensile_modulus"][foam_name],
              G = foam_mat["shear_modulus"][foam_name],
              gamma_12 = foam_mat["shear_strain"][foam_name],
              density = foam_mat["density"][foam_name])
    for foam_name in FOAM_NAMES
]
N_FOAMS = len(DIVINYCELL_H)

# Load steel, wood and glass fiber material data
with open(Path(__file__).parent
          / "_data" / "material_properties.yml", "r") as file:
    mat_data = yaml.safe_load(file)

# Create Glass fiber material object
GFRP_data = mat_data["GFRP"]
GFRP_mat = clt.materials.OrthotropicLamina(E1=GFRP_data["E1"],
                                           E2=GFRP_data["E2"],
                                           G12=GFRP_data["G12"],
                                           nu12=GFRP_data["nu12"],
                                           s_hat_1t=GFRP_data["sigma_hat_1t"],
                                           s_hat_1c=GFRP_data["sigma_hat_1c"],
                                           s_hat_2t=GFRP_data["sigma_hat_2t"],
                                           s_hat_2c=GFRP_data["sigma_hat_2c"],
                                           t_hat_12=GFRP_data["tau_hat_12"],
                                           )
sigma_GFRP_crit = min(GFRP_data["sigma_hat_1t"], GFRP_data["sigma_hat_1c"])

def layup_builder(t, sequence=[0, 90, 90, 0]):
    """
    Build a laminate with a specified total thickness t by repeating the base
    stacking sequence so that the ply thickness remains realistic.

    The function chooses an integer number of repetitions of the base sequence
    such that the individual ply thickness stays as close as possible to the
    target average ply thickness T_PLY_AVG and remains within the allowed ply
    thickness bounds [T_PLY_MIN, T_PLY_MAX].

    Parameters
    ----------
    t : float
        Total laminate thickness [m].
    sequence : list | tuple, optional
        Base stacking sequence of the plies.

    Returns
    -------
    classical_laminate_theory.Laminate
        The laminate object.
    """
    n_sequence = len(sequence)

    # First estimate of the number of repetitions needed to obtain a realistic
    # ply thickness close to T_PLY_AVG.
    n_repeats = max(1, int(round(t / (n_sequence * T_PLY_AVG))))

    # Adjust the repetition count so that the resulting ply thickness stays
    # within the admissible manufacturing bounds.
    t_ply = t / (n_sequence * n_repeats)

    while t_ply > T_PLY_MAX:
        n_repeats += 1
        t_ply = t / (n_sequence * n_repeats)

    while t_ply < T_PLY_MIN and n_repeats > 1:
        n_repeats -= 1
        t_ply = t / (n_sequence * n_repeats)

    angles = [list(sequence)] * n_repeats
    plies = [
        clt.Ply(material=GFRP_mat, theta=angle, thickness=t_ply)
        for angles_group in angles
        for angle in angles_group
    ]

    return clt.Laminate(plies)

# ===================================================================
# %% LOADS
# ===================================================================

g = 9.81  # m/s2

# LC1: uniform pressure 5 t
P_LC1 = 5000.0 * g
q_LC1_PANEL = P_LC1 / W / L # Total distributed load [N/m^2]
q_LC1_CROSS = P_LC1 / 2 / B_CROSS / W

# LC2: 2-tonne patch over 120 mm at mid-span
P_LC2_PANEL   = 2000.0 * g / W  # 'Point' load per width [N/m]
P_LC2_CROSS = 2000.0 * g / 2 / B_CROSS  # 'Point' load per

# ===================================================================
# %% FULL VARIATION CALCULATION
# ===================================================================

def calc_beam(t_s, t_c, L, l_patch, q_LC1, P_LC2, layup):
    t_s = np.atleast_1d(np.asarray(t_s, dtype=float))
    t_c = np.atleast_1d(np.asarray(t_c, dtype=float))
    M1 = l_patch * q_LC1 * (2 * L - l_patch) / 8
    V1 = q_LC1 * l_patch/2
    V2 = P_LC2/2
    M2 = P_LC2/2 * L/2

    t_s_mesh, t_c_mesh = np.meshgrid(t_s, t_c, indexing="ij")
    t_s_mesh = t_s_mesh[..., np.newaxis]
    t_c_mesh = t_c_mesh[..., np.newaxis]

    d = t_s_mesh + t_c_mesh
    skins = [layup_builder(t=t_s_i, sequence=layup) for t_s_i in t_s]

    # Skin dependent parameters from CLT
    A11_skin = np.array([skin.ABD_matrix[0, 0] for skin in skins])
    D11_skin = np.array([skin.ABD_matrix[3, 3] for skin in skins])

    A11_skin = np.reshape(A11_skin, (-1, 1, 1))
    D11_skin = np.reshape(D11_skin, (-1, 1, 1))
    E_f = A11_skin / t_s_mesh

    # Foam dependent parameters
    G_c = np.reshape([foam.G for foam in DIVINYCELL_H], (1, 1, -1))
    E_c = np.reshape([foam.E_t for foam in DIVINYCELL_H], (1, 1, -1))
    E_cc = np.reshape([foam.E_c for foam in DIVINYCELL_H], (1, 1, -1))
    E_core_flex = np.reshape(
        [0.5 * (foam.E_t + foam.E_c) for foam in DIVINYCELL_H],
        (1, 1, -1),
    )

    # Stiffness parameters
    # d is the distance between the centroids of the two skins
    D_faces = A11_skin * d**2 / 2
    D_skin_local = 2 * D11_skin
    D_core = E_core_flex * t_c_mesh**3 / 12
    D = D_faces + D_skin_local + D_core
    S = G_c * (t_c_mesh + t_s_mesh)**2 / t_c_mesh

    # Maximum deflection
    # w_LC1 = 5/384 * q_LC1*L**4 / D + q_LC1*L**2 / (8 * S)
    w_LC1 = ((L ** 2 / 12 + L * l_patch / 24 - l_patch ** 2 / 48) * S + D) \
        * q_LC1 * l_patch * (L - l_patch / 2) / D / S / 4
    w_LC2 = P_LC2 * (L**3 / (48*D) + L / (4*S))

    # Skin bending stresses
    sigma_s = lambda M: M * E_f * (t_s_mesh/2 + t_c_mesh/2) / D
    sigma_LC1 = sigma_s(M1)
    sigma_LC2 = sigma_s(M2)

    # Laminate curvature and ply-by-ply stress recovery
    kappa_x = lambda M: M / D
    kappa_x_LC1 = kappa_x(M1)
    kappa_x_LC2 = kappa_x(M2)

    def recover_ply_stress_profile(kappa_x_field):
        n_plies = len(skins[0].plies)
        z_ply_skin_mid = np.zeros((len(t_s), n_plies))
        sigma_local = np.zeros((len(t_s), len(t_c), N_FOAMS, n_plies, 3))
        eps_local = np.zeros((len(t_s), len(t_c), N_FOAMS, n_plies, 3))
        sigma_x_global = np.zeros((len(t_s), len(t_c), N_FOAMS, n_plies))
        eps_x_global = np.zeros((len(t_s), len(t_c), N_FOAMS, n_plies))

        for i, skin in enumerate(skins):
            z_bot = -t_s[i] / 2

            for j, ply in enumerate(skin.plies):
                z_local = z_bot + ply.thickness / 2
                z_ply_skin_mid[i, j] = z_local

                # Absolute ply centroid position from sandwich mid-plane
                z_global = t_c_mesh[i, :, 0][:, np.newaxis] / 2 + z_local
                eps_x_map = kappa_x_field[i, :, :] * z_global
                eps_x_global[i, :, :, j] = eps_x_map

                strain_map = np.zeros((3, len(t_c), N_FOAMS))
                strain_map[0, :, :] = eps_x_map
                eps_local_map = np.einsum("ab,bij->ija", ply.T_eps_inv, strain_map)
                sigma_local_map = np.einsum("ab,ijb->ija", ply.Q_12, eps_local_map)

                eps_local[i, :, :, j, :] = eps_local_map
                sigma_local[i, :, :, j, :] = sigma_local_map
                sigma_x_global[i, :, :, j] = sigma_local_map[..., 0]

        return z_ply_skin_mid, eps_local, sigma_local, eps_x_global, sigma_x_global

    (z_ply_skin_mid,
     eps_local_profile_LC1,
     sigma_local_profile_LC1,
     eps_x_profile_LC1,
     sigma_x_profile_LC1) = recover_ply_stress_profile(kappa_x_LC1)

    (_, 
     eps_local_profile_LC2,
     sigma_local_profile_LC2,
     eps_x_profile_LC2,
     sigma_x_profile_LC2) = recover_ply_stress_profile(kappa_x_LC2)

    # Maximum ply stress over the thickness, kept for the original failure checks
    sigma_local_LC1 = np.max(np.abs(sigma_local_profile_LC1), axis=3)
    sigma_local_LC2 = np.max(np.abs(sigma_local_profile_LC2), axis=3)

    # Ply failure (Tsai-Hill)
    def failure_tsai_hill(sigma, s_hat_1, s_hat_2, t_hat_12):
        s1 = sigma[..., 0]
        s2 = sigma[..., 1]
        t12 = sigma[..., 2]

        return (s1/s_hat_1)**2 - (s1*s2)/(s_hat_1**2) + (s2/s_hat_2)**2 \
            + (t12/t_hat_12)**2

    def failure_tsai_wu(sigma, s_hat_1t, s_hat_1c, s_hat_2t, s_hat_2c, t_hat_12):
        s1 = sigma[..., 0]
        s2 = sigma[..., 1]

        f1 = 1/s_hat_1t - 1/s_hat_1c
        f2 = 1/s_hat_2t - 1/s_hat_2c
        f11 = 1/(s_hat_1t*s_hat_1c)
        f22= 1/(s_hat_2t*s_hat_2c)
        f12 = -.5*np.sqrt(f11*f22)

        return f1*s1 + f2*s2 + f11*s1**2 + f22*s2**2 + 2*f12*s1*s2

    strength_tensile = (GFRP_mat.s_hat_1t, GFRP_mat.s_hat_2t, GFRP_mat.t_hat_12)
    strength_compressive = (GFRP_mat.s_hat_1c, GFRP_mat.s_hat_2c, GFRP_mat.t_hat_12)

    tsaihill_LC1 = np.min([failure_tsai_hill(sigma_local_LC1,
                                             *strength_tensile),
                           failure_tsai_hill(-sigma_local_LC1,
                                             *strength_compressive)],
                          axis=0)
    tsaihill_LC2 = np.min([failure_tsai_hill(sigma_local_LC2,
                                             *strength_tensile),
                           failure_tsai_hill(-sigma_local_LC2,
                                             *strength_compressive)],
                          axis=0)

    strength_tensile = GFRP_mat.strength_as_dict()
    strength_comp = {key: -val for key, val in GFRP_mat.strength_as_dict().items()}
    tsaiwu_LC1 = np.min([failure_tsai_wu(sigma_local_LC1, **strength_tensile),
                         failure_tsai_wu(-sigma_local_LC1, **strength_comp)],
                        axis=0)
    tsaiwu_LC2 = np.min([failure_tsai_wu(sigma_local_LC2, **strength_tensile),
                         failure_tsai_wu(-sigma_local_LC2, **strength_comp)],
                        axis=0)

    # Core shear stresses
    tau_c = lambda V: V / D * (E_f*t_s_mesh*d / 2 + E_c*t_c_mesh**2 / 2)
    tau_LC1 = tau_c(V1)
    tau_LC2 = tau_c(V2)

    # Core shear failure
    tau_hat_12_foam = np.reshape([foam.tau_hat_12 for foam in DIVINYCELL_H],
                                 (1, 1, -1))
    core_failure_LC1 = tau_hat_12_foam - tau_LC1
    core_failure_LC2 = tau_hat_12_foam - tau_LC2

    # Face wrinkling
    sigma_wrinkling = .5 * np.power(E_f * E_cc * G_c, 1/3)

    # Setup dataset for easy access
    ds = xr.Dataset(
        {
            "D": (["t_s", "t_c", "foam"], D),
            "S": (["t_s", "t_c", "foam"], S),
            "w_LC1": (["t_s", "t_c", "foam"], w_LC1),
            "w_LC2": (["t_s", "t_c", "foam"], w_LC2),
            "sigma_LC1": (["t_s", "t_c", "foam"], sigma_LC1),
            "sigma_LC2": (["t_s", "t_c", "foam"], sigma_LC2),
            "kappa_x_LC1": (["t_s", "t_c", "foam"], kappa_x_LC1),
            "kappa_x_LC2": (["t_s", "t_c", "foam"], kappa_x_LC2),
            "eps_local_profile_LC1": (
                ["t_s", "t_c", "foam", "ply", "strain_comp"],
                eps_local_profile_LC1,
            ),
            "eps_local_profile_LC2": (
                ["t_s", "t_c", "foam", "ply", "strain_comp"],
                eps_local_profile_LC2,
            ),
            "sigma_local_profile_LC1": (
                ["t_s", "t_c", "foam", "ply", "stress_comp"],
                sigma_local_profile_LC1,
            ),
            "sigma_local_profile_LC2": (
                ["t_s", "t_c", "foam", "ply", "stress_comp"],
                sigma_local_profile_LC2,
            ),
            "eps_x_profile_LC1": (
                ["t_s", "t_c", "foam", "ply"],
                eps_x_profile_LC1,
            ),
            "eps_x_profile_LC2": (
                ["t_s", "t_c", "foam", "ply"],
                eps_x_profile_LC2,
            ),
            "sigma_x_profile_LC1": (
                ["t_s", "t_c", "foam", "ply"],
                sigma_x_profile_LC1,
            ),
            "sigma_x_profile_LC2": (
                ["t_s", "t_c", "foam", "ply"],
                sigma_x_profile_LC2,
            ),
            "z_ply_skin_mid": (["t_s", "ply"], z_ply_skin_mid),
            "tsaihill_LC1": (["t_s", "t_c", "foam"], tsaihill_LC1),
            "tsaihill_LC2": (["t_s", "t_c", "foam"], tsaihill_LC2),
            "tsaiwu_LC1": (["t_s", "t_c", "foam"], tsaiwu_LC1),
            "tsaiwu_LC2": (["t_s", "t_c", "foam"], tsaiwu_LC2),
            "tau_LC1": (["t_s", "t_c", "foam"], tau_LC1),
            "tau_LC2": (["t_s", "t_c", "foam"], tau_LC2),
            "core_failure_LC1": (["t_s", "t_c", "foam"], core_failure_LC1),
            "core_failure_LC2": (["t_s", "t_c", "foam"], core_failure_LC2),
            "sigma_wrinkle": (["t_s", "t_c", "foam"], sigma_wrinkling),
        },
        coords={
            "t_s": t_s,
            "t_c": t_c,
            "foam": FOAM_NAMES,
            "ply": np.arange(len(skins[0].plies)),
            "stress_comp": ["sigma_1", "sigma_2", "tau_12"],
            "strain_comp": ["eps_1", "eps_2", "gamma_12"],
        },
    )

    return ds

ds_panel = calc_beam(t_s=SELECTED_CASE_PANEL["t_s"],
                     t_c=SELECTED_CASE_PANEL["t_c"],
                     L=L, l_patch=L,
                     q_LC1=q_LC1_PANEL, P_LC2=P_LC2_PANEL, layup=LAYUP_PANEL)

ds_cross = calc_beam(t_s=SELECTED_CASE_CROSS["t_s"],
                     t_c=SELECTED_CASE_CROSS["t_c"],
                     L=L_CROSS, l_patch=W,
                     q_LC1=q_LC1_CROSS, P_LC2=P_LC2_CROSS, layup=LAYUP_CROSS)


def build_through_thickness_profile(ds, layup, foam_name, t_s_idx, t_c_idx,
                                    lc="LC1", n_core_points=41,
                                    n_points_per_ply=9):
    t_s_val = float(ds.t_s.isel(t_s=t_s_idx).values)
    t_c_val = float(ds.t_c.isel(t_c=t_c_idx).values)
    kappa_val = float(ds[f"kappa_x_{lc}"].sel(foam=foam_name).isel(
        t_s=t_s_idx, t_c=t_c_idx
    ).values)

    skin = layup_builder(t=t_s_val, sequence=layup)
    foam = next(mat for mat in DIVINYCELL_H if mat.name == foam_name)
    E_core = 0.5 * (foam.E_t + foam.E_c)

    z_profile = []
    eps_profile = []
    sigma_profile = []
    region_profile = []

    # Bottom skin: sample multiple points inside each ply
    z_cursor = -t_c_val / 2 - t_s_val
    for ply in skin.plies:
        z_bot = z_cursor
        z_top = z_cursor + ply.thickness
        z_points = np.linspace(z_bot, z_top, n_points_per_ply)

        for z in z_points:
            eps_global = np.array([kappa_val * z, 0.0, 0.0])
            eps_local = ply.T_eps_inv @ eps_global
            sigma_local = ply.Q_12 @ eps_local

            z_profile.append(z)
            eps_profile.append(eps_global[0])
            sigma_profile.append(sigma_local[0])
            region_profile.append("bottom_skin")

        z_cursor = z_top

    # Core: continuous sampling through the thickness
    z_core = np.linspace(-t_c_val / 2, t_c_val / 2, n_core_points)
    for z in z_core:
        eps_x = kappa_val * z
        sigma_x = E_core * eps_x

        z_profile.append(z)
        eps_profile.append(eps_x)
        sigma_profile.append(sigma_x)
        region_profile.append("core")

    # Top skin: sample multiple points inside each ply
    z_cursor = t_c_val / 2
    for ply in skin.plies:
        z_bot = z_cursor
        z_top = z_cursor + ply.thickness
        z_points = np.linspace(z_bot, z_top, n_points_per_ply)

        for z in z_points:
            eps_global = np.array([kappa_val * z, 0.0, 0.0])
            eps_local = ply.T_eps_inv @ eps_global
            sigma_local = ply.Q_12 @ eps_local

            z_profile.append(z)
            eps_profile.append(eps_global[0])
            sigma_profile.append(sigma_local[0])
            region_profile.append("top_skin")

        z_cursor = z_top

    return {
        "z": np.array(z_profile),
        "eps_x": np.array(eps_profile),
        "sigma_x": np.array(sigma_profile),
        "region": np.array(region_profile),
        "t_s": t_s_val,
        "t_c": t_c_val,
        "foam": foam_name,
        "load_case": lc,
    }

# ===================================================================
# %% SINGLE CASE POST-PROCESSING
# ===================================================================

def summarize_single_case(ds, case_name, foam_name):
    print(f"\n--- {case_name} ---")
    print(f"Foam grade: {foam_name}")
    print(f"Skin thickness t_s = {float(ds.t_s.values[0])*1e3:.2f} mm")
    print(f"Core thickness t_c = {float(ds.t_c.values[0])*1e3:.2f} mm")
    print(f"D = {ds['D'].sel(foam=foam_name).values.item():.3e} N m")
    print(f"S = {ds['S'].sel(foam=foam_name).values.item():.3e} N")
    print(f"w_LC1 = {ds['w_LC1'].sel(foam=foam_name).values.item()*1e3:.3f} mm")
    print(f"w_LC2 = {ds['w_LC2'].sel(foam=foam_name).values.item()*1e3:.3f} mm")
    print(f"sigma_LC1 = {ds['sigma_LC1'].sel(foam=foam_name).values.item()/1e6:.3f} MPa")
    print(f"sigma_LC2 = {ds['sigma_LC2'].sel(foam=foam_name).values.item()/1e6:.3f} MPa")
    print(f"tau_LC1 = {ds['tau_LC1'].sel(foam=foam_name).values.item()/1e6:.3f} MPa")
    print(f"tau_LC2 = {ds['tau_LC2'].sel(foam=foam_name).values.item()/1e6:.3f} MPa")
    print(f"Tsai-Hill LC1 = {ds['tsaihill_LC1'].sel(foam=foam_name).values.item():.4f}")
    print(f"Tsai-Hill LC2 = {ds['tsaihill_LC2'].sel(foam=foam_name).values.item():.4f}")
    print(f"Tsai-Wu LC1 = {ds['tsaiwu_LC1'].sel(foam=foam_name).values.item():.4f}")
    print(f"Tsai-Wu LC2 = {ds['tsaiwu_LC2'].sel(foam=foam_name).values.item():.4f}")
    print(f"Face wrinkling reserve LC1 = {(ds['sigma_wrinkle'].sel(foam=foam_name).values.item() - ds['sigma_LC1'].sel(foam=foam_name).values.item())/1e6:.3f} MPa")
    print(f"Face wrinkling reserve LC2 = {(ds['sigma_wrinkle'].sel(foam=foam_name).values.item() - ds['sigma_LC2'].sel(foam=foam_name).values.item())/1e6:.3f} MPa")


def plot_through_thickness_profile(profile, title_prefix=""):
    z_mm = profile["z"] * 1e3
    sigma_mpa = profile["sigma_x"] / 1e6
    region = profile["region"]

    fig, ax = plt.subplots(figsize=(6.5, 7.0), constrained_layout=True)

    # Background shading to visually separate the three regions
    region_order = ["bottom_skin", "core", "top_skin"]
    region_labels = {
        "bottom_skin": "Bottom skin",
        "core": "Core",
        "top_skin": "Top skin",
    }
    region_styles = {
        "bottom_skin": {"linewidth": 2.8, "linestyle": "-", "color": "#1f77b4"},
        "core": {"linewidth": 2.2, "linestyle": "-", "color": "#2ca02c"},
        "top_skin": {"linewidth": 2.8, "linestyle": "-", "color": "#d62728"},
    }
    region_fill = {
        "bottom_skin": "#dbe9f6",
        "core": "#e8f5e9",
        "top_skin": "#f9e0e0",
    }

    y_limits = []
    for reg in region_order:
        mask = region == reg
        if np.any(mask):
            y_reg = z_mm[mask]
            y_min = np.min(y_reg)
            y_max = np.max(y_reg)
            y_limits.append((y_min, y_max))
            ax.axhspan(y_min, y_max, color=region_fill[reg], alpha=0.55, zorder=0)

    # Plot the stress profile region by region
    for reg in region_order:
        mask = region == reg
        if np.any(mask):
            ax.plot(
                sigma_mpa[mask],
                z_mm[mask],
                label=region_labels[reg],
                zorder=3,
                **region_styles[reg],
            )

    # Zero-stress reference line
    ax.axvline(0.0, color="0.25", linewidth=1.1, linestyle="--", zorder=1)

    # Horizontal lines at the interfaces between skins and core
    if len(y_limits) == 3:
        ax.axhline(y_limits[0][1], color="0.35", linewidth=1.0, linestyle=":", zorder=2)
        ax.axhline(y_limits[1][1], color="0.35", linewidth=1.0, linestyle=":", zorder=2)

    # Region labels on the right side of the plot
    x_min, x_max = np.min(sigma_mpa), np.max(sigma_mpa)
    x_text = x_max + 0.06 * max(1.0, x_max - x_min)
    for reg in region_order:
        mask = region == reg
        if np.any(mask):
            y_reg = z_mm[mask]
            y_mid = 0.5 * (np.min(y_reg) + np.max(y_reg))
            ax.text(
                x_text,
                y_mid,
                region_labels[reg],
                va="center",
                ha="left",
                fontsize=10,
                color="0.25",
            )

    ax.set_xlabel(r"$\sigma_x\;[\mathrm{MPa}]$", fontsize=12)
    ax.set_ylabel(r"$z\;[\mathrm{mm}]$", fontsize=12)
    ax.set_title(
        f"{title_prefix} {profile['load_case']} through-thickness stress profile".strip(),
        fontsize=13,
        pad=12,
    )
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=10)
    ax.margins(x=0.08, y=0.02)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    return fig, ax


foam_panel = SELECTED_CASE_PANEL["foam"]
foam_cross = SELECTED_CASE_CROSS["foam"]

summarize_single_case(ds_panel, "PANEL CASE", foam_panel)
summarize_single_case(ds_cross, "CROSS-BEAM CASE", foam_cross)

profile_panel = build_through_thickness_profile(
    ds=ds_panel,
    layup=LAYUP_PANEL,
    foam_name=foam_panel,
    t_s_idx=0,
    t_c_idx=0,
    lc=SELECTED_CASE_PANEL["load_case"],
)

profile_cross = build_through_thickness_profile(
    ds=ds_cross,
    layup=LAYUP_CROSS,
    foam_name=foam_cross,
    t_s_idx=0,
    t_c_idx=0,
    lc=SELECTED_CASE_CROSS["load_case"],
)

plot_through_thickness_profile(profile_panel, title_prefix="Panel")
plot_through_thickness_profile(profile_cross, title_prefix="Cross-beam")
plt.show()

