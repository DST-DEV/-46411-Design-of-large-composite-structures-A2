from pathlib import Path
from typing import List
from dataclasses import dataclass

import cmcrameri.cm as cmc
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
B = 0.12  # Beam width [m]
T_PLY_MIN , T_PLY_MAX = 2e-4, 4e-4  # Ply thickness limits [m]
T_PLY_AVG = (T_PLY_MAX + T_PLY_MIN) / 2
LAYUP_STR = "[0/90]_S"
LAYUP , _ = clt.laminate.builder.StackParser.parse(LAYUP_STR)

# Skin thickness search bounds [m]
T_S_MIN, T_S_MAX = T_PLY_AVG*len(LAYUP), 18e-3  # T_PLY_AVG*N_LAYERS - 18 mm per face sheet
N_T_S = 400

# Core thickness search bounds [m]
T_C_MIN, T_C_MAX = 5e-3, 150e-3   # 5 - 150 mm
N_T_C = 400

DELTA_MAX = .020  # allowable mid-span deflection [m]

# ===================================================================
# %% PARAMETER VARIATION
# ===================================================================

t_s = np.linspace(T_S_MIN, T_S_MAX, N_T_S)
t_c = np.linspace(T_C_MIN, T_C_MAX, N_T_C)

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
    Build a laminate with a specified thickness t by repeating a layup.

    The thickness of each ply is determined so that it lies close to the
    user-defined average ply thickness. Each ply of the layup is then repeated
    to fulfill the specified thickness.

    Parameters
    ----------
    t_s : float
        Thickness of the laminate.
    sequence : list | tuple, optional
        Stacking sequence of the plies. The default is [0, 90, 90, 0].

    Returns
    -------
    classical_laminate_theory.Laminate
        The laminate object.

    """
    t_ply = t / len(sequence)
    plies = [clt.Ply(material=GFRP_mat, theta=angle, thickness=t_ply)
             for angle in sequence]

    return clt.Laminate(plies)

# ===================================================================
# %% LOADS
# ===================================================================

g = 9.81  # m/s2

# LC1: uniform pressure 5 t
P_LC1 = 5000.0 * g
q_LC1 = P_LC1 / W / L # Total distributed load [N/m^2]

# LC2: 2-tonne patch over 120 mm at mid-span
P_LC2   = 2000.0 * g / W  # 'Point' load per width [N/m]

# Calculate resulting cross-sectional loads
M1 = q_LC1 * L**2 / 8
V1 = q_LC1 * L/2
V2 = P_LC2/2
M2 = P_LC2/2 * L/2

# ===================================================================
# %% FULL VARIATION CALCULATION
# ===================================================================

t_s_mesh, t_c_mesh = np.meshgrid(t_s, t_c, indexing="ij")
t_s_mesh = t_s_mesh[..., np.newaxis]
t_c_mesh = t_c_mesh[..., np.newaxis]

d = t_s_mesh + t_c_mesh
skins = [layup_builder(t=t_s_i, sequence=LAYUP) for t_s_i in t_s]

# Skin dependent parameters
A_11_inv = np.array([np.linalg.inv(skin.ABD_matrix[:3, :3])[0, 0]
                     for skin in skins])
E_f = np.reshape(1 / (A_11_inv * t_s), (-1, 1, 1))

# Foam dependent parameters
G_c = np.reshape([foam.G for foam in DIVINYCELL_H], (1, 1, -1))
E_c = np.reshape([foam.E_t for foam in DIVINYCELL_H], (1, 1, -1))
E_cc = np.reshape([foam.E_c for foam in DIVINYCELL_H], (1, 1, -1))

# Stiffness parameters
D = E_f * t_s_mesh * d**2 / 2
S = G_c * (t_c_mesh + t_s_mesh)**2 / t_c_mesh

# Maximum deflection
w_LC1 = 5/384 * q_LC1*L**4 / D + q_LC1*L**2 / (8 * S)
w_LC2 = P_LC2 * (L**3 / (48*D) + L / (4*S))
# w_LC2_new = (L - a / 2) * q_LC2 * a * ((L ** 2 / 12 + L * a / 24 - a ** 2 / 48) * S + D) / D / S / 4

# =============================================================================
# fig, ax = plt.subplots(figsize=(5.5, 5),
#                        constrained_layout=True)
# cf = ax.contourf(t_s_mesh[..., -1], t_c_mesh[..., -1], w_LC2_new[..., -1],
#                  levels=20, cmap=cmc.devon.reversed(), alpha=0.85,
#                  zorder=1)
# cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
#
# fig, ax = plt.subplots(figsize=(5.5, 5),
#                        constrained_layout=True)
# cf = ax.contourf(t_s_mesh[..., -1], t_c_mesh[..., -1], w_LC2[..., -1],
#                  levels=20, cmap=cmc.devon.reversed(), alpha=0.85,
#                  zorder=1)
# cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
#
# fig, ax = plt.subplots(figsize=(5.5, 5),
#                        constrained_layout=True)
# cf = ax.contourf(t_s_mesh[..., -1], t_c_mesh[..., -1], (w_LC2_new - w_LC2)[..., -1],
#                  levels=20, cmap=cmc.devon.reversed(), alpha=0.85,
#                  zorder=1)
# cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
# =============================================================================

# Skin bending stresses
sigma_s = lambda M: M * E_f * (t_s_mesh/2 + t_c_mesh/2) / D
sigma_LC1 = sigma_s(M1)
sigma_LC2 = sigma_s(M2)

# Laminate and ply strains
eps_x = lambda M: M * (t_s_mesh/2 + t_c_mesh/2) / D
eps_x_LC1 = eps_x(M1)
eps_x_LC2 = eps_x(M2)

def eps_1 (eps_x):
    T_eps_inv = np.array([[ply.T_eps_inv[0, 0] for ply in skin.plies]
                          for skin in skins])

    return np.max(np.einsum("ij,ik->ijk", eps_x[..., 0], T_eps_inv),
                  axis=2)[..., np.newaxis]

eps_1_LC1 = eps_1(eps_x_LC1)
eps_1_LC2 = eps_1(eps_x_LC2)

# Ply stresses
E1_local = np.array([skin.plies[0].Q_12[0, 0] for skin in skins])
E21_local = np.array([skin.plies[0].Q_12[1, 0] for skin in skins])

def sigma_local (eps_1):
    return np.stack([eps_1 * E1_local[:, np.newaxis, np.newaxis],
                     eps_1 * E21_local[:, np.newaxis, np.newaxis],
                     np.zeros_like(eps_1)],
                    axis=-1)

sigma_local_LC1 = sigma_local(eps_1_LC1)
sigma_local_LC2 = sigma_local(eps_1_LC2)

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

# =============================================================================
# fig, ax = plt.subplots(figsize=(5.5, 5),
#                        constrained_layout=True)
# cf = ax.contourf(t_s_mesh[..., -1], t_c_mesh[..., -1], tsaiwu_LC1[..., -1],
#                  levels=20, cmap=cmc.devon.reversed(), alpha=0.85,
#                  zorder=1)
# cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
# =============================================================================

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
sigma_wrinkling = np.repeat(sigma_wrinkling, len(t_c), axis=1)

# Setup dataset for easy access
ds = xr.Dataset(
    {
        "D": (["t_s", "t_c", "foam"], np.repeat(D, N_FOAMS, axis=2)),
        "S": (["t_s", "t_c", "foam"], S),
        "w_LC1": (["t_s", "t_c", "foam"], w_LC1),
        "w_LC2": (["t_s", "t_c", "foam"], w_LC2),
        "sigma_LC1": (["t_s", "t_c", "foam"],
                      np.repeat(sigma_LC1, N_FOAMS, axis=2)),
        "sigma_LC2": (["t_s", "t_c", "foam"],
                      np.repeat(sigma_LC2, N_FOAMS, axis=2)),
        "tsaihill_LC1": (["t_s", "t_c", "foam"],
                         np.repeat(tsaihill_LC1, N_FOAMS, axis=2)),
        "tsaihill_LC2": (["t_s", "t_c", "foam"],
                         np.repeat(tsaihill_LC2, N_FOAMS, axis=2)),
        "tsaiwu_LC1": (["t_s", "t_c", "foam"],
                         np.repeat(tsaiwu_LC1, N_FOAMS, axis=2)),
        "tsaiwu_LC2": (["t_s", "t_c", "foam"],
                         np.repeat(tsaiwu_LC2, N_FOAMS, axis=2)),
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
    },
)

# ===================================================================
# %% PLOT COMPARISONS
# ===================================================================

rc_params = scivis.rcparams._prepare_rcparams()
t_s_lgd = np.linspace(T_S_MIN, T_S_MAX, 10)*1e3
t_c_lgd = np.linspace(T_C_MIN, T_C_MAX, 10)*1e3

def add_colorbar(ax, rc_params, vmin, vmax, ticks, lbl="", fontsize_factor=.85):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    formatter = FuncFormatter(lambda x, pos: f"{x:.3f}")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="plasma"),
                        ax=ax, ticks=ticks, format=formatter)
    cbar.ax.tick_params(direction="out", length=10, width=1.5)
    # enforce rc font
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(rc_params["font.family"])
        label.set_fontsize(rc_params["font.size"]*fontsize_factor)

    if lbl:
        cbar.set_label(lbl, fontsize=rc_params["axes.labelsize"]*fontsize_factor)

# with mpl.rc_context(rc_params):
#     fig, ax, _ = scivis.plot_line(t_s, ds["w_LC1"].sel(foam="H80").values.T,
#                                   linestyles="-", cmap="plasma",
#                                   show_legend=False)
#     add_colorbar(ax=ax, rc_params=rc_params, vmin=T_S_MIN*1e3, vmax=T_S_MAX*1e3,
#                  ticks=t_s_lgd, lbl=r"$t_s\:\text{[mm]}$")

#     fig, ax, _ = scivis.plot_line(t_s, ds["sigma_LC1"].sel(foam="H80").values.T,
#                                   linestyles="-", cmap="plasma",
#                                   show_legend=False)
#     add_colorbar(ax=ax, rc_params=rc_params, vmin=T_S_MIN*1e3, vmax=T_S_MAX*1e3,
#                  ticks=t_s_lgd, lbl=r"$t_s\:\text{[mm]}$")


def plot_limit_boundary(foam, constraint_thin=True, show_legend=False):
    rc_params = {"font.family": "serif",
                 "font.sans-serif": ["Times New Roman"],
                 'mathtext.fontset': 'cm'}

    with mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(5.5, 5),
                               constrained_layout=True)

        # Coordinate grids in mm for display
        t_s_mm, t_c_mm = np.meshgrid(ds.t_s.values * 1e3,
                                     ds.t_c.values * 1e3, indexing="ij")
        t_total = 2 * t_s_mm + t_c_mm

        # Contour fill: total thickness
        cf = ax.contourf(t_s_mm, t_c_mm, t_total,
                         levels=20, cmap=cmc.devon.reversed(), alpha=0.85,
                         zorder=1)

        # Feasibility mask (both limits satisfied)
        cond_w1 = ds["w_LC1"].sel(foam=foam).values <= DELTA_MAX
        cond_w2 = ds["w_LC2"].sel(foam=foam).values <= DELTA_MAX
        cond_tsaihill_1 = ds["tsaihill_LC1"].sel(foam=foam).values >0
        cond_tsaihill_2 = ds["tsaihill_LC2"].sel(foam=foam).values >0
        cond_core_failure_1 = ds["core_failure_LC1"].sel(foam=foam).values >0
        cond_core_failure_2 = ds["core_failure_LC2"].sel(foam=foam).values >0
        cond_wrinkling_1 = (ds["sigma_wrinkle"].sel(foam=foam).values
                            - ds["sigma_LC1"].sel(foam=foam).values) >0
        cond_wrinkling_2 = (ds["sigma_wrinkle"].sel(foam=foam).values
                            - ds["sigma_LC2"].sel(foam=foam).values) >0
        feasible = cond_w1 & cond_w2 & cond_tsaihill_1 & cond_tsaihill_2 \
            & cond_core_failure_1 & cond_core_failure_2 & cond_wrinkling_1 \
            & cond_wrinkling_2

        if constraint_thin: # Enforce thin skin
            cond_thin_skin = ((t_s_mm + t_c_mm) / t_s_mm) > 5.77
            feasible = feasible & cond_thin_skin
        feasible_float = feasible.astype(float)

        # Hatch overlay on the feasible region
        ax.contourf(t_s_mm, t_c_mm, feasible_float, levels=[0.5, 1.5],
                    colors="none", hatches=["////"], alpha=0.0, zorder=3)

        # Constraing boundary line
        ax.contour(t_s_mm, t_c_mm, feasible_float, levels=[0.5],
                   colors=["k"], linewidths=2, zorder=3)

        # Axes ticks
        ax.set_xlabel(r"$t_s\:\text{[mm]}$", fontsize=13)
        ax.set_ylabel(r"$t_c\:\text{[mm]}$", fontsize=13)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.grid(visible=True, color="0.85", linewidth=0.5, alpha=0.4, zorder=2)

        # Colorbar
        cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(r"$t_\text{beam}\:\text{[mm]}$", fontsize=14)

        # Legend
        if show_legend:
            legend_elements = [
                Line2D([0], [0], color="k", lw=2.5, ls="-",
                       label="Combined feasibility boundary"),
                plt.Rectangle((0, 0), 1, 1, fc="white", alpha=0.3,
                              hatch="////", ec="gray",
                              label="Feasible region (all conditions met)"),
            ]
            fig.legend(handles=legend_elements, loc="lower center",
                       fontsize=10, framealpha=0.9,
                       bbox_to_anchor=(0.5, -0.10))

    return fig, ax

def plot_limit_boundary_overlap(constraint_thin=True):
    rc_params = {"font.family": "serif",
                 "font.sans-serif": ["Times New Roman"],
                 'mathtext.fontset': 'cm'}

    with mpl.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(6, 5),
                               constrained_layout=True)

        # Coordinate grids in mm for display
        t_s_mm, t_c_mm = np.meshgrid(ds.t_s.values * 1e3,
                                     ds.t_c.values * 1e3, indexing="ij")
        t_total = 2 * t_s_mm + t_c_mm

        # Contour fill: total thickness
        cf = ax.contourf(t_s_mm, t_c_mm, t_total,
                         levels=20, cmap=cmc.devon.reversed(), alpha=0.85,
                         zorder=1)

        legend_elements = []
        # colors = mpl.colormaps['autumn'](np.linspace(0, 1, N_FOAMS))
        colors = cmc.lajolla.reversed()(np.linspace(0, 1, N_FOAMS))
        for i, foam in enumerate(FOAM_NAMES):
            # Feasibility mask (both limits satisfied)
            cond_w1 = ds["w_LC1"].sel(foam=foam).values <= DELTA_MAX
            cond_w2 = ds["w_LC2"].sel(foam=foam).values <= DELTA_MAX
            cond_tsaihill_1 = ds["tsaihill_LC1"].sel(foam=foam).values >0
            cond_tsaihill_2 = ds["tsaihill_LC2"].sel(foam=foam).values >0
            cond_core_failure_1 = ds["core_failure_LC1"].sel(foam=foam).values >0
            cond_core_failure_2 = ds["core_failure_LC2"].sel(foam=foam).values >0
            cond_wrinkling_1 = (ds["sigma_wrinkle"].sel(foam=foam).values
                                - ds["sigma_LC1"].sel(foam=foam).values) >0
            cond_wrinkling_2 = (ds["sigma_wrinkle"].sel(foam=foam).values
                                - ds["sigma_LC2"].sel(foam=foam).values) >0
            feasible = cond_w1 & cond_w2 & cond_tsaihill_1 & cond_tsaihill_2 \
                & cond_core_failure_1 & cond_core_failure_2 & cond_wrinkling_1 \
                & cond_wrinkling_2

            if constraint_thin:  # Enforce thin skin
                cond_thin_skin = ((t_s_mm + t_c_mm) / t_s_mm) > 5.77
                feasible = feasible & cond_thin_skin
            feasible_float = feasible.astype(float)

            # Constraing boundary line
            ax.contour(t_s_mm, t_c_mm, feasible_float, levels=[0.5],
                       colors=[colors[i]], linewidths=2, zorder=4)

            legend_elements.append(
                Line2D([0], [0], color=colors[i], lw=2.5, ls="-", label=foam))

        # Hatch overlay on the feasible region (only of the densest foam)
        ax.contourf(t_s_mm, t_c_mm, feasible_float, levels=[0.5, 1.5],
                    colors="none", hatches=["////"], alpha=0.0, zorder=3)
        # legend_elements.append(
        #     plt.Rectangle((0, 0), 1, 1, fc="white", alpha=0.3,
        #                   hatch="////", ec="0.2",
        #                   label="Feasible region"))

        # Axes ticks
        ax.set_xlabel(r"$t_s\:\text{[mm]}$", fontsize=13)
        ax.set_ylabel(r"$t_c\:\text{[mm]}$", fontsize=13)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.grid(visible=True, color="0.85", linewidth=0.5, alpha=0.4, zorder=2)

        # Colorbar
        cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(r"$t_\text{beam}\:\text{[mm]}$", fontsize=14)

        fig.legend(handles=legend_elements, loc="center left",
                   fontsize=10, framealpha=0.9,
                   bbox_to_anchor=(1.02, .5))

    return fig, ax

# fig, ax = plot_limit_boundary(DIVINYCELL_H[0].name)
# fig, ax = plot_limit_boundary(DIVINYCELL_H[-1].name)
fig, ax = plot_limit_boundary_overlap(constraint_thin=True)
plt.show()
