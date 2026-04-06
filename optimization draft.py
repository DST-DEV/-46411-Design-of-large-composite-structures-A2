"""
Sandwich Beam Weight Optimizer
================================
Simply supported sandwich beam with GFRP skins and Divinycell H foam core.

Design variables:
  - t_skin : skin (face sheet) thickness [m]  (symmetric, top = bottom)
  - t_core : core thickness [m]
  - foam   : Divinycell H grade (discrete choice — iterated over)

Constraints (must all be satisfied simultaneously):
  C1.  Mid-span deflection  <=  delta_max (LC1 and LC2)
  C2.  Skin bending stress  <=  sigma_allow (LC1 and LC2)
  C3.  Core shear stress    <=  tau_allow (LC1 and LC2)
  C3.  Skin bending stress  <=  Face wrinkling stress (LC1 and LC2)

Load cases:
  LC1 – Uniform pressure q = 5 t/m^2 over the panel area
  LC2 – Patch load P = 2 t distributed over a 120 mm centred patch

Fixed geometry:
  L = 4 m span,  b = beam width (configurable)

Optimisation strategy:
  Outer loop  -> discrete: iterate over all 8 Divinycell H grades
  Inner loop  -> continuous: scipy.optimize.minimize (SLSQP) over (t_skin, t_core)
               with multiple restarts to escape local minima, plus a
               Differential Evolution fallback for difficult cases.
  Objective   -> total beam mass [kg]

References for formulae:
  - Zenkert, D. (1995). An Introduction to Sandwich Structures.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List
import warnings

import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
import yaml

import classical_laminate_theory as clt


warnings.filterwarnings("ignore")

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
N_PLIES = len(LAYUP)

DELTA_MAX = .040  # allowable mid-span deflection [m]

# Skin thickness search bounds [m]
T_S_MIN, T_S_MAX = T_PLY_AVG*len(LAYUP), 20e-3  # T_PLY_AVG*N_LAYERS - 25 mm per face sheet

# Core thickness search bounds [m]
T_C_MIN, T_C_MAX = 5e-3, 150e-3   # 5 - 150 mm

N_RESTARTS = 8  # Grid restarts per foam grade per axis. Total starts = N_RESTARTS^2.

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
    for foam_name in foam_mat["density"].keys()
]

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
sigma_hat_GFRP = GFRP_mat.strength_as_dict()

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
q_LC1 = 5000.0 * g / L  # Total distributed load [N/m^2]

# LC2: 2-tonne patch over 120 mm at mid-span
P_LC2   = 2000.0 * g      # Total force [N]

# ===================================================================
# %% SANDWICH BEAM MECHANICS
# ===================================================================

def skin_stiffness(skin, t_s):
    return 1 / (np.linalg.inv(skin.ABD_matrix[:3, :3])[0, 0] * t_s)

def shear_stiffness_core(t_s, t_c, G_c):
    return G_c * (t_c + t_s)**2 / t_c

def bending_stiffness(t_s, t_c, E_f, E_c=0):
    d = t_c + t_s
    # return (E_f * t_s * d**2 / 2)  +  (E_f * t_s**3 / 6) + (E_c * t_c**3 / 12)
    return (E_f * t_s * d**2 / 2)

def bending_stiffness_automatic(t_s, t_c, skin, E_c=0):
    skin = layup_builder(t_s, LAYUP)
    E_f = skin_stiffness(skin=skin, t_s=t_s)

    return bending_stiffness(t_s=t_s, t_c=t_c, E_f=E_f, E_c=E_c)

def stiffness(t_s, t_c, skin, foam):
    """
    Bending stiffness D [N*m] and shear stiffness S [N/m] using classical
    (Zenkert) sandwich theory.
    """

    D = bending_stiffness_automatic(t_s=t_s, t_c=t_c, skin=skin, E_c=foam.E_t)
    S = shear_stiffness_core(t_s=t_s, t_c=t_c, G_c=foam.G)
    return D, S

def deflection_LC1(D, S):
    """
    Mid-span deflection of a simply supported sandwich beam under uniform
    pressure q [N/m]:

    w = q*L**4 / (24*D) * ((x/L)**4 - 2*(x/L)**3 + (x/L))
        + q/(2*S) * (L*x - x**2)
    => Maximum at L/2
    => w_max = 5/384 * q*L**4 / D + q*L**2 / (8 * S)
    """
    return 5/384 * q_LC1*L**4 / D + q_LC1*L**2 / (8 * S)

def deflection_LC2(D, S):
    """
    Mid-span deflection under a symmetric patch load of intensity w_LC2 [N/m]
    over length a_patch centred at mid-span (simplified to point load).

    w(x <= L/2) = (P * L**3) / (6*D) * b * ((1 - b**2) * (x/L) - (x/L)**3) \
                  + (P * b * x)/S
    => Maximum at L/2
    => w_max = P * (L**3 / (48*D) + L / (4*S))
    """
    return P_LC2 * (L**3 / (48*D) + L / (4*S))

def max_moment_shear():
    """
    Returns (M_LC1, V_LC1, M_LC2, V_LC2) in Nm and N.

    LC1 (uniform load):
        M_max = q*L^2/8   (at mid-span)
        V_max = q*L/2     (at support)

    LC2 (symmetric patch load at mid-span (treated as a point load)):
        V_max = P/2  (at support)
        M_max = P/2 * L/2  (at mid-span)
    """
    M1 = q_LC1 * L**2 / 8
    V1 = q_LC1 * L/2

    P = P_LC2
    V2 = P/2
    M2 = P/2 * L/2
    return M1, V1, M2, V2

def skin_bending_stress(M, t_s, t_c, E_f, D):
    """
    Peak tensile/compressive bending stress in the face sheets [Pa].
    """

    return M * E_f * (t_s + t_c/2) / D

def skin_bending_stress_standalone(M, t_s, t_c, skin, foam):
    """
    Peak tensile/compressive bending stress in the face sheets [Pa].
    """
    E_f = skin_stiffness(skin=skin, t_s=t_s)
    D = bending_stiffness(t_s=t_s, t_c=t_c, E_f=E_f, E_c=foam.E_t)

    return skin_bending_stress(M=M, t_s=t_s, t_c=t_c, E_f=E_f, D=D)

def core_shear_stress(V, t_s, t_c, E_f, E_c, D):
    """
    Maximum shear stress in the core [Pa].
    """
    d = (t_s + t_c)
    return V / D * (E_f*t_s*d / 2 + E_c*t_c**2 / 2)

def core_shear_stress_standalone(V, t_s, t_c, skin, foam):
    """
    Maximum shear stress in the core [Pa].
    """
    E_f = skin_stiffness(skin=skin, t_s=t_s)
    E_c = foam.E_t
    D = bending_stiffness(t_s=t_s, t_c=t_c, E_f=E_f, E_c=E_c)

    return core_shear_stress(V=V, t_s=t_s, t_c=t_c, E_f=E_f, E_c=E_c, D=D)

def skin_ply_stresses_max(M, t_s, t_c, skin, foam):
    E_f = skin_stiffness(skin=skin, t_s=t_s)
    D = bending_stiffness(t_s=t_s, t_c=t_c, E_f=E_f, E_c=foam.E_t)

    # Calculate strains (assuming constant stress over laminate)
    # sigma = M * E_f * (y + t_c/2) / D
    # eps_x = sigma / E_f = M * (y + t_c/2) / D
    eps_x = M * (t_s/2 + t_c/2) / D
    eps_1 = eps_x * np.array([ply.T_eps_inv[0, 0] for ply in skin.plies])
    eps_1_max = np.max(eps_1)

    # Calculate local stresses
    # Note: all the plies are the same => local stiffness is the same
    E1_local = skin.plies[0].Q_12[0, 0]
    E21_local = skin.plies[0].Q_12[1, 0]
    sigma_local_max = np.array([eps_1_max * E1_local,
                                eps_1_max * E21_local,
                                0])

    return sigma_local_max

def skin_failure_yielding(M, t_s, t_c, skin, foam):
    sigma_local_max = skin_ply_stresses_max(M=M, t_s=t_s, t_c=t_c, skin=skin,
                                            foam=foam)

    idx_tsaihill_tension = clt.failure.TsaiHill.failure_index(
        stress=sigma_local_max, **sigma_hat_GFRP)
    idx_tsaihill_compression = clt.failure.TsaiHill.failure_index(
        stress=-sigma_local_max, **sigma_hat_GFRP)

    return min(idx_tsaihill_tension, idx_tsaihill_compression)

def core_shear_failure(V, t_s, t_c, skin, foam):
    tau_12 = core_shear_stress_standalone(V=V, t_s=t_s, t_c=t_c, skin=skin,
                                          foam=foam)

    failure_idx = foam.tau_hat_12 - tau_12

    return failure_idx

def face_wrinkling(M, t_s, t_c, skin, foam):
    E_f = skin_stiffness(skin=skin, t_s=t_s)
    D = bending_stiffness(t_s=t_s, t_c=t_c, E_f=E_f, E_c=foam.E_t)

    sigma_max = skin_bending_stress(M=M, t_s=t_s, t_c=t_c, E_f=E_f, D=D)
    sigma_wrinkling = .5 * np.power(E_f * foam.E_c * foam.G, 1/3)

    return sigma_wrinkling - sigma_max


# Pre-compute moments and shears (since they are geometry-independent)
M1, V1, M2, V2 = max_moment_shear()

# ===================================================================
# %% FULL DESIGN EVALUATION
# ===================================================================

def evaluate(t_s, t_c, foam):
    """
    Compute all performance metrics for a given design.
    Returns a result dict, or None if dimensions are non-physical.
    """
    if t_s <= 0 or t_c <= 0:
        return None

    skin = layup_builder(t_s, LAYUP)
    D, S = stiffness(t_s, t_c, skin, foam)
    E_f = skin_stiffness(skin=skin, t_s=t_s)

    d1 = deflection_LC1(D, S)
    d2 = deflection_LC2(D, S)

    sig1 = skin_bending_stress(M=M1, t_s=t_s, t_c=t_c, E_f=E_f, D=D)
    sig2 = skin_bending_stress(M=M2, t_s=t_s, t_c=t_c, E_f=E_f, D=D)
    tau1 = core_shear_stress(V=V1, t_s=t_s, t_c=t_c, E_f=E_f, E_c=foam.E_t,
                             D=D)
    tau2 = core_shear_stress(V=V2, t_s=t_s, t_c=t_c, E_f=E_f, E_c=foam.E_t,
                             D=D)
    sig_wrinkling = .5 * np.power(E_f * foam.E_t * foam.G, 1/3)

    m_sigma_LC1 = skin_failure_yielding(M1, t_s=t_s, t_c=t_c, skin=skin, foam=foam)
    m_sigma_LC2 = skin_failure_yielding(M2, t_s=t_s, t_c=t_c, skin=skin, foam=foam)
    m_tau_LC1 = core_shear_failure(V1, t_s=t_s, t_c=t_c, skin=skin, foam=foam)
    m_tau_LC2 = core_shear_failure(V2, t_s=t_s, t_c=t_c, skin=skin, foam=foam)
    m_wrinkling_LC1 = face_wrinkling(M1, t_s=t_s, t_c=t_c, skin=skin, foam=foam)
    m_wrinkling_LC2 = face_wrinkling(M2, t_s=t_s, t_c=t_c, skin=skin, foam=foam)

    mass = (2*t_s*GFRP_data["rho"] + t_c*foam.density) * B * L

    return dict(
        foam = foam.name,
        t_skin = t_s,
        t_core = t_c,
        t_total = (2*t_s + t_c),
        mass = mass,
        delta_LC1 = d1,
        delta_LC2 = d2,
        delta_max = DELTA_MAX,
        sigma_LC1 = sig1,
        sigma_LC2 = sig2,
        tau_LC1 = tau1,
        tau_LC2 = tau2,
        sig_wrinkling = sig_wrinkling,

        # Constraint margins: positive = satisfied, negative = violated
        m_defl_LC1 = DELTA_MAX - d1,
        m_defl_LC2 = DELTA_MAX - d2,
        m_sigma_LC1 = m_sigma_LC1,
        m_sigma_LC2 = m_sigma_LC2,
        m_tau_LC1 = m_tau_LC1,
        m_tau_LC2 = m_tau_LC2,
        m_wrinkling_LC1 = m_wrinkling_LC1,
        m_wrinkling_LC2 = m_wrinkling_LC2
    )


def feasible(r):
    return all(r[k] >= -1e-9 for k in ['m_defl_LC1', 'm_defl_LC2',
                                       'm_sigma_LC1', 'm_sigma_LC2',
                                       'm_tau_LC1',  'm_tau_LC2'])


# ===================================================================
# %% OPTIMISATION
# ===================================================================

def optimise_for_foam(foam):
    """
    Find the minimum-mass sandwich design for a given foam grade.
    Uses SLSQP with a grid of starting points, then a Differential
    Evolution fallback. Returns the best feasible result or None.
    """

    def obj(x):
        t_s, t_c = x
        return 2*t_s + t_c
        # return (2*t_s*GFRP_data["rho"] + t_c*foam.density) * B * L

    constraints = [
        {'type': 'ineq',
         'fun': lambda x: (x[0] + x[1]) / x[0] - 5.77},  # Ensure thin skin
        {'type': 'ineq',
         'fun': lambda x: (DELTA_MAX
                           - deflection_LC1(*stiffness(x[0], x[1], layup_builder(x[0], LAYUP), foam)))},
        {'type': 'ineq',
         'fun': lambda x: (DELTA_MAX - deflection_LC2(*stiffness(x[0], x[1], layup_builder(x[0], LAYUP), foam)))},
        {'type': 'ineq',
         'fun': lambda x: skin_failure_yielding(M1, x[0], x[1], layup_builder(x[0], LAYUP), foam)},
        {'type': 'ineq',
         'fun': lambda x: skin_failure_yielding(M2, x[0], x[1], layup_builder(x[0], LAYUP), foam)},
        {'type': 'ineq',
         'fun': lambda x: core_shear_failure(V1, x[0], x[1], layup_builder(x[0], LAYUP), foam)},
        {'type': 'ineq',
         'fun': lambda x: core_shear_failure(V2, x[0], x[1], layup_builder(x[0], LAYUP), foam)},
        {'type': 'ineq',
         'fun': lambda x: face_wrinkling(M1, x[0], x[1], layup_builder(x[0], LAYUP), foam)},
        {'type': 'ineq',
         'fun': lambda x: face_wrinkling(M2, x[0], x[1], layup_builder(x[0], LAYUP), foam)},
    ]
    bounds = [(T_S_MIN, T_S_MAX), (T_C_MIN, T_C_MAX)]

    best = None
    starts = [(T_S_MIN + (T_S_MAX-T_S_MIN)*s,
               T_C_MIN + (T_C_MAX-T_C_MIN)*c)
              for s in np.linspace(0.05, 0.95, N_RESTARTS)
              for c in np.linspace(0.05, 0.95, N_RESTARTS)]

    # SLSQP restarts
    for t_s0, t_c0 in starts:
        res = minimize(obj, x0=[t_s0, t_c0], method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'ftol': 1e-11, 'maxiter': 5000})
        if res.success or res.fun < 1e10:
            r = evaluate(res.x[0], res.x[1], foam)
            if r and feasible(r):
                if best is None or r['t_total'] < best['t_total']:
                    best = r

    # Differential Evolution fallback
    if best is None:
        print(f"No feasible solution found for foam {foam.name}. "
              "Starting differential evolution.\n")

        constraints_DE = [NonlinearConstraint(fun = constraint["fun"],
                                              lb=0, ub=np.inf)
                          for constraint in constraints]

        res = differential_evolution(obj, constraints=constraints_DE,
                                     bounds=bounds,
                                     seed=42, maxiter=500, tol=1e-9,
                                     popsize=20, mutation=(0.5, 1.5))
        r = evaluate(res.x[0], res.x[1], foam)
        if r and feasible(r):
            best = r

    return best


# ===================================================================
# %% REPORTING
# ===================================================================

def sep(c="-", w=72): print(c * w)

def run():
    sep("=")
    print("  SANDWICH BEAM WEIGHT OPTIMISER")
    sep("=")
    print(f"  Span:      L = {L*1000:.0f} mm")
    print(f"  Width:     B = {B*1000:.0f} mm")
    print("  Skin:      GFRP")
    print(f"  LC1:       Uniform pressure 5 t/m2  ->  "
          f"q = {q_LC1*1e-3:.3f} kN/m")
    print(f"  LC2:       Point load -> P_LC2 = {P_LC2*1e-3:.0f} kN")
    print(f"  delta_max: {DELTA_MAX*1000:.1f} mm  (L/{int(round(L/DELTA_MAX))})")
    print(f"  M_LC1 = {M1/1000:.3f} kN.m  |  V_LC1 = {V1/1000:.3f} kN")
    print(f"  M_LC2 = {M2/1000:.3f} kN.m  |  V_LC2 = {V2/1000:.3f} kN")
    print()

    sep()
    print(f"  {'Foam':<8} {'t_skin mm':>10} {'t_core mm':>10} "
          f"{'t_total mm':>11} {'Mass kg':>9}")
    sep()

    all_feasible = []
    best_design  = None
    t_total    = np.inf

    for foam in DIVINYCELL_H:
        result = optimise_for_foam(foam)
        if result:
            all_feasible.append(result)
            flag = ""
            if result['t_total'] < t_total:
                t_total   = result['t_total']
                best_design = result
                flag = "  <- best"
            print(f"  {foam.name:<8} {result['t_skin']*1e3:>10.2f} "
                  f"{result['t_core']*1e3:>10.1f} "
                  f"{result['t_total']*1e3:>11.1f} "
                  f"{result['mass']:>9.2f}{flag}")
        else:
            print(f"  {foam.name:<8} {'no feasible solution':>42}")

    print()
    sep("=")
    print("  OPTIMAL DESIGN — DETAIL")
    sep("=")

    if best_design is None:
        print("  No feasible design found.")
        print("  Try:  increasing b, relaxing DELTA_MAX,")
        print("        or widening T_S_MAX / T_C_MAX.")
        return

    r = best_design
    print(f"\n  Foam               : {r['foam']}")
    print(f"  Skin thickness     : {r['t_skin']*1e3:.2f} mm  per face")
    print(f"  Core thickness     : {r['t_core']*1e3:.1f} mm")
    print(f"  Total beam thickness  : {r['t_total']*1e3:.1f} mm")
    print(f"  Total beam mass    : {r['mass']:.2f} kg")

    print(f"\n  {'Constraint':<32} {'LC1':>9} {'LC2':>9} {'Limit':>10}  Status")
    sep()

    def ok(v): return "OK" if v >= -1e-9 else "FAIL"
    def ok2(a, b_): return f"{ok(a)} / {ok(b_)}"

    print(f"  {'Deflection (mm)':<32}"
          f"{r['delta_LC1']*1e3:>9.2f}"
          f"{r['delta_LC2']*1e3:>9.2f}"
          f"{r['delta_max']*1e3:>10.2f}"
          f"  {ok2(r['m_defl_LC1'], r['m_defl_LC2'])}")

    print(f"  {'Skin bending stress (MPa)':<32}"
          f"{r['sigma_LC1']*1e-6:>9.2f}"
          f"{r['sigma_LC2']*1e-6:>9.2f}"
          f"  {ok2(r['m_sigma_LC1'], r['m_sigma_LC2'])}")

    print(f"  {'Core shear stress (MPa)':<32}"
          f"{r['tau_LC1']*1e-6:>9.4f}"
          f"{r['tau_LC2']*1e-6:>9.4f}"
          f"  {ok2(r['m_tau_LC1'], r['m_tau_LC2'])}")

    print(f"  {'Allowable face wrinkling stress (MPa)':<32}"
          f"{r['sig_wrinkling']*1e-6:>9.4f}"
          f"{r['sig_wrinkling']*1e-6:>9.4f}"
          f"  {ok2(r['m_wrinkling_LC1'], r['m_wrinkling_LC2'])}")

    if len(all_feasible) > 1:
        print()
        sep()
        print("  All feasible designs ranked by mass:")
        sep()
        print(f"  {'#':<4} {'Foam':<8} {'t_skin':>8} {'t_core':>9} "
              f"{'t_total':>9} {'Mass':>9}  Governing constraint")
        sep()
        for i, d in enumerate(
                sorted(all_feasible, key=lambda x: x['mass']), 1):
            # Identify which constraint is most active (smallest margin)
            margins = {
                'deflection': min(d['m_defl_LC1'], d['m_defl_LC2'])*1e3,
                'skin stress': min(d['m_sigma_LC1'], d['m_sigma_LC2'])*1e3,
                'core shear': min(d['m_tau_LC1'], d['m_tau_LC2'])*1e3,
                'face wrinkling': min(d['m_wrinkling_LC1'], d['m_wrinkling_LC2'])*1e3,
            }
            active = min(margins, key=margins.get)
            tag = " <- optimal" if d is best_design else ""
            print(f"  {i:<4} {d['foam']:<8} "
                  f"{d['t_skin']*1e3:>7.2f}  "
                  f"{d['t_core']*1e3:>8.1f}  "
                  f"{d['t_total']*1e3:>8.1f}  "
                  f"{d['mass']:>8.2f}  {active}{tag}")

    print()
    sep("=")
    print("  Done.")
    sep("=")


if __name__ == "__main__":
    run()
    # res = evaluate(t_s=12e-3, t_c=60e-3, foam=DIVINYCELL_H[-1])
