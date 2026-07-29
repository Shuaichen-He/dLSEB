"""
Energy Balance Module
-----------------------
Computes net radiation, ground heat flux, and solves the surface energy
balance via optimistix root-finding for surface temperature. Soil heat
transfer uses a Crank-Nicolson scheme.  Scan functions drive time-series
simulation and spin-up.

Dependencies: model.turbulence (solve_mo_coupled), config, lineax, optimistix
"""

import jax
import jax.numpy as jnp
from jax import lax

import lineax as lx
import optimistix as optx

from config import SIGMA0, DT, SOIL_CENTER, DZ, DCZ
from model.turbulence import solve_mo_coupled

# =============================================================================
# Energy Balance Module
# =============================================================================
# All fluxes are defined positive UPWARD (atmospheric convention):
#   R_n = R_{su} + R_{lu} - R_{sd} - R_{ld}  (outgoing - incoming)
#   G = -k \cdot (T_{\mathrm{sfc}} - T_{z_1}) / \Delta z_1   (Fourier's law)
#   H = -\rho c_p (\theta_{\mathrm{atm}} - \theta_{\mathrm{sfc}}) / r_{ah}
#
# Surface energy balance (all upward-positive):
#   -R_n - H + G = 0
#
# Derivation: Energy into surface = Energy out of surface
#   (1-\alpha)R_{sd} + \sigma R_{ld} = \sigma\sigma_0 T^4 + H + (-G)
#   \Rightarrow (R_{sd}+R_{ld}) - (R_{su}+R_{lu}) = H - G
#   \Rightarrow -R_n - H + G = 0
#
# Note: The classic "Rn + H + G = 0" form is only valid when Rn is defined
# as positive DOWNWARD (R_{sd}+R_{ld}-R_{su}-R_{lu}), i.e. the land-surface
# convention. With our atmospheric convention (Rn positive upward), the
# correct form is -Rn - H + G = 0.


def Net_Rad(alpha, sigma, Rsd, Rld, T_sfc):
    r"""
    Calculate net radiation components (all fluxes positive upward).

    Computes surface net radiation as:
        # Math: R_{su} = \alpha \cdot R_{sd}
        # Math: R_{lu} = (1 - \varepsilon) \cdot R_{ld} + \varepsilon \cdot \sigma_0 \cdot T_{\mathrm{sfc}}^4
        # Math: R_n = R_{su} + R_{lu} - R_{sd} - R_{ld}

    Rn > 0: net upward radiation (surface loses radiative energy)
    Rn < 0: net downward radiation (surface gains radiative energy)

    Args:
        alpha: Surface shortwave albedo
        sigma: Surface emissivity
        Rsd: Incoming shortwave radiation [W m^-2]
        Rld: Incoming longwave radiation [W m^-2]
        T_sfc: Surface temperature [K]

    Returns:
        Rn: Net radiation [W m^-2] (positive upward, outgoing - incoming)
        # Math: R_n = R_{su} + R_{lu} - R_{sd} - R_{ld}  (positive upward)
        Rsu: Reflected shortwave radiation [W m^-2]
        Rlu: Outgoing longwave radiation [W m^-2]
    """
    
    Rsd = lax.cond(Rsd >= 0, lambda: Rsd, lambda: 0.0)
    Rsu = alpha * Rsd
    Rlu = (1 - sigma) * Rld + sigma * SIGMA0 * T_sfc**4
    Rn = Rsu - Rsd + (Rlu - Rld)
    # jax.debug.print("Rn={}, Rsu={}, Rlu={}, Rsd={}, Rld={}", Rn, Rsu, Rlu, Rsd, Rld)
    return Rn, Rsu, Rlu


def G_sfc(k, T_sfc, T_z1):
    r"""
    Calculate ground/surface heat flux using Fourier's law (positive upward).

    G = -k \cdot (T_{\mathrm{sfc}} - T_{z_1}) / \Delta z_1

    G > 0: heat flows from soil to surface (upward, soil cools)
    G < 0: heat flows from surface to soil (downward, soil warms)

    Args:
        k: Soil thermal conductivity [W m^-1 K^-1]
        T_sfc: Surface temperature [K]
        T_z1: Temperature at first soil layer center [K]

    Returns:
        G: Ground heat flux [W m^-2] (positive upward)
        # Math: G = -k \cdot (T_{\mathrm{sfc}} - T_{z_1})  / \Delta z_1
    """
    
    return -k * (T_sfc - T_z1) / SOIL_CENTER[0]


# =============================================================================
# Crank-Nicolson Scheme for Soil Heat Transfer
# =============================================================================
# Math: \frac{C}{\Delta t}(T^{n+1} - T^n) = \frac{1}{2}\left[ \frac{\partial}{\partial z}\!\left(k\frac{\partial T}{\partial z}\right)^{n+1} + \frac{\partial}{\partial z}\!\left(k\frac{\partial T}{\partial z}\right)^{n} \right]


# simplify form to aline with Richards Equation
def Soil_Crank_Nicolson_diff(k, C, T0, G_sfc):
    A = jnp.exp(C) / DT
    B = jnp.append(0, k / (2 * DZ[1:] * DCZ))
    D = jnp.append(k / (2 * DZ[:-1] * DCZ), 0)
    diag_mid = A + B + D
    diag_up = -D
    diag_low = -B
    tri_b = jnp.diag(A - B - D) + jnp.diag(D[:-1], 1) + jnp.diag(B[1:], -1)
    b = tri_b @ T0 - jnp.append(G_sfc / DZ[0], jnp.zeros((len(T0) - 1)))

    operator = lx.TridiagonalLinearOperator(diag_mid, diag_low[1:], diag_up[:-1])
    solution = lx.linear_solve(operator, b, solver=lx.Tridiagonal())
    return solution.value


# redirection the EB residual to the soil_Crank_Nicolson_diff
# def Soil_Crank_Nicolson_diff(k, C, T0, G_sfc):
#     A = 2 * jnp.exp(C) * DZ / DT
#     B = jnp.append(0, k / DCZ)
#     C = jnp.append(-k / DCZ, 0)
#     diag_mid = A + B - C
#     diag_up = C
#     diag_low = -B
#     tri_b = jnp.diag(A - B + C) + jnp.diag(-C[:-1], 1) + jnp.diag(B[1:], -1)
#     b = tri_b @ T0 - jnp.append(2 * G_sfc, jnp.zeros((len(T0) - 1)))
#     operator = lx.TridiagonalLinearOperator(diag_mid, diag_low[1:], diag_up[:-1])
#     solution = lx.linear_solve(operator, b, solver=lx.Tridiagonal())
#     return solution.value


# Crank_Nicolson_original
# def Soil_Crank_Nicolson_diff(k, C, T0, G_sfc):
#     C = jnp.exp(C)
#     D_up = jnp.array([k * DT / (2 * C * DZ[0] * DCZ[0])])
#     D_mid = k * DT / (2 * C * DZ[1:-1] * DCZ[:-1] * DCZ[1:])
#     D_low = jnp.array([k * DT / (2 * C * DZ[-1] * DCZ[-1])])

#     diag_low = jnp.concatenate([jnp.zeros(1), -D_mid * DCZ[1:], -D_low])
#     diag_mid = jnp.concatenate([1 + D_up, 1 + D_mid * (DCZ[1:] + DCZ[:-1]), 1 + D_low])
#     diag_up = jnp.concatenate([-D_up, -D_mid * DCZ[:-1], jnp.zeros(1)])

#     # A = jnp.diag(diag_up, 1) + jnp.diag(diag_mid) + jnp.diag(diag_low, -1)

#     b_up = (1 - D_up) * T0[0] + D_up * T0[1] - D_up * DCZ[0] * (2 * G_sfc) / k
#     b_mid = (
#         T0[:-2] * D_mid * DCZ[1:]
#         + T0[1:-1] * (1 - D_mid * (DCZ[1:] + DCZ[:-1]))
#         + T0[2:] * D_mid * DCZ[:-1]
#     )
#     b_low = D_low * T0[-2] + (1 - D_low) * T0[-1]
#     b = jnp.concatenate([b_up, b_mid, b_low])
#     operator = lx.TridiagonalLinearOperator(diag_mid, diag_low[1:], diag_up[:-1])
#     solution = lx.linear_solve(operator, b, solver=lx.Tridiagonal())
#     return solution.value


# =============================================================================
# Surface Energy Balance: optimistix Root-Finding for Surface Temperature
# =============================================================================


def _eb_residual(T_sfc, args):
    r"""
    Energy balance residual for root-finding.

    At the solution T_sfc*, we have F_sfc(T_sfc*) = 0:
        -R_n(T_sfc) - H(T_sfc) + G_sfc(T_sfc) = 0

    This corresponds to the surface energy balance in the all-upward-positive
    convention (see module header for derivation):
        # Math: -R_n^\uparrow - H^\uparrow + G^\uparrow = 0

    Equivalent to the downward-Rn form:
        # Math: R_n^\downarrow = H^\uparrow - G^\uparrow  (i.e., R_n^\downarrow = H^\uparrow + G^\downarrow)

    Args:
        T_sfc: Surface temperature guess [K] (scalar, the variable to solve for)
        args: Tuple of (alpha, sigma, z0m, k, Rsd, Rld, T_atm, q_atm, u, rho, P, T_z1)

    Returns:
        F_sfc: Energy balance residual [W m^-2]
    """
    # Math: F_{\mathrm{sfc}}(T_{\mathrm{sfc}}) = -R_n(T_{\mathrm{sfc}}) - H(T_{\mathrm{sfc}}) + G(T_{\mathrm{sfc}}) = 0
    alpha, sigma, z0m, k, Rsd, Rld, T_atm, q_atm, u, rho, P, T_z1 = args
    q_sfc = 0.0  # Assume dry surface (q_sfc = 0)
    _, _, _, _, H, _ = solve_mo_coupled(z0m, T_atm, T_sfc, q_atm, q_sfc, u, rho, P)
    net_rad, _, _ = Net_Rad(alpha, sigma, Rsd, Rld, T_sfc)
    F_sfc = -net_rad - H + G_sfc(k, T_sfc, T_z1)
    # jax.debug.print(
    #     "nrad={}, H={}, G_sfc={}, F_sfc={}", net_rad, H, G_sfc(k, T_sfc, T_z1), F_sfc
    # )
    return F_sfc


@jax.jit
def solving_EB_surface_T(alpha, sigma, z0m, k, Rsd, Rld, T_atm, q_atm, u, rho, P, T_z1):
    """
    Solve for surface temperature using optimistix root-finding (Newton's method).

    Finds T_sfc such that the surface energy balance residual is zero:
        -R_n(T_sfc) - H(T_sfc) + G_sfc(T_sfc) = 0

    (All fluxes positive upward; see module header for derivation.)

    Replaces the previous hand-written lax.while_loop Newton iteration with
    optimistix.root_find, which provides built-in differentiable adjoints
    (ImplicitAdjoint / RecursiveCheckpointAdjoint) — no need for @custom_jvp.

    Args:
        alpha: Surface shortwave albedo
        sigma: Surface emissivity
        z0m: Roughness length for momentum [m]
        k: Soil thermal conductivity [W m^-1 K^-1]
        Rsd: Incoming shortwave radiation [W m^-2]
        Rld: Incoming longwave radiation [W m^-2]
        T_atm: Air temperature [K]
        q_atm: Atmospheric specific humidity [kg kg^-1]
        u: Wind speed [m s^-1]
        rho: Air density [kg m^-3]
        P: Air pressure [hPa]
        T_z1: Temperature at first soil layer [K]

    Returns:
        T_sfc: Converged surface temperature [K]
    """
    # Math: T_{\mathrm{sfc}}^* = \arg\min_{T} |F_{\mathrm{sfc}}(T)|,\quad F_{\mathrm{sfc}} = -R_n - H + G
    args = (alpha, sigma, z0m, k, Rsd, Rld, T_atm, q_atm, u, rho, P, T_z1)
    # T_sfc_init = 300.0  # Initial guess [K]

    # Newton solver: recomputes Jacobian at each step via JAX autodiff
    solver = optx.Newton(
        rtol=1e-2,
        atol=1e-4,
    )

    sol = optx.root_find(
        _eb_residual,
        solver,
        y0=T_z1,  # Initial guess [K] use first layer
        args=args,
        max_steps=15,
        adjoint=optx.ImplicitAdjoint(
            linear_solver=lx.AutoLinearSolver(well_posed=False)
        ),
        # adjoint=optx.RecursiveCheckpointAdjoint(),
        throw=False,
    )

    return sol.value


# NOTE: The hand-written @custom_jvp and JVP function (solving_EB_surface_T_jvp)
# have been removed. optimistix handles differentiation via its adjoint mechanism
# (ImplicitAdjoint or RecursiveCheckpoint Adjoint). JAX autodiff now differentiates
# through the optimistix solver directly.


# =============================================================================
# Energy Balance Scanning and Training
# =============================================================================


def spin_up_soil_T(alpha, sigma, z0m, k, C, meteo_input, Train_T, spin_up_iter):
    """
    Perform spin-up to establish initial soil temperature profile.

    Runs the model forward for spin_up_iter iterations to reach
    equilibrium soil temperature conditions.

    Args:
        alpha: Surface shortwave albedo
        sigma: Surface emissivity
        z0m: Roughness length for momentum [m]
        k: Soil thermal conductivity [W m^-1 K^-1]
        C: Soil volumetric heat capacity [J m^-3 K^-1]
        meteo_input: Meteorological forcing array
        Train_T: Training soil temperature observations [K]
        spin_up_iter: Number of spin-up iterations

    Returns:
        spin_up_T0: Initial soil temperature profile after spin-up [K]
    """
    spin_up_T0 = jnp.hstack([Train_T[0, :], jnp.full(6, 300.0)])
    for _ in range(spin_up_iter):
        _, _, _, _, _, T0 = scan_training_periods(
            alpha, sigma, z0m, k, C, meteo_input, spin_up_T0
        )
        spin_up_T0 = T0[-1]
    return spin_up_T0


def EB_bound_G_count_T_soil(carry, meteo_input):
    """
    Single time step energy balance calculation (for JAX scan).

    Computes:
    - Surface temperature from energy balance
    - Radiative fluxes
    - Turbulent fluxes
    - Ground heat flux
    - Soil temperature update

    Args:
        carry: State vector (alpha, sigma, z0m, k, C, T_0)
        meteo_input: Meteorological forcing for one time step [Rsd, Rld, T_atm, q_atm, u, rho, P]

    Returns:
        Updated carry and energy balance outputs
    """
    alpha, sigma, z0m, k, C, T_0 = carry
    Rsd, Rld, T_atm, q_atm, u, rho, P = meteo_input
    q_sfc = 0.0  # Assume dry surface (q_sfc = 0)

    # Solve surface temperature from energy balance
    T_sfc = solving_EB_surface_T(
        alpha, sigma, z0m, k, Rsd, Rld, T_atm, q_atm, u, rho, P, T_0[0]
    )

    # Calculate radiative fluxes
    _, Rsu, Rlu = Net_Rad(alpha, sigma, Rsd, Rld, T_sfc)

    # Calculate turbulent fluxes via solve_mo_coupled (LE computed but unused)
    _, _, _, _, H, _ = solve_mo_coupled(z0m, T_atm, T_sfc, q_atm, q_sfc, u, rho, P)

    # Calculate ground heat flux
    G = G_sfc(k, T_sfc, T_0[0])
    # jax.debug.print("G={}", G)

    # Before soil solve, ensure T_0 is finite
    # T_0 = jnp.where(jnp.isnan(T_0), 300.0, T_0)
    # Update soil temperature using Crank-Nicolson scheme
    new_T0 = Soil_Crank_Nicolson_diff(k, C, T_0, G)

    return (alpha, sigma, z0m, k, C, new_T0), (Rsu, Rlu, H, G, T_sfc, new_T0)


def scan_training_periods(alpha, sigma, z0m, k, C, meteo_input, spin_up_T0):
    """
    Scan through all training time periods using JAX scan.

    Args:
        alpha: Surface shortwave albedo
        sigma: Surface emissivity
        z0m: Roughness length for momentum [m]
        k: Soil thermal conductivity [W m^-1 K^-1]
        C: Soil volumetric heat capacity [J m^-3 K^-1]
        meteo_input: Meteorological forcing array
        spin_up_T0: Initial soil temperature profile [K]

    Returns:
        EB_out: Energy balance outputs for all time steps
    """
    init = (alpha, sigma, z0m, k, C, spin_up_T0)
    xs = meteo_input.T
    _, EB_out = lax.scan(EB_bound_G_count_T_soil, init, xs)
    return EB_out
