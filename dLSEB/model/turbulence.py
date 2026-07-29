"""
Turbulence Module

Surface layer similarity theory for turbulent flux calculation.

This module provides:
- Stability correction functions (psi_m, psi_h)
- Roughness length calculation for heat (zh)
- Friction velocity calculation (U_star)
- Temperature scale calculation (Theta_star)
- Transport resistance terms (dlg_m, dlg_h)
- Bulk Richardson number and initial guess functions
- Obukhov length calculation
- Main solver for turbulent fluxes (solve_turbulence)

Dependencies:
- Requires jax, jax.numpy, lineax, optimistix
"""

import jax
import jax.numpy as jnp
from jax import lax

import lineax as lx
import optimistix as optx

import config
from config import D, KAM, G, CP, LAM_V


# =============================================================================
# Turbulence Module: Surface Layer Similarity Theory
# =============================================================================
# Math: u_* = \frac{\kappa \cdot V_a}{\ln[(z-d)/z_{0m}] - \psi_m(\zeta) + \psi_m(z_{0m}/L)}
# Math: \theta_* = \frac{\kappa \cdot (\theta_{\mathrm{atm}} - \theta_s)}{\ln[(z-d)/z_{0h}] - \psi_h(\zeta) + \psi_h(z_{0h}/L)}
# Math: L = -\frac{u_*^3 \bar{T}_v}{\kappa g \theta_{v*}} \quad (\text{Obukhov length})
def psi_m(zeta):
    """
    Integrated stability correction function for momentum (Paulson 1970).

    Args:
        zeta: Stability parameter (z/L)

    Returns:
        psi_m: Integrated stability correction for momentum
    """
    # Math: \psi_m(\zeta) = 2\ln\!\left(\frac{1+x}{2}\right) + \ln\!\left(\frac{1+x^2}{2}\right) - 2\arctan(x) + \frac{\pi}{2},\; x = (1 - 16\zeta)^{1/4}
    x = (1 - 16 * zeta) ** 0.25
    return (
        2 * jnp.log((1 + x) / 2)
        + jnp.log((1 + x**2) / 2)
        - 2 * jnp.arctan(x)
        + jnp.pi / 2
    )


def psi_h(zeta):
    """
    Integrated stability correction function for heat (Paulson 1970).

    Args:
        zeta: Stability parameter (z/L)

    Returns:
        psi_h: Integrated stability correction for heat
    """
    # Math: \psi_h(\zeta) = 2\ln\!\left(\frac{1+x^2}{2}\right), \quad x = (1 - 16\zeta)^{1/4}
    x = (1 - 16 * zeta) ** 0.25
    return 2 * jnp.log((1 + x**2) / 2)


def zh(z0m, u_star):
    """
    Calculate roughness length for heat using Chirtwood-Zilitinkevich formula.

    Args:
        z0m: Roughness length for momentum [m]
        u_star: Friction velocity [m s^-1]

    Returns:
        z0h: Roughness length for heat [m]
    """
    # Math: z_{0h} = z_{0m} \cdot \exp\!\left(-0.13 \cdot (u_* \cdot z_{0m} / \nu)^{0.45}\right),\quad \nu = 1.5\times 10^{-5}~\mathrm{m^2/s}
    return z0m * jnp.exp(-0.13 * (u_star * z0m / 1.5e-5) ** 0.45)


def T_2_theta(T, p, p0=1e3):
    """
    Convert temperature to potential temperature.

    θ = T · (p₀ / p)^κ,  κ = R_d / c_p ≈ 0.2857

    Args:
        T:  Air temperature [K]
        p:  Air pressure [Pa]
        p0: Reference pressure [hPa], default 1000 hPa

    Returns:
        theta: Potential temperature [K]
    """
    # Math: \theta = T \cdot (p_0 / p)^{\kappa}, \quad \kappa = R_d / c_p \approx 0.2857
    Rd = 287.05  # J kg⁻¹ K⁻¹
    cp = 1004.0  # J kg⁻¹ K⁻¹
    kappa = Rd / cp  # ≈ 0.2854
    return T * (p0 / p) ** kappa


def theta_2_theta_v(theta, q):
    """Convert (potential) temperature to virtual (potential) temperature: θ_v = θ · (1 + 0.61·q)."""
    return theta * (1 + 0.61 * q)


def delta_theta_v(theta_atm, theta_sfc, q_atm, q_s):
    """
    Virtual potential temperature difference between atmosphere and surface.

    Δθ_v = (θ_atm - θ_sfc)·(1 + 0.61·q_atm) + 0.61·½(θ_atm + θ_sfc)·(q_atm - q_s)

    Returns:
        d_theta_v: Virtual potential temperature difference [K]
    """
    d_theta_v = (theta_atm - theta_sfc) * (1 + 0.61 * q_atm) + 0.61 * 0.5 * (
        theta_atm + theta_sfc
    ) * (q_atm - q_s)
    return d_theta_v


def init_wind(u, delta_theta_v):
    """
    Initialize wind speed for turbulent flux calculation.
    For unstable conditions, adds convective velocity component.

    Args:
        u: Wind speed [m s^-1]
        delta_theta_v: Virtual temperature difference [K]

    Returns:
        Va: Initialized wind speed [m s^-1]
    """

    def stable_branch(_):
        return u

    def unstable_branch(_):
        return jnp.maximum(jnp.sqrt(u**2 + 0.25), 1.0)

    return lax.cond(delta_theta_v >= 0, stable_branch, unstable_branch, None)


def Va_convective(u, L, theta_v_star, theta_v_atm, zi=1e3):
    """
    Calculate effective velocity scale with convective gustiness.

    The constraint Va ≥ 1 is required for numerical reasons to prevent
    H and LE from becoming spuriously small with weak wind speeds.
    The convective velocity Uc accounts for the contribution of
    large eddies in the convective boundary layer to surface fluxes.

    Args:
        u: Wind speed at reference height [m s⁻¹]
        L: Obukhov length [m]
        theta_v_star: Virtual temperature scale [K]
        theta_v_atm: Atmospheric virtual temperature [K]
        zi: Convective boundary layer height [m], default 1000

    Returns:
        Va: Effective velocity scale [m s⁻¹]
    """
    # Math: V_a = \max\!\left(\sqrt{u^2 + U_c^2},\; 1\right), \quad U_c = \left(\frac{-g u \theta_{v*} z_i}{\theta_{v,\mathrm{atm}}}\right)^{1/3}
    zeta = lax.cond(L != 0.0, lambda _: (config.Z_ATM - D) / L, lambda _: 0.0, None)
    Uc = ((-G * u * theta_v_star * zi) / theta_v_atm) ** (1 / 3)
    Va_unstable = jnp.maximum(jnp.sqrt(u**2 + Uc**2), 1.0)
    Va_stable = jnp.maximum(u, 1.0)

    def unstable_branch(_):
        return Va_unstable

    def stable_branch(_):
        return Va_stable

    return lax.cond(zeta < 0, unstable_branch, stable_branch, None)


def U_star(z0m, u, L):
    """
    Calculate friction velocity using surface layer similarity.

    Handles different stability regimes:
    - Very unstable (zeta < -1.574)
    - Unstable (-1.574 <= zeta < 0)
    - Stable (0 <= zeta <= 1)
    - Very stable (zeta > 1)

    Args:
        z0m: Roughness length for momentum [m]
        u: Wind speed at reference height [m s^-1]
        L: Obukhov length [m]

    Returns:
        u_star: Friction velocity [m s^-1]
    """
    zeta = lax.cond(L != 0.0, lambda _: (config.Z_ATM - D) / L, lambda _: 0.0, None)

    # -------------------------------------------------------------------------
    # Stability regime selection using lax.cond
    # -------------------------------------------------------------------------
    def very_unstable_branch(_):
        return (
            u
            * KAM
            / (
                jnp.log(-1.574 * L / z0m)
                - psi_m(-1.574)
                + 1.14 * ((-zeta) ** (1 / 3) - (1.574) ** (1 / 3))
                + psi_m(z0m / L)
            )
        )

    def unstable_branch(_):
        return u * KAM / (jnp.log((config.Z_ATM - D) / z0m) - psi_m(zeta) + psi_m(z0m / L))

    def stable_branch(_):
        return u * KAM / (jnp.log((config.Z_ATM - D) / z0m) + 5 * zeta - 5 * z0m / L)

    def very_stable_branch(_):
        return (
            u
            * KAM
            / (jnp.log(L / z0m) + 5 + 5 * jnp.log(zeta) + zeta - 1 - 5 * z0m / L)
        )

    # Nested lax.cond for stability regime selection
    def unstable_regime(_):
        return lax.cond(zeta < -1.574, very_unstable_branch, unstable_branch, None)

    def stable_regime(_):
        return lax.cond(zeta <= 1, stable_branch, very_stable_branch, None)

    return lax.cond(zeta < 0, unstable_regime, stable_regime, None)


def Theta_star(z0h, theta_atm, theta_sfc, L):
    """
    Calculate temperature/humidity scale using surface layer similarity.

    Handles different stability regimes similar to U_star.
    The same function is reused for the specific-humidity scale q_star
    by passing (q_atm, q_sfc) in place of (theta_atm, theta_sfc), since
    the stability functions for heat and moisture are identical.

    Args:
        z0h: Roughness length for heat [m]
        theta_atm: Atmospheric value at reference height (e.g., potential temperature [K] or specific humidity [kg/kg])
        theta_sfc: Surface value (e.g., surface potential temperature [K] or specific humidity [kg/kg])
        L: Obukhov length [m]

    Returns:
        Scale parameter (theta_star or q_star)
    """
    zeta = lax.cond(L != 0.0, lambda _: (config.Z_ATM - D) / L, lambda _: 0.0, None)

    # -------------------------------------------------------------------------
    # Stability regime selection using lax.cond
    # -------------------------------------------------------------------------
    def very_unstable_branch(_):
        return (
            (theta_atm - theta_sfc)
            * KAM
            / (
                jnp.log(-0.465 * L / z0h)
                - psi_h(-0.465)
                + 0.8 * (((0.465) ** (-1 / 3)) - (-zeta) ** (-1 / 3))
                + psi_h(z0h / L)
            )
        )

    def unstable_branch(_):
        return (
            (theta_atm - theta_sfc)
            * KAM
            / (jnp.log((config.Z_ATM - D) / z0h) - psi_h(zeta) + psi_h(z0h / L))
        )

    def stable_branch(_):
        return (
            (theta_atm - theta_sfc)
            * KAM
            / (jnp.log((config.Z_ATM - D) / z0h) + 5 * zeta - 5 * z0h / L)
        )

    def very_stable_branch(_):
        return (
            (theta_atm - theta_sfc)
            * KAM
            / (jnp.log(L / z0h) + 5 + 5 * jnp.log(zeta) + zeta - 1 - 5 * z0h / L)
        )

    def unstable_regime(_):
        return lax.cond(zeta < -0.465, very_unstable_branch, unstable_branch, None)

    def stable_regime(_):
        return lax.cond(zeta <= 1, stable_branch, very_stable_branch, None)

    return lax.cond(zeta < 0, unstable_regime, stable_regime, None)


def Theta_v_star(theta_star, theta_atm, theta_sfc, q_atm, q_star):
    return (
        theta_star * (1 + 0.61 * q_atm) + 0.61 * 0.5 * (theta_atm + theta_sfc) * q_star
    )


def dlg_m(z0m, L):
    """
    Calculate momentum transport resistance term.

    Integrates stability correction over the surface layer.

    Args:
        z0m: Roughness length for momentum [m]
        L: Obukhov length [m]

    Returns:
        dlg_m: Logarithmic resistance term for momentum
    """
    zeta = lax.cond(L != 0.0, lambda _: (config.Z_ATM - D) / L, lambda _: 0.0, None)

    # -------------------------------------------------------------------------
    # Stability regime selection using lax.cond
    # -------------------------------------------------------------------------
    def very_unstable_branch(_):
        return (
            jnp.log(-1.574 * L / z0m)
            - psi_m(-1.574)
            + 1.14 * ((-zeta) ** (1 / 3) - (1.574) ** (1 / 3))
            + psi_m(z0m / L)
        )

    def unstable_branch(_):
        return jnp.log((config.Z_ATM - D) / z0m) - psi_m(zeta) + psi_m(z0m / L)

    def stable_branch(_):
        return jnp.log((config.Z_ATM - D) / z0m) + 5 * zeta - 5 * z0m / L

    def very_stable_branch(_):
        return jnp.log(L / z0m) + 5 + 5 * jnp.log(zeta) + zeta - 1 - 5 * z0m / L

    def unstable_regime(_):
        return lax.cond(zeta < -1.574, very_unstable_branch, unstable_branch, None)

    def stable_regime(_):
        return lax.cond(zeta <= 1, stable_branch, very_stable_branch, None)

    return lax.cond(zeta < 0, unstable_regime, stable_regime, None)


def dlg_h(z0h, L):
    """
    Calculate heat transport resistance term.

    Integrates stability correction over the surface layer for heat.

    Args:
        z0h: Roughness length for heat [m]
        L: Obukhov length [m]

    Returns:
        dlg_h: Logarithmic resistance term for heat
    """
    zeta = lax.cond(L != 0.0, lambda _: (config.Z_ATM - D) / L, lambda _: 0.0, None)

    # -------------------------------------------------------------------------
    # Stability regime selection using lax.cond
    # -------------------------------------------------------------------------
    def very_unstable_branch(_):
        return (
            jnp.log(-0.465 * L / z0h)
            - psi_h(-0.465)
            + 0.8 * ((0.465) ** (-1 / 3) - (-zeta) ** (-1 / 3))
            + psi_h(z0h / L)
        )

    def unstable_branch(_):
        return jnp.log((config.Z_ATM - D) / z0h) - psi_h(zeta) + psi_h(z0h / L)

    def stable_branch(_):
        return jnp.log((config.Z_ATM - D) / z0h) + 5 * zeta - 5 * z0h / L

    def very_stable_branch(_):
        return jnp.log(L / z0h) + 5 + 5 * jnp.log(zeta) + zeta - 1 - 5 * z0h / L

    def unstable_regime(_):
        return lax.cond(zeta < -0.465, very_unstable_branch, unstable_branch, None)

    def stable_regime(_):
        return lax.cond(zeta <= 1, stable_branch, very_stable_branch, None)

    return lax.cond(zeta < 0, unstable_regime, stable_regime, None)


def Rib_for_first_guess(theta_v_atm, theta_v_sfc, Va):
    """
    Calculate Bulk Richardson number for initial guess of stability.

    Args:
        theta_v_atm: Atmospheric virtual temperature [K]
        theta_v_sfc: Surface virtual temperature [K]
        Va: Effective wind speed [m s⁻¹]

    Returns:
        Rib: Bulk Richardson number
    """
    # Math: Ri_b = \frac{g (z-d) (\theta_{v,\mathrm{atm}} - \theta_{v,\mathrm{sfc}})}{0.5(\theta_{v,\mathrm{atm}} + \theta_{v,\mathrm{sfc}}) V_a^2}
    return ((theta_v_atm - theta_v_sfc) * G * (config.Z_ATM - D)) / (
        0.5 * (theta_v_atm + theta_v_sfc) * Va**2
    )


def zeta_for_first_guess(z0m, rib):
    """
    Calculate initial guess for stability parameter zeta from Richardson number.

    Args:
        z0m: Roughness length for momentum [m]
        rib: Bulk Richardson number

    Returns:
        zeta: Initial guess for stability parameter
    """

    # Use lax.cond for vectorized conditional
    def unstable_branch(_):
        return rib * jnp.log((config.Z_ATM - D) / z0m)

    def stable_branch(_):
        rib_clipped = jnp.minimum(rib, 0.19)
        return rib * jnp.log((config.Z_ATM - D) / z0m) / (1 - 5 * rib_clipped)

    return lax.cond(rib < 0, unstable_branch, stable_branch, None)


def L_renew(u_star, theta_v_star, theta_v_atm, theta_v_sfc):
    """
    Calculate Obukhov length from friction velocity and temperature scale.

    Uses the virtual-temperature form:
        L = (z-d) / [ (z-d) κ g θ_{v*} / (½(T_v,atm+T_v,sfc) u_*²) ]

    Neutral stability fallback: L → 1e4 m when the formula diverges.

    Args:
        u_star:        Friction velocity [m s⁻¹]
        theta_v_star:  Virtual-temperature scale [K]
        theta_v_atm:   Atmospheric virtual temperature [K]
        theta_v_sfc:   Surface virtual temperature [K]

    Returns:
        L: Obukhov length [m]
    """

    L = (config.Z_ATM - D) / (
        ((config.Z_ATM - D) * KAM * G * theta_v_star)
        / (0.5 * (theta_v_atm + theta_v_sfc) * u_star**2)
    )
    return lax.cond(jnp.isfinite(L), lambda _: L, lambda _: 1e4, None)


# =============================================================================
# Initial Guess for the Coupled State
# =============================================================================


def _init_state(z0m, theta_atm, theta_sfc, q_atm, q_sfc, Va):
    """
    Derive a physically grounded initial guess (u_*⁰, θ_{v*}⁰) from Rib.

    Steps
    -----
    1.  Rib  →  ζ₀  →  L₀
    2.  L₀   →  u_*⁰  via U_star (single call, no iteration)
    3.  L₀   →  θ_*⁰  via Theta_star,  q_*⁰  via Theta_star
    4.  θ_{v*}⁰ = Theta_v_star(θ_*⁰, ...)
    """
    theta_v_atm = theta_2_theta_v(theta_atm, q_atm)
    theta_v_sfc = theta_2_theta_v(theta_sfc, q_sfc)

    # --- 1. Rib → L₀ ---
    Rib = Rib_for_first_guess(theta_v_atm, theta_v_sfc, Va)
    zeta0 = zeta_for_first_guess(z0m, Rib)
    # Guard against zeta0 == 0  (pure neutral)
    L0 = lax.cond(
        jnp.abs(zeta0) > 1e-8,
        lambda _: jnp.clip((config.Z_ATM - D) / zeta0, -1e4, 1e4),
        lambda _: 1e4,
        None,
    )

    # --- 2. u_*⁰ ---
    u_star0 = U_star(z0m, Va, L0)

    # --- 3. θ_*⁰,  q_*⁰ ---
    z0h0 = zh(z0m, u_star0)
    theta_star0 = Theta_star(z0h0, theta_atm, theta_sfc, L0)
    q_star0 = Theta_star(z0h0, q_atm, q_sfc, L0)

    # --- 4. θ_{v*}⁰ ---
    theta_v_star0 = Theta_v_star(theta_star0, theta_atm, theta_sfc, q_atm, q_star0)
    # theta_vstar0 = jnp.clip(theta_vstar0, -5.0, 5.0)

    return u_star0, theta_v_star0


# =============================================================================
# Obukhov Length from the Coupled State
# =============================================================================
def L_from_state(u_star, theta_v_star, theta_v_atm, theta_v_sfc):
    """
    Derive Obukhov length from the coupled state (u_*, θ_{v*}).

    L = u_*² · ½(T_v,atm + T_v,sfc) / (κ g θ_{v*})

    A finite fallback of 1e4 m is applied when the formula diverges
    (θ_{v*} → 0, neutral limit).

    Args:
        u_star:        Friction velocity [m s⁻¹]
        theta_v_star:  Virtual-temperature scale [K]
        theta_v_atm:   Atmospheric virtual temperature [K]
        theta_v_sfc:   Surface virtual temperature [K]

    Returns:
        L: Obukhov length [m]
    """
    # Math: L = \frac{u_*^2 \cdot 0.5(\theta_{v,\mathrm{atm}} + \theta_{v,\mathrm{sfc}})}{\kappa g \theta_{v*}}
    L_raw = ((u_star**2) * 0.5 * (theta_v_atm + theta_v_sfc)) / (KAM * G * theta_v_star)
    return lax.cond(jnp.isfinite(L_raw), lambda _: L_raw, lambda _: 1e4, None)


# =============================================================================
# Coupled Fixed-Point Map  F : (u_*, θ_{v*}) → (u_*_new, θ_{v*_new})
# =============================================================================


def _make_fp_map(z0m, theta_atm, theta_sfc, q_atm, q_sfc, u):
    """
    Factory that closes over all *static* inputs and returns the fixed-point
    map consumed by `optimistix.fixed_point`.

    The returned function has the signature required by optimistix:

        f(state, args) -> state_new

    where `state = (u_star, theta_vstar)` is the PyTree being iterated and
    `args` is unused (all inputs are captured by closure).

    Iteration equations
    -------------------
    Given current (u_*, θ_{v*}):

        L       = -u_*³ T̄_v / (κ g θ_{v*})
        ζ       = (z - d) / L
        z₀h     = zh(z₀m, u_*)
        Va      = Va_convective(u_ref, L, θ_{v*}, θ_v,atm)   [convective gust]

        u_*_new     = U_star(z₀m, Va, L)
        θ_*_new     = Theta_star(z₀h, θ_atm, θ_s, L)
        q_*_new     = Theta_star(z₀h, q_atm, q_s, L)
        θ_{v*_new}  = Theta_v_star(θ_*_new, θ_atm, θ_s, q_atm, q_*_new)
    """
    theta_v_atm = theta_2_theta_v(theta_atm, q_atm)
    theta_v_sfc = theta_2_theta_v(theta_sfc, q_sfc)

    def fp_map(state, args):
        u_star, theta_v_star = state

        # --- Derived L ---
        L = L_from_state(u_star, theta_v_star, theta_v_atm, theta_v_sfc)

        # --- Roughness length for heat (depends on u_*) ---
        z0h = zh(z0m, u_star)

        # --- Convective wind speed (adds gustiness in free convection) ---
        Va = Va_convective(u, L, theta_v_star, theta_v_atm)

        # --- New u_* ---
        u_star_new = U_star(z0m, Va, L)

        # --- New θ_* and q_* ---
        theta_star_new = Theta_star(z0h, theta_atm, theta_sfc, L)
        q_star_new = Theta_star(z0h, q_atm, q_sfc, L)

        # --- New θ_{v*} ---
        theta_vstar_new = Theta_v_star(
            theta_star_new, theta_atm, theta_sfc, q_atm, q_star_new
        )
        # theta_vstar_new = jnp.clip(theta_vstar_new, -5.0, 5.0)

        return (u_star_new, theta_vstar_new)

    return fp_map


@jax.jit
def solve_mo_coupled(z0m, T_atm, T_sfc, q_atm, q_sfc, u, rho, P):
    """
    Solve Monin-Obukhov similarity theory via joint (u_*, θ_{v*}) iteration.

    The converged state is obtained by finding the fixed point of the map
    F(u_*, θ_{v*}) described in `_make_fp_map`.  Gradients with respect to
    any input flow through `optimistix.ImplicitAdjoint` (implicit function
    theorem), giving exact O(1) backward passes suitable for DA / PINNs.

    Args:
        z0m:   Roughness length for momentum [m]
        T_atm: Air temperature at z_atm [K]
        T_sfc: Surface (skin) temperature [K]
        q_atm: Specific humidity at z_atm [kg kg⁻¹]
        q_sfc: Surface specific humidity [kg kg⁻¹]
        u:     Horizontal wind speed at z_atm [m s⁻¹]
        rho:   Air density [kg m⁻³]
        P:     Air pressure [hPa]

    Returns:
        rah:          Aerodynamic resistance for heat [s m⁻¹]
        u_star:       Friction velocity [m s⁻¹]
        theta_v_star: Virtual-temperature scale [K]
        q_star:       Specific-humidity scale [kg kg⁻¹]
        H:            Sensible heat flux [W m⁻²]
        LE:           Latent heat flux [W m⁻²] (λ = 2.5e6 J kg⁻¹)
        L:            Obukhov length [m]
    """
    # ------------------------------------------------------------------
    # Math: H = -\rho c_p (\theta_{\mathrm{atm}} - \theta_{\mathrm{sfc}}) / r_{ah}
    # Math: LE = -\rho \lambda_v (q_{\mathrm{atm}} - q_{\mathrm{sfc}}) / r_{aw}
    # Math: r_{ah} = d\lg_m \cdot d\lg_h / (\kappa^2 V_a)
    # ------------------------------------------------------------------
    # 1.  Derived potential and virtual temperatures
    # ------------------------------------------------------------------
    theta_atm = T_2_theta(T_atm, P)
    theta_sfc = T_2_theta(T_sfc, P)
    theta_v_atm = theta_2_theta_v(theta_atm, q_atm)
    theta_v_sfc = theta_2_theta_v(theta_sfc, q_sfc)
    # ------------------------------------------------------------------
    # 2.  Wind-speed initialisation (add convective velocity for unstable)
    # ------------------------------------------------------------------
    d_theta_v = delta_theta_v(theta_atm, theta_sfc, q_atm, q_sfc)
    Va = init_wind(u, d_theta_v)

    # ------------------------------------------------------------------
    # 3.  Initial guess for (u_*, θ_{v*})
    # ------------------------------------------------------------------
    u_star0, theta_vstar0 = _init_state(z0m, theta_atm, theta_sfc, q_atm, q_sfc, Va)
    state0 = (u_star0, theta_vstar0)

    # ------------------------------------------------------------------
    # 4.  Fixed-point iteration
    # ------------------------------------------------------------------
    fp_map = _make_fp_map(z0m, theta_atm, theta_sfc, q_atm, q_sfc, u)

    solver = optx.FixedPointIteration(rtol=1e-1, atol=1e-2)

    sol = optx.fixed_point(
        fp_map,
        solver,
        state0,
        args=None,
        max_steps=5,
        adjoint=optx.ImplicitAdjoint(
            linear_solver=lx.AutoLinearSolver(well_posed=False)
        ),
        throw=False,
    )

    u_star, theta_v_star = sol.value

    # ------------------------------------------------------------------
    # 5.  Post-convergence diagnostics
    # ------------------------------------------------------------------
    L = L_from_state(u_star, theta_v_star, theta_v_atm, theta_v_sfc)
    z0h = zh(z0m, u_star)
    theta_star = Theta_star(z0h, theta_atm, theta_sfc, L)
    # q_* is recomputed from converged L (consistent with CLM postprocessing)
    q_star = Theta_star(z0h, q_atm, q_sfc, L)

    # Aerodynamic resistance: rah = dlg_m · dlg_h / (κ² Va)
    #   (uses the *effective* Va for heat flux)
    # theta_v_atm = theta_2_theta_v(theta_atm, q_atm)
    # Va_conv = Va_convective(u, L, theta_vstar, theta_v_atm)
    # rah = dlg_m(z0m, L) * dlg_h(z0h, L) / (kam**2 * Va_conv)
    rah = (theta_atm - theta_sfc) / (u_star * theta_star)
    raw = (q_atm - q_sfc) / (u_star * q_star)

    H_valid = -rho * CP * (theta_atm - theta_sfc) / rah
    H = jnp.where(jnp.isnan(rah) | (rah == 0.0), 0.0, H_valid)
    # LE = -rho * LAM_V * (q_atm - q_sfc) / raw
    LE_valid = -rho * LAM_V * (q_atm - q_sfc) / raw
    LE = jnp.where(jnp.isnan(raw) | (raw == 0.0), 0.0, LE_valid)

    return L, u_star, theta_v_star, q_star, H, LE


@jax.jit
def solve_turbulence(z0m, T_atm, T_sfc, q_atm, q_sfc, u, rho, P):
    """Solve MO similarity for turbulent fluxes via L-iteration (alternative to solve_mo_coupled)."""
    theta_atm = T_2_theta(T_atm, P)
    theta_sfc = T_2_theta(T_sfc, P)
    theta_v_atm = theta_2_theta_v(theta_atm, q_atm)
    theta_v_sfc = theta_2_theta_v(theta_sfc, q_sfc)

    def while_body_func(L, args):
        u_star = U_star(z0m, u, L)
        z0h = zh(z0m, u_star)
        theta_star = Theta_star(z0h, theta_atm, theta_sfc, L)
        q_star = Theta_star(z0h, q_atm, q_sfc, L)
        theta_v_star = Theta_v_star(theta_star, theta_atm, theta_sfc, q_atm, q_star)
        L_new = L_renew(u_star, theta_v_star, theta_v_atm, theta_v_sfc)
        return L_new

    solver = optx.FixedPointIteration(
        rtol=1e-1,
        atol=1e-2,
    )

    Rib = Rib_for_first_guess(theta_v_atm, theta_v_sfc, u)
    fist_guss_L = (config.Z_ATM - D) / zeta_for_first_guess(z0m, Rib)
    init_val = jnp.clip(fist_guss_L, -1e6, 1e6)

    sol = optx.fixed_point(
        while_body_func,
        solver,
        init_val,
        args=(z0m, theta_atm, theta_sfc, q_atm, q_sfc, u, rho),
        max_steps=5,
        adjoint=optx.ImplicitAdjoint(
            linear_solver=lx.AutoLinearSolver(well_posed=False)
        ),
        # adjoint=optx.RecursiveCheckpointAdjoint(),
        throw=False,
    )
    L = sol.value
    u_star = U_star(z0m, u, L)
    z0h = zh(z0m, u_star)
    # theta_star = Theta_star(z0h, theta_atm, theta_sfc, L)
    q_star = Theta_star(z0h, q_atm, q_sfc, L)
    raw = (q_atm - q_sfc) / (u_star * q_star)
    rah = dlg_m(z0m, L) * dlg_h(z0h, L) / 0.16 / u
    H = rho * CP * (T_sfc - T_atm) / rah
    LE = -rho * LAM_V * (q_atm - q_sfc) / raw
    return L, rah, u_star, H, LE
