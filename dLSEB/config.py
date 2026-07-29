"""
Configuration Module for dLSEB Model
---------------------------------------
Defines physical constants, soil layer geometry, optimizer settings,
T0 data assimilation flags, and training hyperparameters shared across
the turbulence, energy balance, and loss/optimization modules.
"""

import jax.numpy as jnp
import os

# =============================================================================
# Physical Parameters (Initial Values)
# =============================================================================

ALPHA = 0.264  # Surface shortwave albedo
SIGMA = 0.903  # Surface emissivity
Z0M = 2.25e-3  # Roughness length for momentum [m]
K = 2.4728  # Soil thermal conductivity [W m^-1 K^-1]
C = 2.18e6  # Soil volumetric heat capacity [J m^-3 K^-1]

# =============================================================================
# Turbulence Module Parameters
# =============================================================================

Z_ATM = 4.5  # Atmospheric reference height [m]
D = 0  # Zero-plane displacement height [m]
KAM = 0.4  # Von Karman constant
G = 9.81  # Gravitational acceleration [m s^-2]
CP = 1004.0  # Air specific heat at constant pressure [J kg^-1 K^-1]

# =============================================================================
# Surface Humidity
# =============================================================================

Q_SFC = 0.0  # Surface specific humidity [kg kg^-1] (assumes dry surface)

# =============================================================================
# Energy Balance Parameters
# =============================================================================

SIGMA0 = 5.670374419e-8  # Stefan-Boltzmann constant [W m^-2 K^-4]
DT = 1800  # Time step [s] (30 minutes)
LAM_V = 2.5e6  # Latent heat of vaporisation [J kg^-1]

# =============================================================================
# Soil Layer Configuration
# =============================================================================

# Boundary depths of soil layers [m]
SOIL_BOUND = jnp.array([0, 0.03, 0.05, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0])
# Center depths of soil layers [m]
SOIL_CENTER = jnp.array([0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.6, 0.85, 1.2, 1.6])
# Layer thickness [m]
DZ = jnp.diff(SOIL_BOUND)
# Distance between layer centers [m]
DCZ = jnp.diff(SOIL_CENTER)

# =============================================================================
# Optimizer Settings
# =============================================================================

# Optimizer type: "adam" (optax) or "sgd" (manual gradient descent)
OPTIMIZER_TYPE = "adam"  # Options: "adam", "sgd"

# BASE_LR = 1e-2  # Adam optimizer learning rate (legacy, for single optimizer)

# Three separate optimizers for different parameter groups:
#   Group 1: alpha, sigma, z0m — radiative & roughness parameters
#   Group 2: k — soil thermal conductivity
#   Group 3: logC — soil volumetric heat capacity (log-transformed)
BASE_LR_ASZ = 1e-2  # Adam lr for [alpha, sigma, z0m]
BASE_LR_K = 1e-1  # Adam lr for [k]
BASE_LR_C = 1  # Adam lr for [logC]

# For SGD
# Per-parameter gradient scale: [alpha, sigma, z0m, k, C]
#   - 1.0  → normal update (default, i.e. no scaling)
#   - 0    → freeze that parameter (zero gradient → no update)
#   - >1   → amplify learning / <1 → dampen learning for that parameter
grad_scale = [1.0, 1.0, 1.0, 1.0, 1.0]

# =============================================================================
# T0 Data Assimilation Settings
# =============================================================================

DO_DA_T0 = True  # Whether to perform T0 data assimilation
USE_TEMP_DA = False  # True = manual SGD, False = Adam optimizer

T0_DA_LR = 100  # T0 data assimilation learning rate (for manual SGD)
T0_LR = 1e-2  # T0 assimilation learning rate

# =============================================================================
# Training Configuration
# =============================================================================

NUM_EPOCHS = 100  # Number of training epochs
SPIN_UP_DAYS = 20  # Spin-up iterations (repeated passes over training period)

# =============================================================================
# Output Directory
# =============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "./output")
