"""
dLSEB Core Model Package
-------------------------
Submodules:
  - turbulence:       Surface layer similarity theory for flux calculation
  - energy_balance:   Surface energy balance, soil heat transfer, spin-up
  - loss_update:      Loss functions (RMSE), parameter constraints, optimizers

Called by: main.py
"""

# Must be set BEFORE importing equinox/lineax/optimistix (inside submodules).
import os

if "EQX_ON_ERROR" not in os.environ:
    os.environ["EQX_ON_ERROR"] = "nan"

# Import commonly used functions for convenience
from .energy_balence import spin_up_soil_T, scan_training_periods
from .loss_update import safe_update, loss

__all__ = [
    "spin_up_soil_T",
    "scan_training_periods",
    "safe_update",
    "loss",
]
