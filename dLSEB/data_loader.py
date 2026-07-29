"""
Data Loading Module for dLSEB Model
-------------------------------------
Reads meteorological forcing, soil temperature/moisture, and eddy-covariance
observations from CSV files, and assembles JAX arrays ready for the energy
balance and training pipeline.

Called by: main.py
"""

import os
import pandas as pd
import jax.numpy as jnp

# Data directory (relative to this file)
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/2.data_selection/Ejin")


def load_all_data():
    """
    Load and process all data for model training/inference.

    Returns:
        dict: Dictionary containing all loaded and processed data
    """
    # Load raw data (all at 30-min resolution after pre-processing)
    data = pd.read_csv(os.path.join(DATA_DIR, "meteo_var.csv"))
    soil_data = pd.read_csv(os.path.join(DATA_DIR, "soil.csv"))
    flux = pd.read_csv(os.path.join(DATA_DIR, "ec_var.csv"))
    air_constant = pd.read_csv(os.path.join(DATA_DIR, "air_constants.csv"))

    # Atmospheric constants
    rho = air_constant["rho"].astype(float).values
    # Tv = air_constant["Tv"].astype(float).values

    # Meteorological forcing
    Rsd = data["DR"].astype(float).values  # Incoming shortwave
    Rld = data["DLR_Cor"].astype(float).values  # Incoming longwave
    T_atm = data["Ta_5m"].astype(float).values + 273.15  # Air temperature [K]
    u = flux["Wnd"].astype(float).values  # Wind speed
    P = data["Press"].astype(float).values  # hPa

    # Atmospheric specific humidity from EC observations [kg/kg]
    # H2O is the observed water vapor content from open-path IRGA
    q_atm = flux["H2O"].astype(float).values / 1000.0  # Convert g/m³ to kg/kg
    # q_atm = jnp.clip(q_atm, 0.0, 0.05)  # Physical bounds

    # Build meteorological input array
    meteo_input = jnp.array([Rsd, Rld, T_atm, q_atm, u, rho, P])

    # Observation constraints
    Rsu = jnp.where(
        data["UR"].astype(float).values > 0, data["UR"].astype(float).values, 0.0
    )
    Rlu = data["ULR_Cor"].astype(float).values
    Hs = flux["Hs"].astype(float).values

    # Ground heat flux (average of 3 plates)
    G = -1 * jnp.array(
        [
            soil_data["Gs_1"].astype(float).values,
            soil_data["Gs_2"].astype(float).values,
            soil_data["Gs_3"].astype(float).values,
        ]
    ).mean(axis=0)

    # Surface temperature [K]
    T_sfc = jnp.array(soil_data["Ts_0cm"].astype(float).values).T + 273.15

    # Soil temperature at multiple depths [K]
    Train_T = (
        jnp.array(
            [
                soil_data["Ts_2cm"].astype(float).values,
                soil_data["Ts_4cm"].astype(float).values,
                soil_data["Ts_10cm"].astype(float).values,
                soil_data["Ts_20cm"].astype(float).values,
            ]
        ).T
        + 273.15
    )

    # First-layer soil moisture [m³/m³]
    # Using volumetric water content at first soil layer (Ms_2cm)
    # Note: Ms_2cm is in % (mass water content), convert to volumetric
    # For sandy soil, bulk density ~ 1.6 g/cm³, so θ ≈ SM% * 0.01 * 1.6
    sm_mass = soil_data["Ms_2cm"].astype(float).values  # Mass water content [%]
    theta_soil = jnp.array(sm_mass) * 0.01 * 1.6  # Convert to volumetric [m³/m³]
    theta_soil = jnp.clip(theta_soil, 0.02, 0.5)  # Physical bounds

    return {
        "meteo_input": meteo_input,
        "Train_T": Train_T,
        "Rsu": Rsu,
        "Rlu": Rlu,
        "Hs": Hs,
        "G": G,
        "T_sfc": T_sfc,
        "theta_soil": theta_soil,  # First-layer soil moisture
    }
