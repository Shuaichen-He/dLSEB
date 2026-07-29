"""
Calculate Instantaneous Effective Emissivity for Selected 10-Day Periods

Uses the filtered periods from data/2.data_selection/final_periods.log,
then extracts IRT data from original AWS files for that period.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime, timedelta
import os
from glob import glob

# Physical constants
SIGMA = 5.67e-8  # Stefan-Boltzmann constant [W m⁻² K⁻⁴]

# Base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


# Periods from final_periods.log
PERIODS = {
    'Ejin': ('2024-03-26', '2024-04-04', 'Ejin Desert station/AWS'),
    'Huazhaizi': ('2019-03-29', '2019-04-07', 'Huazhaizi desert station/AWS'),
    'Shenshawo': ('2013-10-14', '2013-10-23', 'Shenshawo sandy desert/AWS'),
}


def load_irt_from_aws(aws_dir, t_start, t_end):
    """Load IRT data from AWS Excel files for a specific time period."""
    aws_files = sorted(glob(os.path.join(aws_dir, '*.xlsx')))
    if not aws_files:
        print(f"    No AWS files found in {aws_dir}")
        return pd.DataFrame()

    dfs = []
    for f in aws_files:
        try:
            df = pd.read_excel(f)
            if 'TIMESTAMP' not in df.columns:
                continue

            df['time'] = pd.to_datetime(df['TIMESTAMP'])

            if 'IRT_1' in df.columns and 'IRT_2' in df.columns:
                df_subset = df[['time', 'IRT_1', 'IRT_2']].copy()
                dfs.append(df_subset)
        except Exception as e:
            print(f"    Error reading {f}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    irt_data = pd.concat(dfs, ignore_index=True)
    irt_data = irt_data.drop_duplicates(subset=['time']).sort_values('time')
    irt_data = irt_data.set_index('time')

    # Filter to period
    mask = (irt_data.index >= t_start) & (irt_data.index <= t_end)
    return irt_data[mask]


def calculate_emissivity(R_ld, R_lu, T_s):
    """Calculate effective emissivity: ε = (R_lu - R_ld) / (σ * T_s^4 - R_ld)"""
    try:
        if any(pd.isna([R_ld, R_lu, T_s])):
            return np.nan
        T_s_K = T_s + 273.15
        sigma_T_s_4 = SIGMA * T_s_K**4
        if sigma_T_s_4 <= R_ld:
            return np.nan
        epsilon = (R_lu - R_ld) / (sigma_T_s_4 - R_ld)
        if 0.6 <= epsilon <= 1.2:
            return epsilon
        return np.nan
    except:
        return np.nan


def get_xtick_labels(t_start_str, t_end_str):
    """Generate xtick labels from date range."""
    start_date = datetime.strptime(t_start_str, "%Y-%m-%d")
    end_date = datetime.strptime(t_end_str, "%Y-%m-%d")
    n_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    xtick_labels = [d.strftime("%m-%d") for d in dates]
    xlabel = f"Date ({start_date.year})"
    return xtick_labels, xlabel


def main():
    # Matplotlib style (matching batch_visualization.py)
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 15,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "savefig.dpi": 300,
    })

    colors = {
        'Ejin': '#E74C3C',      # Red
        'Huazhaizi': '#3498DB', # Blue
        'Shenshawo': '#808080', # Gray
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for station_key, (t_start, t_end, aws_path) in PERIODS.items():
        print(f"\nProcessing: {station_key} ({t_start} to {t_end})")

        # Load filtered meteo data
        meteo_file = os.path.join(BASE_DIR, 'data/2.data_selection', station_key, 'meteo_var.csv')
        meteo_df = pd.read_csv(meteo_file)
        print(f"  Meteo records: {len(meteo_df)}")

        # Load IRT data from AWS
        aws_dir = os.path.join(BASE_DIR, 'data/1.origin', aws_path)
        irt_data = load_irt_from_aws(aws_dir, t_start, t_end)
        print(f"  IRT records in period: {len(irt_data)}")

        if irt_data.empty:
            print(f"  Warning: No IRT data found")
            continue

        # Align by position (both datasets are 30-min interval)
        n_rows = len(meteo_df)
        irt_aligned = irt_data.iloc[:n_rows].reset_index(drop=True)
        meteo_df = meteo_df.reset_index(drop=True)

        # Merge
        merged = pd.concat([meteo_df, irt_aligned], axis=1)
        merged['IRT_mean'] = merged[['IRT_1', 'IRT_2']].mean(axis=1)

        # Calculate emissivity
        merged['epsilon'] = merged.apply(
            lambda row: calculate_emissivity(row['DLR_Cor'], row['ULR_Cor'], row['IRT_mean']), axis=1
        )

        # Plot
        plot_data = merged.dropna(subset=['epsilon']).copy()
        if len(plot_data) > 0:
            ax.plot(plot_data.index, plot_data['epsilon'],
                   label=station_key, color=colors[station_key], linewidth=1.5, alpha=0.8)
            print(f"  Valid ε values: {len(plot_data)}/{len(merged)}")
            print(f"  ε mean: {plot_data['epsilon'].mean():.4f}, std: {plot_data['epsilon'].std():.4f}")

    # Set x-axis ticks at noon (12:00) of each day
    # 480 points = 10 days * 48 points/day, noon is at position 24 of each day
    xtick_positions = np.arange(24, 481, 48)  # 24, 72, 120, ..., 456
    xtick_labels = [str(i) for i in range(1, 11)]  # 1 to 10

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=12)

    ax.set_xlabel('Day', fontsize=15)
    ax.set_ylabel('Effective Emissivity (ε)', fontsize=15)
    ax.set_title('Instantaneous Effective Emissivity from IRT Surface Temperature', fontsize=16, pad=10)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.5, 1.5)
    ax.set_xlim(0, 480)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(BASE_DIR, 'Experiments/discussion')
    os.makedirs(output_dir, exist_ok=True)
    svg_file = os.path.join(output_dir, 'effective_emissivity_comparison.svg')
    plt.savefig(svg_file, format='svg', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {svg_file}")

    plt.show()


if __name__ == '__main__':
    main()
