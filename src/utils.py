"""
Utility helpers — data loading, plotting, and common operations.
"""

import torch
import numpy as np
import pandas as pd


def load_experimental_data(csv_path, transmission=60, power_nW=3, column='fwhm'):
    """
    Load and filter experimental FWHM data from the CSV.

    Parameters
    ----------
    csv_path : str
        Path to the experimental data CSV.
    transmission : int
        Filter by transmission value.
    power_nW : int
        Filter by laser power (nW).
    column : str
        Column name containing FWHM values.

    Returns
    -------
    data : tensor (N,)
        Cleaned FWHM values, NaN removed, as float32 tensor.
    """
    df = pd.read_csv(csv_path)
    filtered = df[(df["transmission"] == transmission) & (df["power_nW"] == power_nW)][column]
    data = filtered.dropna().reset_index(drop=True).copy()
    return torch.tensor(data.values, dtype=torch.float32)
