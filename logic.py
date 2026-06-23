"""
logic.py  -  Pure analysis logic, no UI code here.
All detection and model fitting comes from your original scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Detection parameters (from your I2t_finding_time.py)
FILTER_WINDOW   = 50
CONFIRM_SAMPLES = 200


@dataclass
class TestRun:
    file_name:       str
    file_path:       str
    df:              pd.DataFrame
    activation_time: Optional[float] = None
    target_torque:   Optional[float] = None


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(keyword):
        for c in df.columns:
            if keyword in c:
                return c
        raise ValueError(f"Column containing '{keyword}' not found.\nAvailable: {list(df.columns)}")

    out = pd.DataFrame()
    out["time_s"]        = pd.to_numeric(df[find_col("time")],                 errors="coerce")
    out["speed_raw"]     = pd.to_numeric(df[find_col("Velocity actual value")],errors="coerce")
    out["torque_actual"] = pd.to_numeric(df[find_col("Torque actual value")],  errors="coerce")
    out["torque_target"] = pd.to_numeric(df[find_col("Target Torque")],        errors="coerce")
    out["temp_motor"]    = pd.to_numeric(df[find_col("MotorTemp")],            errors="coerce")
    out["temp_drive"]    = pd.to_numeric(df[find_col("DriveTemp")],            errors="coerce")
    out["temp_core"]     = pd.to_numeric(df[find_col("CoreTemp")],             errors="coerce")
    out = out.dropna().reset_index(drop=True)

    if out.empty:
        raise ValueError("No valid numeric rows found after preprocessing.")

    out["time_s"]       = (out["time_s"] - out["time_s"].iloc[0]).round(3)
    out["speed_rpm"]    = out["speed_raw"]   * 1e-3
    out["temp_motor_c"] = out["temp_motor"]
    out["temp_drive_c"] = out["temp_drive"]  * 1e-3
    out["temp_core_c"]  = out["temp_core"]   * 1e-3
    return out


def detect_activation(df: pd.DataFrame) -> tuple[Optional[float], float]:
    torque_abs      = df["torque_actual"].abs()
    torque_filtered = torque_abs.rolling(FILTER_WINDOW, center=True, min_periods=1).median()
    running_mask    = (df["time_s"] >= 2) & (df["time_s"] <= 24)
    rated_running   = torque_filtered[running_mask].median()
    target_torque   = df["torque_target"].abs().mode()[0]
    drop_threshold  = (rated_running + 1000) / 2
    below           = torque_filtered < drop_threshold

    drop_time_s = None
    for i in range(FILTER_WINDOW, len(df) - CONFIRM_SAMPLES):
        if (not below.iloc[i - 1]
                and below.iloc[i]
                and below.iloc[i: i + CONFIRM_SAMPLES].mean() >= 0.90):
            drop_time_s = float(df["time_s"].iloc[i])
            break
    return drop_time_s, float(target_torque)


def inverse_time_model(M, K, M_base):
    return K / (M ** 2 - M_base ** 2)


def fit_inverse_time(torques_per_mil, times_s):
    M    = np.array(torques_per_mil) / 1000.0
    t    = np.array(times_s)
    popt, _ = curve_fit(inverse_time_model, M, t, p0=[10.0, 0.95], maxfev=10000)
    K_fit, M_base_fit = popt
    t_pred    = inverse_time_model(M, K_fit, M_base_fit)
    ss_res    = np.sum((t - t_pred) ** 2)
    ss_tot    = np.sum((t - np.mean(t)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    return float(K_fit), float(M_base_fit), float(r_squared)


def predict_from_model(torque_per_mil, K, M_base):
    M = torque_per_mil / 1000.0
    return float(inverse_time_model(np.array([M]), K, M_base)[0])


def copy_to_data_folder(src_path, data_folder):
    import shutil
    os.makedirs(data_folder, exist_ok=True)
    dst_path = os.path.join(data_folder, os.path.basename(src_path))
    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
    return dst_path


def append_result(run, output_csv):
    if run.activation_time is None:
        return
    row = pd.DataFrame([{
        "file_name":             run.file_name,
        "target_torque_per_mil": run.target_torque,
        "activation_time_s":     run.activation_time,
    }])
    write_header = not os.path.exists(output_csv)
    row.to_csv(output_csv, mode="a", index=False, header=write_header)


def load_results_csv(path):
    return pd.read_csv(path)
