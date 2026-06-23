"""
generate_sample_data.py
=======================
Generates synthetic I²t test CSV files that match the structure and signal
behaviour of real actuator test data.

All values are physically plausible but entirely fabricated — no proprietary
measurement data is embedded. Safe to publish.

Usage
-----
    python generate_sample_data.py

Output
------
    sample_data/
        actuator_test_1200perMil.csv
        actuator_test_1150perMil.csv
        actuator_test_1100perMil.csv
        actuator_test_1085perMil_hot.csv
        actuator_test_1085perMil_warm.csv
        actuator_test_1050perMil.csv

Column names match what logic.py expects (keyword search):
    time, Velocity actual value, Torque actual value,
    Target Torque, MotorTemp, CoreTemp, DriveTemp
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ── Output folder ─────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Generic column names (no proprietary device/node identifiers) ─────────────
COL_TIME        = "time"
COL_POS_ACT     = "actuator/Position actual value"
COL_POS_TGT     = "actuator/Target position"
COL_VEL_ACT     = "actuator/Velocity actual value"
COL_VEL_TGT     = "actuator/Target velocity"
COL_TRQ_TGT     = "actuator/Target Torque"
COL_TRQ_ACT     = "actuator/Torque actual value"
COL_MOTOR_TEMP  = "actuator/MotorTemp"
COL_CORE_TEMP   = "actuator/CoreTemp"
COL_DRIVE_TEMP  = "actuator/DriveTemp"


def generate_test(
    target_torque_per_mil: int,
    trip_time_s: float,
    duration_s: float = 30.0,
    sample_rate_hz: int = 1000,
    motor_temp_start: float = 52.0,
    core_temp_start: float = 55.0,
    drive_temp_start: float = 48.0,
    osc_freq_hz: float = 1.0,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate one I²t test run.

    Parameters
    ----------
    target_torque_per_mil : int
        Commanded torque in per-mil (e.g. 1085 = 108.5 % of rated).
    trip_time_s : float
        Time at which I²t protection trips and torque drops to ~0.
    duration_s : float
        Total recording length in seconds.
    sample_rate_hz : int
        Samples per second (real hardware runs at 1000 Hz).
    motor_temp_start : float
        Motor winding temperature at t=0 [°C].
    core_temp_start : float
        Core temperature at t=0 [°C]  (stored ×1000 as integer in CSV).
    drive_temp_start : float
        Drive temperature at t=0 [°C] (stored ×1000 as integer in CSV).
    osc_freq_hz : float
        Back-and-forth oscillation frequency of the actuator [Hz].
    rng_seed : int
        Reproducibility seed.
    """
    rng = np.random.default_rng(rng_seed)
    n   = int(duration_s * sample_rate_hz)
    dt  = 1.0 / sample_rate_hz

    # ── Time base (arbitrary epoch, 1 ms spacing) ─────────────────────────────
    t_epoch_start = 1_700_000_000.0          # generic Unix timestamp
    time_raw = t_epoch_start + np.arange(n) * dt

    # ── Relative time (seconds from start) ───────────────────────────────────
    t = np.arange(n) * dt

    # ── Torque signal ─────────────────────────────────────────────────────────
    # Actuator oscillates in position so torque alternates in sign.
    # Magnitude holds near target_torque_per_mil until I²t trips.
    noise_scale = target_torque_per_mil * 0.015          # ±1.5 % noise
    phase       = rng.uniform(0, 2 * np.pi)
    sign        = np.sign(np.sin(2 * np.pi * osc_freq_hz * t + phase))
    sign[sign == 0] = 1

    torque_magnitude = (
        target_torque_per_mil
        + rng.normal(0, noise_scale, n)          # Gaussian noise
        + 5 * np.sin(2 * np.pi * 3.7 * t)       # low-amplitude ripple
    )
    torque_magnitude = np.clip(torque_magnitude, 0, target_torque_per_mil * 1.12)

    torque_actual = sign * torque_magnitude

    # Zero-crossings — brief dip through zero when direction reverses
    zero_window = max(1, int(0.004 * sample_rate_hz))     # 4 ms
    for i in range(1, n):
        if sign[i] != sign[i - 1]:
            lo = max(0, i - zero_window)
            hi = min(n, i + zero_window)
            ramp = np.linspace(0, 1, hi - lo)
            torque_actual[lo:hi] *= ramp if sign[i] > 0 else (1 - ramp)

    # I²t trip — torque collapses after trip_time_s
    trip_idx   = int(trip_time_s * sample_rate_hz)
    decay_len  = int(0.08 * sample_rate_hz)               # 80 ms ramp-down
    decay_end  = min(trip_idx + decay_len, n)
    ramp_down  = np.linspace(1.0, 0.0, decay_end - trip_idx)
    torque_actual[trip_idx:decay_end] *= ramp_down
    torque_actual[decay_end:] = rng.normal(0, 30, n - decay_end)   # residual noise

    # ── Velocity signal ───────────────────────────────────────────────────────
    # Roughly in phase with torque (same sign), peaks ~11 000 raw units.
    vel_peak  = 11_000
    vel_noise = vel_peak * 0.02
    velocity  = (
        sign * vel_peak * (0.92 + 0.08 * np.abs(np.sin(2 * np.pi * osc_freq_hz * t + phase)))
        + rng.normal(0, vel_noise, n)
    )
    velocity[trip_idx:] = rng.normal(0, 50, n - trip_idx)

    # ── Position (cumulative integral of velocity, integer encoder counts) ────
    pos_scale  = 0.08                          # raw vel units → encoder counts/sample
    position   = np.cumsum(velocity * pos_scale).astype(int)
    position  -= position[0]                   # start at 0

    # ── Temperature signals ───────────────────────────────────────────────────
    # Motor temp — warms during run, flattens after trip.
    motor_heat_rate  = 0.08    # °C/s under load
    motor_cool_rate  = 0.01    # °C/s after trip (slow)

    motor_temp = np.zeros(n)
    motor_temp[0] = motor_temp_start
    for i in range(1, n):
        if t[i] < trip_time_s:
            delta = motor_heat_rate * dt + rng.normal(0, 0.002)
        else:
            delta = -motor_cool_rate * dt + rng.normal(0, 0.001)
        motor_temp[i] = motor_temp[i - 1] + delta

    # Core temp — similar but ~3–5 °C higher, slightly slower
    core_heat_rate = 0.06
    core_temp_c    = np.zeros(n)
    core_temp_c[0] = core_temp_start
    for i in range(1, n):
        if t[i] < trip_time_s:
            delta = core_heat_rate * dt + rng.normal(0, 0.002)
        else:
            delta = -0.008 * dt + rng.normal(0, 0.001)
        core_temp_c[i] = core_temp_c[i - 1] + delta
    # Stored as integer × 1000 in real hardware
    core_temp_raw = (core_temp_c * 1000).astype(int)

    # Drive temp — cooler, slowest response
    drive_heat_rate = 0.05
    drive_temp_c    = np.zeros(n)
    drive_temp_c[0] = drive_temp_start
    for i in range(1, n):
        if t[i] < trip_time_s:
            delta = drive_heat_rate * dt + rng.normal(0, 0.002)
        else:
            delta = -0.006 * dt + rng.normal(0, 0.001)
        drive_temp_c[i] = drive_temp_c[i - 1] + delta
    # Stored as integer × 1000 in real hardware
    drive_temp_raw = (drive_temp_c * 1000).astype(int)

    # ── Target signals (command values) ──────────────────────────────────────
    target_torque_arr = np.where(t < 0.5, 0, target_torque_per_mil) * sign
    target_vel_arr    = np.zeros(n, dtype=int)    # velocity-mode not used here

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        COL_TIME:       time_raw,
        COL_POS_ACT:    position,
        COL_POS_TGT:    np.zeros(n, dtype=int),
        COL_VEL_ACT:    velocity.astype(int),
        COL_VEL_TGT:    target_vel_arr,
        COL_TRQ_TGT:    target_torque_arr.astype(int),
        COL_TRQ_ACT:    torque_actual.astype(int),
        COL_MOTOR_TEMP: np.round(motor_temp, 6),
        COL_CORE_TEMP:  core_temp_raw,
        COL_DRIVE_TEMP: drive_temp_raw,
    })
    return df


# ── Test matrix ───────────────────────────────────────────────────────────────
# Torque levels and trip times chosen to fit the inverse-time model
# t = K / (M² − M_base²) with K ≈ 5.5, M_base ≈ 0.95
# These are synthetic values — not measurements from any real system.
TEST_CASES = [
    # (torque_per_mil, trip_time_s, motor_start, core_start, drive_start, filename_suffix, seed)
    (1200, 8.5,  50.0, 53.0, 46.0, "1200perMil",       10),
    (1150, 12.3, 51.0, 54.0, 47.0, "1150perMil",       20),
    (1100, 18.7, 51.5, 54.5, 47.5, "1100perMil",       30),
    (1085, 22.4, 52.0, 55.0, 48.0, "1085perMil_warm",  40),
    (1085, 15.1, 58.5, 61.0, 53.0, "1085perMil_hot",   41),  # higher start temp → shorter trip
    (1050, 38.0, 50.5, 53.5, 46.5, "1050perMil",       50),
]


def main():
    print("Generating synthetic I²t test data ...\n")
    for torque, trip, m_start, c_start, d_start, suffix, seed in TEST_CASES:
        df = generate_test(
            target_torque_per_mil=torque,
            trip_time_s=trip,
            duration_s=45.0,
            motor_temp_start=m_start,
            core_temp_start=c_start,
            drive_temp_start=d_start,
            rng_seed=seed,
        )
        fname = f"actuator_test_{suffix}.csv"
        path  = os.path.join(OUT_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  ✓  {fname}  ({len(df):,} rows,  trip at {trip:.1f} s)")

    print(f"\nAll files written to: {OUT_DIR}")
    print("\nLoad these into the app via  Tab 1 → Load CSV File(s).")
    print("Then use  Tab 2 → Load Results CSV  to fit the inverse-time model.")


if __name__ == "__main__":
    main()
