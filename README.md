# I²t Protection Activation Analyzer

A desktop tool for analysing I²t thermal protection activation in electric actuator systems. Built during an engineering internship working with industrial servo drives.

## What it does

Electric actuators have built-in I²t (current-squared-time) thermal protection that trips the drive when the motor is at risk of overheating. This tool automates the detection of the exact moment that protection activates, fits a physics-based inverse-time model to the results, and predicts trip times at untested torque levels.

Without this tool, engineers had to manually inspect time-series CSV files from each test run — a slow and error-prone process.

## How it works

**Tab 1 — Load & Analyse**
- Load one or more CSV files exported from the drive's data logger (1 kHz sample rate)
- Automatically detects the activation time using a rolling-median filter and sustained-drop algorithm
- Displays torque, speed, and temperature signals with the detected trip point marked

**Tab 2 — Curve Fit & Predict**
- Loads the saved results CSV
- Fits the inverse-time thermal model: `t = K / (M² − M_base²)` where M is torque ratio
- Predicts trip time at any torque level and plots it on the fitted curve

## Physics background

The inverse-time model comes from the thermal energy integral. Under constant overload torque M (as a ratio of rated torque), the time to reach the thermal limit follows:

```
t_trip = K / (M² − M_base²)
```

Where:
- `K` — thermal time constant, fitted from experimental data
- `M_base` — effective base torque threshold, below which no trip occurs
- R² > 0.95 indicates an excellent fit

## Tech stack

- Python 3.10+
- PyQt6 — desktop GUI
- pandas — data loading and preprocessing
- scipy — nonlinear curve fitting (`curve_fit`)
- matplotlib — signal and model visualisation

## Getting started

**1. Install dependencies**
```bash
pip install PyQt6 pandas numpy scipy matplotlib
```

**2. Generate sample data**
```bash
python generate_sample_data.py
```
This creates 6 synthetic test CSVs in `sample_data/` at different torque levels, ready to load into the app.

**3. Run the app**
```bash
python app.py
```

**4. Try it end to end**
- Tab 1 → Load CSV File(s) → select files from `sample_data/`
- Click **Detect All Runs** → results saved to `results/I2t_times.csv`
- Tab 2 → Load Results CSV → Fit Inverse-Time Model → enter a torque value → Predict

## Project structure

```
├── app.py                  # PyQt6 GUI (two-tab interface)
├── logic.py                # Signal processing and model fitting (no UI)
├── generate_sample_data.py # Synthetic data generator for demo and testing
├── data/                   # CSV files copied here on load
└── results/                # Detection results saved here
```

## Key engineering decisions

**Rolling median filter** — chosen over a simple threshold because the torque signal alternates in sign (reciprocating actuator motion). The median filter removes zero-crossing spikes without phase distortion, giving a clean magnitude envelope.

**Sustained-drop confirmation** — a single sample below threshold is not enough to declare a trip (noise can dip briefly). The algorithm requires 200 consecutive samples (200 ms at 1 kHz) below threshold before confirming activation.

**Separation of logic and UI** — `logic.py` contains all analysis code with no PyQt6 imports. This means the detection and fitting functions can be tested independently or used in a script without launching the GUI.
