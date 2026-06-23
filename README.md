# I²t Protection Activation Analyzer

A desktop tool for analysing I²t thermal protection activation in electric actuator systems. Built during an engineering internship working with industrial servo drives.

## What it does

Electric actuators have built-in I²t (current-squared-time) thermal protection that trips the drive when the motor is at risk of overheating. This tool automates the detection of the exact moment that protection activates, fits a physics-based inverse-time model to the results, and predicts trip times at untested torque levels.

## How it works

**Tab 1 — Load & Analyse**
- Load CSV files exported from the drive's data logger (1 kHz sample rate)
- Automatically detects activation time using a rolling-median filter and sustained-drop algorithm
- Displays torque, speed, and temperature signals with the detected trip point marked

**Tab 2 — Curve Fit & Predict**
- Fits the inverse-time thermal model: t = K / (M² − M_base²)
- Predicts trip time at any torque level and plots it on the fitted curve

## Getting started

1. Install dependencies: pip install PyQt6 pandas numpy scipy matplotlib
2. Generate sample data: python generate_sample_data.py
3. Run the app: python app.py

## Tech stack

Python 3.10, PyQt6, pandas, scipy, matplotlib