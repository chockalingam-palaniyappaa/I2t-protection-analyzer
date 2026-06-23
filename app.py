"""
app.py  -  Two-tab PyQt6 GUI for I2t Protection Activation Analysis
Tab 1: Load CSVs, view graphs, auto-detect activation time, save results CSV
Tab 2: Load results CSV, fit inverse-time model, predict activation time
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QTextEdit, QDoubleSpinBox,
    QGroupBox, QFormLayout, QSplitter, QSpacerItem, QSizePolicy,
)

from logic import (
    TestRun, preprocess, detect_activation,
    fit_inverse_time, predict_from_model, inverse_time_model,
    copy_to_data_folder, append_result, load_results_csv,
)

# Folder / file paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
RESULTS_CSV = os.path.join(BASE_DIR, "results", "I2t_times.csv")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

plt.style.use("dark_background")
COLORS = {
    "torque_actual": "#f0a500",
    "torque_target": "#58a6ff",
    "speed":         "#7ee787",
    "temp_motor":    "#f85149",
    "temp_drive":    "#d2a8ff",
    "temp_core":     "#ffa657",
    "activation":    "#ffffff",
    "fit":           "#58a6ff",
    "pts":           "#79c0ff",
    "predict_pt":    "#f0a500",
}


# ── Signal canvas (3 stacked plots) ─────────────────────────────────────────
class SignalCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(9, 7), tight_layout=True)
        self.fig.patch.set_facecolor("#0d1117")
        super().__init__(self.fig)
        self.setMinimumWidth(600)
        self.ax_torque = self.fig.add_subplot(311)
        self.ax_speed  = self.fig.add_subplot(312)
        self.ax_temp   = self.fig.add_subplot(313)
        self._style()

    def _style(self):
        for ax in [self.ax_torque, self.ax_speed, self.ax_temp]:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=9)
            ax.xaxis.label.set_color("#8b949e")
            ax.yaxis.label.set_color("#8b949e")
            ax.title.set_color("#e6edf3")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

    def plot_run(self, run: TestRun):
        self.ax_torque.cla()
        self.ax_speed.cla()
        self.ax_temp.cla()
        self._style()

        df = run.df
        t  = df["time_s"]

        # Torque
        self.ax_torque.plot(t, df["torque_target"], color=COLORS["torque_target"], lw=1.5, label="Target Torque")
        self.ax_torque.plot(t, df["torque_actual"], color=COLORS["torque_actual"], lw=1.5, label="Actual Torque")
        self.ax_torque.set_title("Torque vs Time", fontsize=10)
        self.ax_torque.set_ylabel("Torque (per mil)", fontsize=9)
        self.ax_torque.set_xlabel("Time (s)", fontsize=9)
        self.ax_torque.set_yticks([-1000, -500, 0, 500, 1000])
        self.ax_torque.legend(loc="upper right", fontsize=8)
        self.ax_torque.grid(True, alpha=0.2)

        # Speed
        self.ax_speed.plot(t, df["speed_rpm"], color=COLORS["speed"], lw=1.5, label="Speed")
        self.ax_speed.set_title("Speed vs Time", fontsize=10)
        self.ax_speed.set_ylabel("Speed (RPM)", fontsize=9)
        self.ax_speed.set_xlabel("Time (s)", fontsize=9)
        spd_max  = df["speed_rpm"].abs().max()
        spd_step = max(1, round(spd_max / 5))
        self.ax_speed.yaxis.set_major_locator(plt.MultipleLocator(spd_step))
        self.ax_speed.legend(loc="upper right", fontsize=8)
        self.ax_speed.grid(True, alpha=0.2)

        # Temperature
        self.ax_temp.plot(t, df["temp_motor_c"], color=COLORS["temp_motor"], lw=1.5, label="Motor")
        self.ax_temp.plot(t, df["temp_drive_c"], color=COLORS["temp_drive"], lw=1.5, label="Drive")
        self.ax_temp.plot(t, df["temp_core_c"],  color=COLORS["temp_core"],  lw=1.5, label="Core")
        self.ax_temp.set_title("Temperature vs Time", fontsize=10)
        self.ax_temp.set_ylabel("Temperature (C)", fontsize=9)
        self.ax_temp.set_xlabel("Time (s)", fontsize=9)
        self.ax_temp.yaxis.set_major_locator(plt.MultipleLocator(5))
        self.ax_temp.legend(loc="upper right", fontsize=8)
        self.ax_temp.grid(True, alpha=0.2)

        # Activation line
        if run.activation_time is not None:
            for ax in [self.ax_torque, self.ax_speed, self.ax_temp]:
                ax.axvline(run.activation_time, color=COLORS["activation"],
                           linestyle="--", lw=1.2,
                           label=f"Activation {run.activation_time:.2f} s")

        self.fig.tight_layout(pad=1.5)
        self.draw()

    def clear(self):
        for ax in [self.ax_torque, self.ax_speed, self.ax_temp]:
            ax.cla()
        self._style()
        self.draw()


# ── Model canvas (curve fit plot) ────────────────────────────────────────────
class ModelCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 5), tight_layout=True)
        self.fig.patch.set_facecolor("#0d1117")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style()
        # Store model params for prediction marker
        self._K      = None
        self._M_base = None

    def _style(self):
        self.ax.set_facecolor("#161b22")
        self.ax.tick_params(colors="#8b949e", labelsize=9)
        self.ax.xaxis.label.set_color("#8b949e")
        self.ax.yaxis.label.set_color("#8b949e")
        self.ax.title.set_color("#e6edf3")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#30363d")

    def plot_model(self, torques: list, times: list, K: float, M_base: float, R2: float):
        self.ax.cla()
        self._style()
        self._K      = K
        self._M_base = M_base

        torques_arr = np.array(torques)
        times_arr   = np.array(times)

        # Measured data points  x=time, y=torque
        self.ax.scatter(times_arr, torques_arr,
                        color=COLORS["pts"], zorder=5, s=70, label="Measured data")

        # Fitted curve  x=time, y=torque
        asymptote  = M_base * 1000 + 20
        q_range    = np.linspace(asymptote, 2500, 500)
        t_curve    = inverse_time_model(q_range / 1000.0, K, M_base)
        mask       = t_curve > 0
        self.ax.plot(t_curve[mask], q_range[mask],
                     color=COLORS["fit"], lw=2,
                     label=f"Fitted model  K={K:.3f}  M_base={M_base:.4f}  R2={R2:.4f}")

        # Axis limits and ticks
        x_max = max(times_arr) * 1.4
        self.ax.set_xlim(0, x_max)
        raw_step = max(x_max / 6, 1)
        mag      = 10 ** int(np.floor(np.log10(raw_step)))
        x_step   = max(1, mag * round(raw_step / mag))
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(x_step))

        y_min = min(torques_arr) * 0.97
        y_max = max(torques_arr) * 1.03
        self.ax.set_ylim(y_min, y_max)
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(50))

        self.ax.set_xlabel("Trip Time (s)", fontsize=10)
        self.ax.set_ylabel("Torque (per mil)", fontsize=10)
        self.ax.set_title("I2t Inverse-Time Curve Fit", fontsize=11)
        self.ax.legend(fontsize=9, loc="upper right")
        self.ax.grid(True, alpha=0.2)
        self.fig.tight_layout()
        self.draw()

    def plot_prediction_marker(self, torque_per_mil: float, trip_time: float):
        """Add a single coloured marker at the predicted point."""
        if self._K is None:
            return
        # Remove old prediction marker if any
        for artist in self.ax.collections:
            if getattr(artist, "_is_prediction_marker", False):
                artist.remove()
        for line in self.ax.lines:
            if getattr(line, "_is_prediction_marker", False):
                line.remove()

        sc = self.ax.scatter(trip_time, torque_per_mil,
                             color=COLORS["predict_pt"], zorder=10, s=120,
                             marker="*", label=f"Prediction: {torque_per_mil:.0f} per mil -> {trip_time:.3f} s")
        sc._is_prediction_marker = True

        # Dashed crosshair lines
        hl = self.ax.axhline(torque_per_mil, color=COLORS["predict_pt"],
                             linestyle=":", lw=1.0, alpha=0.6)
        vl = self.ax.axvline(trip_time, color=COLORS["predict_pt"],
                             linestyle=":", lw=1.0, alpha=0.6)
        hl._is_prediction_marker = True
        vl._is_prediction_marker = True

        self.ax.legend(fontsize=9, loc="upper right")
        self.draw()

    def clear(self):
        self.ax.cla()
        self._style()
        self._K      = None
        self._M_base = None
        self.draw()


# ── TAB 1: Load & Analyse ────────────────────────────────────────────────────
class Tab1(QWidget):
    def __init__(self):
        super().__init__()
        self.runs: List[TestRun] = []
        self._build_ui()

    def _build_ui(self):
        main     = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setSpacing(8)

        self.load_btn = QPushButton("Load CSV File(s)")
        self.load_btn.setFixedHeight(38)
        self.load_btn.clicked.connect(self.load_csvs)
        lv.addWidget(self.load_btn)

        lv.addWidget(QLabel("Loaded Test Runs:"))
        self.run_list = QListWidget()
        self.run_list.setMaximumHeight(180)
        self.run_list.currentRowChanged.connect(self.on_run_selected)
        lv.addWidget(self.run_list)

        meta_box  = QGroupBox("Selected Run")
        meta_form = QFormLayout(meta_box)
        self.lbl_file   = QLabel("-")
        self.lbl_torque = QLabel("-")
        self.lbl_act    = QLabel("-")
        meta_form.addRow("File:",            self.lbl_file)
        meta_form.addRow("Target Torque:",   self.lbl_torque)
        meta_form.addRow("Activation Time:", self.lbl_act)
        lv.addWidget(meta_box)

        self.detect_btn = QPushButton("Detect Activation Time")
        self.detect_btn.setFixedHeight(36)
        self.detect_btn.clicked.connect(self.detect_single)
        lv.addWidget(self.detect_btn)

        self.detect_all_btn = QPushButton("Detect All Runs")
        self.detect_all_btn.setFixedHeight(36)
        self.detect_all_btn.clicked.connect(self.detect_all)
        lv.addWidget(self.detect_all_btn)

        self.folder_btn = QPushButton("Open Results Folder")
        self.folder_btn.setFixedHeight(36)
        self.folder_btn.clicked.connect(self.open_results_folder)
        lv.addWidget(self.folder_btn)

        lv.addWidget(QLabel("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        lv.addWidget(self.log_box)

        lv.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        right = QWidget()
        rv    = QVBoxLayout(right)
        self.canvas = SignalCanvas()
        rv.addWidget(self.canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([300, 950])

    def log(self, msg):
        self.log_box.append(msg)

    def get_selected_run(self) -> Optional[TestRun]:
        idx = self.run_list.currentRow()
        return self.runs[idx] if 0 <= idx < len(self.runs) else None

    def _resolve_csv_path(self) -> str:
        if not hasattr(self, "_active_csv"):
            self._active_csv = RESULTS_CSV
        if not os.path.exists(self._active_csv):
            return self._active_csv
        if not hasattr(self, "_csv_choice_made"):
            existing_name = os.path.basename(self._active_csv)
            msg = (
                "Results file already exists: " + existing_name + "\n\n"
                "Append to the same file?\n\n"
                "Yes -> append to existing file\n"
                "No  -> create a new numbered file"
            )
            reply = QMessageBox.question(
                self, "Results File", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                base = os.path.splitext(RESULTS_CSV)[0]
                i = 1
                while True:
                    candidate = f"{base}_{i}.csv"
                    if not os.path.exists(candidate):
                        self._active_csv = candidate
                        break
                    i += 1
                self.log(f"New results file: {os.path.basename(self._active_csv)}")
            else:
                self.log(f"Appending to: {os.path.basename(self._active_csv)}")
            self._csv_choice_made = True
        return self._active_csv

    def load_csvs(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open CSV Files", "", "CSV Files (*.csv)")
        if not paths:
            return
        already = {r.file_name for r in self.runs}
        for path in paths:
            fname = os.path.basename(path)
            if fname in already:
                self.log(f"Skipped (already loaded): {fname}")
                continue
            try:
                raw      = pd.read_csv(path)
                df       = preprocess(raw)
                dst      = copy_to_data_folder(path, DATA_FOLDER)
                run      = TestRun(file_name=fname, file_path=dst, df=df)
                self.runs.append(run)
                already.add(fname)
                self.run_list.addItem(QListWidgetItem(fname))
                self.log(f"Loaded: {fname}  ({len(df)} rows)")
            except Exception as e:
                self.log(f"ERROR loading {fname}: {e}")
        if self.runs and self.run_list.currentRow() == -1:
            self.run_list.setCurrentRow(0)

    def on_run_selected(self, idx: int):
        run = self.get_selected_run()
        if run is None:
            self.canvas.clear()
            return
        self.lbl_file.setText(run.file_name)
        self.lbl_torque.setText(f"{run.target_torque:.0f} per mil" if run.target_torque is not None else "-")
        self.lbl_act.setText(f"{run.activation_time:.3f} s" if run.activation_time is not None else "-")
        self.canvas.plot_run(run)

    def detect_single(self):
        run = self.get_selected_run()
        if run is None:
            QMessageBox.warning(self, "No Selection", "Select a test run first.")
            return
        act_time, target_torque = detect_activation(run.df)
        run.activation_time = act_time
        run.target_torque   = target_torque
        if act_time is None:
            self.lbl_act.setText("Not found")
            self.log(f"{run.file_name}: no activation detected.")
        else:
            self.lbl_act.setText(f"{act_time:.3f} s")
            self.lbl_torque.setText(f"{target_torque:.0f} per mil")
            csv_path = self._resolve_csv_path()
            append_result(run, csv_path)
            self.log(f"{run.file_name}: {act_time:.3f} s | torque {target_torque:.0f} per mil -> saved to {os.path.basename(csv_path)}")
        self.canvas.plot_run(run)

    def detect_all(self):
        if not self.runs:
            QMessageBox.information(self, "No Runs", "Load at least one CSV first.")
            return
        csv_path = self._resolve_csv_path()
        saved = 0
        for run in self.runs:
            act_time, target_torque = detect_activation(run.df)
            run.activation_time = act_time
            run.target_torque   = target_torque
            if act_time is not None:
                append_result(run, csv_path)
                saved += 1
                self.log(f"{run.file_name}: {act_time:.3f} s | torque {target_torque:.0f} -> saved")
            else:
                self.log(f"{run.file_name}: not found")
        self.on_run_selected(self.run_list.currentRow())
        self.log(f"-- Done: {saved}/{len(self.runs)} results saved to {os.path.basename(csv_path)} --")

    def open_results_folder(self):
        import subprocess, platform
        folder = os.path.dirname(RESULTS_CSV)
        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])


# ── TAB 2: Curve Fit & Predict ───────────────────────────────────────────────
class Tab2(QWidget):
    def __init__(self):
        super().__init__()
        self.K      = None
        self.M_base = None
        self.df_results = None
        self._build_ui()

    def _build_ui(self):
        main     = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setSpacing(8)

        self.load_csv_btn = QPushButton("Load Results CSV (I2t_times.csv)")
        self.load_csv_btn.setFixedHeight(38)
        self.load_csv_btn.clicked.connect(self.load_results)
        lv.addWidget(self.load_csv_btn)

        lv.addWidget(QLabel("Loaded Data:"))
        self.data_box = QTextEdit()
        self.data_box.setReadOnly(True)
        self.data_box.setMaximumHeight(160)
        lv.addWidget(self.data_box)

        self.fit_btn = QPushButton("Fit Inverse-Time Model")
        self.fit_btn.setFixedHeight(36)
        self.fit_btn.clicked.connect(self.fit_model)
        lv.addWidget(self.fit_btn)

        # Fitted parameters display
        param_box  = QGroupBox("Fitted Parameters")
        param_form = QFormLayout(param_box)
        self.lbl_K      = QLabel("-")
        self.lbl_M_base = QLabel("-")
        self.lbl_R2     = QLabel("-")
        param_form.addRow("K:",      self.lbl_K)
        param_form.addRow("M_base:", self.lbl_M_base)
        param_form.addRow("R2:",     self.lbl_R2)
        lv.addWidget(param_box)

        # Prediction
        pred_box    = QGroupBox("Predict Activation Time")
        pred_layout = QVBoxLayout(pred_box)
        pred_form   = QFormLayout()
        self.pred_input = QDoubleSpinBox()
        self.pred_input.setRange(1000, 3000)
        self.pred_input.setValue(1200)
        self.pred_input.setSuffix(" per mil")
        pred_form.addRow("Torque:", self.pred_input)
        pred_layout.addLayout(pred_form)

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setFixedHeight(34)
        self.predict_btn.clicked.connect(self.predict)
        pred_layout.addWidget(self.predict_btn)

        self.pred_output = QTextEdit()
        self.pred_output.setReadOnly(True)
        self.pred_output.setMaximumHeight(60)
        pred_layout.addWidget(self.pred_output)
        lv.addWidget(pred_box)

        lv.addWidget(QLabel("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        lv.addWidget(self.log_box)

        right = QWidget()
        rv    = QVBoxLayout(right)
        self.canvas = ModelCanvas()
        rv.addWidget(self.canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([320, 900])

    def log(self, msg):
        self.log_box.append(msg)

    def load_results(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Results CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.df_results = load_results_csv(path)
            self.data_box.setPlainText(self.df_results.to_string(index=False))
            self.log(f"Loaded {len(self.df_results)} rows from {os.path.basename(path)}")
        except Exception as e:
            self.log(f"ERROR: {e}")

    def fit_model(self):
        if self.df_results is None:
            QMessageBox.information(self, "No Data", "Load a results CSV first.")
            return
        df = self.df_results.dropna(subset=["target_torque_per_mil", "activation_time_s"])
        if len(df) < 2:
            QMessageBox.warning(self, "Not Enough Data", "Need at least 2 data points to fit.")
            return
        try:
            torques = df["target_torque_per_mil"].tolist()
            times   = df["activation_time_s"].tolist()
            K, M_base, R2 = fit_inverse_time(torques, times)
            self.K      = K
            self.M_base = M_base
            self.lbl_K.setText(f"{K:.4f}")
            self.lbl_M_base.setText(f"{M_base:.4f}")
            self.lbl_R2.setText(f"{R2:.4f}")
            self.log(f"Fit: K={K:.4f}  M_base={M_base:.4f}  R2={R2:.4f}")
            self.canvas.plot_model(torques, times, K, M_base, R2)
        except Exception as e:
            self.log(f"Fit failed: {e}")

    def predict(self):
        if self.K is None:
            QMessageBox.information(self, "No Model", "Fit the model first.")
            return
        torque = self.pred_input.value()
        try:
            t = predict_from_model(torque, self.K, self.M_base)
            self.pred_output.setPlainText(f"Torque: {torque:.0f} per mil  ->  Trip time: {t:.3f} s")
            self.log(f"Prediction: {torque:.0f} per mil -> {t:.3f} s")
            self.canvas.plot_prediction_marker(torque, t)
        except Exception as e:
            self.log(f"Prediction error: {e}")


# ── Main window ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("I2t Protection Activation Analyzer")
        self.resize(1400, 860)
        self._apply_style()
        tabs = QTabWidget()
        self.tab1 = Tab1()
        self.tab2 = Tab2()
        tabs.addTab(self.tab1, "  Load & Analyse  ")
        tabs.addTab(self.tab2, "  Curve Fit & Predict  ")
        self.setCentralWidget(tabs)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #e6edf3;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QTabWidget::pane { border: 1px solid #30363d; background: #0d1117; }
            QTabBar::tab {
                background: #161b22; color: #8b949e;
                padding: 8px 18px; border: 1px solid #30363d;
                border-bottom: none; font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #0d1117; color: #f0a500;
                border-top: 2px solid #f0a500;
            }
            QPushButton {
                background-color: #21262d; color: #e6edf3;
                border: 1px solid #30363d; border-radius: 4px;
                padding: 6px 14px; font-weight: 600;
            }
            QPushButton:hover { background-color: #30363d; border-color: #f0a500; color: #f0a500; }
            QPushButton:pressed { background-color: #f0a500; color: #0d1117; }
            QListWidget {
                background-color: #161b22; border: 1px solid #30363d; border-radius: 4px;
            }
            QListWidget::item:selected { background-color: #1f3a5f; color: #58a6ff; }
            QGroupBox {
                border: 1px solid #30363d; border-radius: 6px;
                margin-top: 10px; font-weight: 600; color: #58a6ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 6px; color: #58a6ff;
            }
            QTextEdit, QLineEdit {
                background-color: #161b22; border: 1px solid #30363d;
                border-radius: 4px; color: #e6edf3; padding: 4px;
            }
            QDoubleSpinBox {
                background-color: #161b22; border: 1px solid #30363d;
                border-radius: 4px; color: #e6edf3; padding: 3px;
            }
            QLabel { color: #e6edf3; }
            QSplitter::handle { background: #30363d; }
            QScrollBar:vertical { background: #161b22; width: 8px; }
            QScrollBar::handle:vertical { background: #30363d; border-radius: 4px; }
        """)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
