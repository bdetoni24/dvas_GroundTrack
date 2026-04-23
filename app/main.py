"""
Spaceflight Dynamics - Orbital Simulator
Interactive GUI for orbit analysis and ground-track visualization.

Authors: De Toni Bernardo, Da Ros Nicola
Run with:   python main.py
"""
import sys
import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QGroupBox, QCheckBox, QSlider, QFormLayout, QTextEdit, QScrollArea,
    QFrame, QMessageBox,
)

import orbital as orb
from satellites import SATELLITES, list_satellites, get_satellite_params
from view_2d import GroundTrackView
from view_3d import View3D
from view_orbit_sim import OrbitSimulator2D


# ======================================================================
#   Global dark theme
# ======================================================================
APP_STYLE = """
* { font-family: 'Inter','Segoe UI Variable','Segoe UI',sans-serif; }
QMainWindow, QWidget {
    background-color: #0d1117; color: #e6edf3;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: #0d1117; border: none;
}
QGroupBox {
    border: 1px solid #30363d; border-radius: 8px;
    margin-top: 14px; padding: 10px 8px 6px 8px;
    background-color: #161b22;
    color: #58a6ff; font-weight: 600; font-size: 9pt;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px; padding: 0 6px;
    background-color: #0d1117;
}
QLabel { color: #e6edf3; font-size: 9pt; }
QLabel#title    { color: #58a6ff; font-size: 15pt; font-weight: 700;
                  letter-spacing: 0.3px; }
QLabel#subtitle { color: #8b949e; font-size: 9pt; }
QLabel#credits  { color: #6e7681; font-size: 8pt; font-style: italic; }
QLabel#info     { color: #c9d1d9; font-family: 'JetBrains Mono','Cascadia Mono',
                  'Consolas',monospace; font-size: 8pt; }
QLabel#value    { color: #e3b341; font-family: 'JetBrains Mono','Cascadia Mono',
                  'Consolas',monospace; font-weight: 600; }

QPushButton {
    background-color: #21262d; color: #e6edf3;
    border: 1px solid #363b42; border-radius: 6px;
    padding: 6px 14px; font-weight: 600; font-size: 9pt;
}
QPushButton:hover   { background-color: #30363d; border-color: #58a6ff; }
QPushButton:pressed { background-color: #1f6feb; color: #ffffff; }
QPushButton#play    { background-color: #238636; border-color: #2ea043; color: #ffffff; }
QPushButton#play:hover { background-color: #2ea043; }
QPushButton#pause   { background-color: #9e6a03; border-color: #bb8009; color: #ffffff; }
QPushButton#pause:hover { background-color: #bb8009; }

QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #0d1117; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 5px;
    padding: 4px 6px; font-size: 9pt;
    selection-background-color: #1f6feb;
}
QComboBox:hover, QDoubleSpinBox:hover { border-color: #58a6ff; }
QComboBox QAbstractItemView {
    background-color: #161b22; color: #e6edf3;
    selection-background-color: #1f6feb;
    border: 1px solid #30363d;
}

QSlider::groove:horizontal {
    background: #21262d; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #58a6ff; width: 14px; margin: -6px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 2px; }

QTabWidget::pane {
    border: 1px solid #30363d; background: #0d1117;
    border-top-left-radius: 0; border-top-right-radius: 6px;
}
QTabBar::tab {
    background: #0d1117; color: #8b949e;
    padding: 9px 20px;
    border: 1px solid transparent; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    font-weight: 600; font-size: 9pt; min-width: 140px;
}
QTabBar::tab:hover    { color: #e6edf3; }
QTabBar::tab:selected {
    background: #161b22; color: #58a6ff;
    border: 1px solid #30363d; border-bottom: 1px solid #161b22;
}

QTextEdit {
    background-color: #010409; color: #c9d1d9;
    border: 1px solid #30363d; border-radius: 6px;
    font-family: 'JetBrains Mono','Cascadia Mono','Consolas',monospace;
    font-size: 8pt;
}
QCheckBox { spacing: 8px; font-size: 9pt; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #30363d; border-radius: 4px;
    background: #0d1117;
}
QCheckBox::indicator:hover { border-color: #58a6ff; }
QCheckBox::indicator:checked {
    background: #1f6feb; border: 1px solid #58a6ff;
}
QScrollBar:vertical {
    background: #0d1117; width: 10px; border: none;
}
QScrollBar::handle:vertical {
    background: #30363d; border-radius: 5px; min-height: 25px;
}
QScrollBar::handle:vertical:hover { background: #484f58; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QFrame#separator { background: #30363d; max-height: 1px; }
QSplitter::handle { background: #30363d; width: 1px; }
"""


# ======================================================================
#   Control panel: Keplerian parameters + preset + animation
# ======================================================================
class ControlPanel(QWidget):
    parameters_changed = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    reset_clicked = pyqtSignal()
    frame_changed = pyqtSignal(str)          # 'ECI' or 'ECEF'
    j2_toggled = pyqtSignal(bool)
    duration_changed = pyqtSignal(float)
    ecef_3d_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title = QLabel('Spaceflight Dynamics')
        title.setObjectName('title')
        layout.addWidget(title)
        subtitle = QLabel('Interactive orbital simulator')
        subtitle.setObjectName('subtitle')
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setObjectName('separator')
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # Preset satellites
        preset_box = QGroupBox('Satellite preset')
        pl = QVBoxLayout(preset_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem('— custom —')
        for name in list_satellites():
            self.preset_combo.addItem(name)
        pl.addWidget(self.preset_combo)
        self.preset_desc = QLabel('Choose a preset or enter custom parameters.')
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet('color:#8b949e; font-style:italic; font-size:8pt;')
        pl.addWidget(self.preset_desc)
        layout.addWidget(preset_box)

        # Keplerian parameters
        kep_box = QGroupBox('Keplerian elements')
        form = QFormLayout(kep_box)
        form.setLabelAlignment(Qt.AlignRight)

        self.spin_a = self._mkspin(6500, 45000, 7000, 1, ' km')
        self.spin_e = self._mkspin(0.0, 0.99, 0.1, 3, '')
        self.spin_i = self._mkspin(0.0, 180.0, 45.0, 2, ' °')
        self.spin_raan = self._mkspin(0.0, 360.0, 0.0, 2, ' °')
        self.spin_argp = self._mkspin(0.0, 360.0, 0.0, 2, ' °')
        self.spin_theta = self._mkspin(0.0, 360.0, 0.0, 2, ' °')

        form.addRow('a  (semi-major axis):', self.spin_a)
        form.addRow('e  (eccentricity):', self.spin_e)
        form.addRow('i  (inclination):', self.spin_i)
        form.addRow('Ω  (RAAN):', self.spin_raan)
        form.addRow('ω  (arg. of perigee):', self.spin_argp)
        form.addRow('θ₀ (initial true anom.):', self.spin_theta)

        self.chk_j2 = QCheckBox('Apply J2 perturbation (SSO, Molniya)')
        form.addRow(self.chk_j2)

        layout.addWidget(kep_box)

        # Animation
        anim_box = QGroupBox('Animation')
        al = QVBoxLayout(anim_box)

        btn_row = QHBoxLayout()
        self.btn_play = QPushButton('▶  Play')
        self.btn_play.setObjectName('play')
        self.btn_play.setCheckable(True)
        btn_row.addWidget(self.btn_play)
        self.btn_reset = QPushButton('⟲  Reset')
        btn_row.addWidget(self.btn_reset)
        al.addLayout(btn_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel('Speed:'))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 5000)
        self.slider_speed.setValue(500)
        speed_row.addWidget(self.slider_speed)
        self.lbl_speed = QLabel('500 ×')
        self.lbl_speed.setMinimumWidth(60)
        self.lbl_speed.setObjectName('value')
        speed_row.addWidget(self.lbl_speed)
        al.addLayout(speed_row)

        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel('Duration (periods):'))
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 50.0)
        self.spin_duration.setValue(3.0)
        self.spin_duration.setSingleStep(0.5)
        self.spin_duration.setSuffix(' T')
        dur_row.addWidget(self.spin_duration)
        al.addLayout(dur_row)

        layout.addWidget(anim_box)

        # Visualization (context-aware)
        self.vis_box = QGroupBox('Visualization')
        vl = QVBoxLayout(self.vis_box)

        # Frame selector (only for Ground Track tab)
        self.frame_widget = QWidget()
        fr_row = QHBoxLayout(self.frame_widget)
        fr_row.setContentsMargins(0, 0, 0, 0)
        fr_row.addWidget(QLabel('Frame:'))
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(['ECEF (Earth-fixed)', 'ECI (inertial)'])
        fr_row.addWidget(self.frame_combo)
        vl.addWidget(self.frame_widget)

        # 3D ECEF rotation (only for 3D tab)
        self.chk_ecef_3d = QCheckBox('Lock orbit to Earth (ECEF)')
        vl.addWidget(self.chk_ecef_3d)

        layout.addWidget(self.vis_box)

        # Compact info panel
        self.info_box = QGroupBox('Orbit info')
        il = QVBoxLayout(self.info_box)
        il.setContentsMargins(4, 4, 4, 4)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(180)
        il.addWidget(self.info_text)
        layout.addWidget(self.info_box)

        layout.addStretch(1)

        # Credits
        credits = QLabel('Made by De Toni Bernardo & Da Ros Nicola')
        credits.setObjectName('credits')
        credits.setAlignment(Qt.AlignCenter)
        layout.addWidget(credits)

        # Connections
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        for sp in (self.spin_a, self.spin_e, self.spin_i,
                   self.spin_raan, self.spin_argp, self.spin_theta):
            sp.valueChanged.connect(self._on_param_changed)
        self.chk_j2.toggled.connect(self.j2_toggled.emit)
        self.chk_j2.toggled.connect(lambda _: self.parameters_changed.emit())
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.btn_reset.clicked.connect(self.reset_clicked.emit)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        self.spin_duration.valueChanged.connect(self.duration_changed.emit)
        self.frame_combo.currentIndexChanged.connect(self._on_frame_changed)
        self.chk_ecef_3d.toggled.connect(self.ecef_3d_toggled.emit)

        self._building = False

    # ------------------------------------------------------------------
    def _mkspin(self, low, high, value, decimals, suffix=''):
        sp = QDoubleSpinBox()
        sp.setRange(low, high)
        sp.setDecimals(decimals)
        sp.setValue(value)
        if decimals > 2:
            sp.setSingleStep(0.01)
        elif decimals == 2:
            sp.setSingleStep(1.0)
        else:
            sp.setSingleStep(10.0)
        if suffix:
            sp.setSuffix(suffix)
        sp.setMinimumWidth(110)
        return sp

    def _on_preset_changed(self, idx):
        if self._building:
            return
        if idx <= 0:
            self.preset_desc.setText('Custom parameters.')
            return
        name = self.preset_combo.itemText(idx)
        params = get_satellite_params(name)
        if params is None:
            return
        self.preset_desc.setText(params.get('descrizione', ''))
        self._building = True
        self.spin_a.setValue(params['a'])
        self.spin_e.setValue(params['e'])
        self.spin_i.setValue(params['i'])
        self.spin_raan.setValue(params['raan'])
        self.spin_argp.setValue(params['argp'])
        self.spin_theta.setValue(params['theta'])
        self._building = False
        self.parameters_changed.emit()

    def _on_param_changed(self):
        if self._building:
            return
        if self.preset_combo.currentIndex() != 0:
            self._building = True
            self.preset_combo.setCurrentIndex(0)
            self.preset_desc.setText('Custom parameters.')
            self._building = False
        self.parameters_changed.emit()

    def _on_play_toggled(self, checked):
        if checked:
            self.btn_play.setText('❚❚  Pause')
            self.btn_play.setObjectName('pause')
        else:
            self.btn_play.setText('▶  Play')
            self.btn_play.setObjectName('play')
        self.btn_play.style().polish(self.btn_play)
        self.play_toggled.emit(checked)

    def _on_speed_changed(self, value):
        self.lbl_speed.setText(f'{value} ×')
        self.speed_changed.emit(float(value))

    def _on_frame_changed(self, idx):
        frame = 'ECEF' if idx == 0 else 'ECI'
        self.frame_changed.emit(frame)

    # ------------------------------------------------------------------
    def get_parameters(self):
        return {
            'a':     self.spin_a.value(),
            'e':     self.spin_e.value(),
            'i':     np.radians(self.spin_i.value()),
            'raan':  np.radians(self.spin_raan.value()),
            'argp':  np.radians(self.spin_argp.value()),
            'theta': np.radians(self.spin_theta.value()),
            'j2':    self.chk_j2.isChecked(),
            'duration_periods': self.spin_duration.value(),
        }

    def get_color(self):
        idx = self.preset_combo.currentIndex()
        if idx <= 0:
            return '#f85149'
        name = self.preset_combo.itemText(idx)
        params = SATELLITES.get(name, {})
        return params.get('colore', '#f85149')

    def get_name(self):
        idx = self.preset_combo.currentIndex()
        if idx <= 0:
            return 'Custom'
        return self.preset_combo.itemText(idx)

    def set_view_mode(self, mode):
        """Show/hide context-dependent controls. mode ∈ {'3d','ground_track'}."""
        if mode == '3d':
            self.frame_widget.setVisible(False)
            self.chk_ecef_3d.setVisible(True)
            self.vis_box.setTitle('Visualization — 3D view')
        elif mode == 'ground_track':
            self.frame_widget.setVisible(True)
            self.chk_ecef_3d.setVisible(False)
            self.vis_box.setTitle('Visualization — Ground Track')


# ======================================================================
#   Info text helper
# ======================================================================
def update_info_text(text_widget, info_dict, sim_time):
    L = []
    L.append(f"<b style='color:#58a6ff'>Derived quantities</b><br>")
    L.append(f"<span style='color:#8b949e'>Type:</span> "
             f"<b style='color:#e3b341'>{info_dict['tipo_orbita']}</b><br>")
    L.append(f"<span style='color:#8b949e'>T:</span> "
             f"<b style='color:#e3b341'>{info_dict['periodo']/60:.2f} min "
             f"= {info_dict['periodo']/3600:.3f} h</b><br>")
    L.append(f"<span style='color:#8b949e'>r<sub>p</sub>:</span> "
             f"<b style='color:#e3b341'>{info_dict['pericentro']:.1f} km</b> "
             f"(h {info_dict['pericentro']-orb.R_EARTH:.0f} km)<br>")
    L.append(f"<span style='color:#8b949e'>r<sub>a</sub>:</span> "
             f"<b style='color:#e3b341'>{info_dict['apocentro']:.1f} km</b> "
             f"(h {info_dict['apocentro']-orb.R_EARTH:.0f} km)<br>")
    L.append(f"<span style='color:#8b949e'>p:</span> {info_dict['semilato_retto']:.1f} km &nbsp;"
             f"<span style='color:#8b949e'>h:</span> {info_dict['momento_angolare']:.0f} km²/s<br>")
    L.append(f"<span style='color:#8b949e'>ξ:</span> "
             f"{info_dict['energia_specifica']:.3f} km²/s²<br>")
    L.append(f"<br><b style='color:#58a6ff'>Current state</b><br>")
    L.append(f"<span style='color:#8b949e'>t<sub>sim</sub>:</span> "
             f"<b style='color:#e3b341'>{sim_time/60:.2f} min</b><br>")
    L.append(f"<span style='color:#8b949e'>Alt:</span> "
             f"<b style='color:#e3b341'>{info_dict['altitudine']:.1f} km</b> "
             f"&nbsp;|r|: {info_dict['raggio_corrente']:.1f}<br>")
    L.append(f"<span style='color:#8b949e'>|v|:</span> "
             f"<b style='color:#e3b341'>{info_dict['velocita_corrente']:.3f} km/s</b><br>")
    L.append(f"<span style='color:#8b949e'>v<sub>⊥</sub>:</span> "
             f"{info_dict['vel_ortogonale']:.3f}  "
             f"<span style='color:#8b949e'>v<sub>r</sub>:</span> "
             f"{info_dict['vel_radiale']:.3f} km/s<br>")
    L.append(f"<span style='color:#8b949e'>γ (FPA):</span> "
             f"<b style='color:#e3b341'>{info_dict['flight_path_angle']:+.3f}°</b><br>")
    text_widget.setHtml(''.join(L))


# ======================================================================
#   MainWindow
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spaceflight Dynamics — Orbital Simulator')
        self.resize(1500, 900)

        # Simulation state (shared by 3D view and Ground Track)
        self.sim_time = 0.0
        self.speed = 500.0
        self.playing = False
        self._precomputed_track = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.splitter)

        # Left sidebar (Keplerian params + info) — shown for 3D & Ground Track
        self.control = ControlPanel()
        self.ctrl_scroll = QScrollArea()
        self.ctrl_scroll.setWidget(self.control)
        self.ctrl_scroll.setWidgetResizable(True)
        self.ctrl_scroll.setMinimumWidth(320)
        self.ctrl_scroll.setMaximumWidth(400)
        self.splitter.addWidget(self.ctrl_scroll)

        # Tabs: Orbit Simulator 2D → 3D View → Ground Track
        self.tabs = QTabWidget()
        self.splitter.addWidget(self.tabs)

        self.view_sim = OrbitSimulator2D()
        self.tabs.addTab(self.view_sim, '2D Orbit Simulator')

        self.view_3d = View3D()
        self.tabs.addTab(self.view_3d, '3D Orbit View')

        self.view_2d = GroundTrackView()
        self.tabs.addTab(self.view_2d, 'Ground Track')

        self.splitter.setSizes([340, 1160])

        # Main animation timer (3D + Ground Track)
        self.timer = QTimer()
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._on_tick)

        # Signals
        self.control.parameters_changed.connect(self._recompute_orbit)
        self.control.play_toggled.connect(self._toggle_play)
        self.control.reset_clicked.connect(self._reset_simulation)
        self.control.speed_changed.connect(self._set_speed)
        self.control.frame_changed.connect(self.view_2d.set_frame)
        self.control.duration_changed.connect(lambda _: self._recompute_orbit())
        self.control.ecef_3d_toggled.connect(self.view_3d.set_show_ecef)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Initial preset + tab
        self.control.preset_combo.setCurrentIndex(1)  # ISS
        self.tabs.setCurrentIndex(0)
        self._on_tab_changed(0)

    # ------------------------------------------------------------------
    def _on_tab_changed(self, idx):
        if idx == 0:
            # Orbit simulator 2D — hide sidebar, pause main timer
            self.ctrl_scroll.setVisible(False)
            if self.control.btn_play.isChecked():
                self.control.btn_play.setChecked(False)
        else:
            # Pause simulator if running, show sidebar
            if self.view_sim.btn_play.isChecked():
                self.view_sim.btn_play.setChecked(False)
            self.ctrl_scroll.setVisible(True)
            if idx == 1:
                self.control.set_view_mode('3d')
            else:
                self.control.set_view_mode('ground_track')
            self._apply_current_time()

    # ------------------------------------------------------------------
    def _recompute_orbit(self):
        params = self.control.get_parameters()
        color = self.control.get_color()
        name = self.control.get_name()
        a, e, i, raan, argp, theta0 = (
            params['a'], params['e'], params['i'],
            params['raan'], params['argp'], params['theta'],
        )
        if e >= 1.0:
            QMessageBox.warning(self, 'Open orbit',
                                'Ground track only applies to closed orbits (e<1).')
        T = orb.orbital_period(a) if e < 1.0 else 3600.0
        duration = params['duration_periods'] * T
        N = 720
        t_arr = np.linspace(0, duration, N)
        r_eci, v_eci, theta_arr, raan_arr, argp_arr = orb.propagate_keplerian(
            a, e, i, raan, argp, theta0, t_arr, j2=params['j2'],
        )
        r_ecef = orb.eci_array_to_ecef(r_eci, t_arr, theta_g0=0.0)

        lats_ecef, lons_ecef = orb.compute_groundtrack(r_ecef)
        lats_eci, lons_eci = orb.compute_groundtrack(r_eci)

        self._precomputed_track = {
            'r_eci': r_eci, 'r_ecef': r_ecef,
            'theta_arr': theta_arr,
            'raan_arr': raan_arr, 'argp_arr': argp_arr,
            't_arr': t_arr,
            'T': T, 'a': a, 'e': e, 'i': i,
            'raan0': raan, 'argp0': argp, 'theta0': theta0,
            'duration': duration, 'j2': params['j2'],
        }

        # ECEF longitude drift per orbit: Δλ = -ω_E · T
        if e < 1.0:
            delta_lambda_deg = -np.degrees(orb.OMEGA_EARTH * T)
            delta_lambda_deg = ((delta_lambda_deg + 180) % 360) - 180
        else:
            delta_lambda_deg = None

        self.view_2d.set_orbit_data(lats_ecef, lons_ecef, lats_eci, lons_eci,
                                    color=color, name=name,
                                    delta_lambda_deg=delta_lambda_deg)

        self.view_3d.set_orbit(a, e, i, raan, argp, theta0,
                               color=color, name=name)

        self._update_info_panel()

    def _update_info_panel(self):
        track = self._precomputed_track
        if track is None:
            return
        idx = self._current_index()
        a, e, i = track['a'], track['e'], track['i']
        raan_now = track['raan_arr'][idx]
        argp_now = track['argp_arr'][idx]
        theta_now = track['theta_arr'][idx]
        info = orb.orbital_info(a, e, i, raan_now, argp_now, theta_now)
        update_info_text(self.control.info_text, info, self.sim_time)

    def _current_index(self):
        track = self._precomputed_track
        if track is None:
            return 0
        t_arr = track['t_arr']
        t_norm = self.sim_time % track['duration']
        idx = np.searchsorted(t_arr, t_norm)
        return min(idx, len(t_arr) - 1)

    # ------------------------------------------------------------------
    def _on_tick(self):
        dt_real = self.timer.interval() / 1000.0
        self.sim_time += dt_real * self.speed
        self._apply_current_time()

    def _apply_current_time(self):
        track = self._precomputed_track
        if track is None:
            return
        idx = self._current_index()
        cur_tab = self.tabs.currentIndex()
        if cur_tab == 2:
            self.view_2d.set_current_index(idx)
        elif cur_tab == 1:
            self.view_3d.set_time(self.sim_time)
            self.view_3d.set_current_anomaly(track['theta_arr'][idx])
        if not hasattr(self, '_info_throttle'):
            self._info_throttle = 0
        self._info_throttle += 1
        if self._info_throttle >= 5 or not self.playing:
            self._info_throttle = 0
            self._update_info_panel()

    def _toggle_play(self, playing):
        self.playing = playing
        if playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _reset_simulation(self):
        self.sim_time = 0.0
        self._apply_current_time()

    def _set_speed(self, speed):
        self.speed = float(speed)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    # Pick a reasonable default font if available
    families = set(QFontDatabase().families())
    for fam in ('Inter', 'Segoe UI Variable', 'Segoe UI',
                'SF Pro Text', 'Helvetica Neue'):
        if fam in families:
            app.setFont(QFont(fam, 9))
            break
    else:
        app.setFont(QFont('Sans Serif', 9))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
