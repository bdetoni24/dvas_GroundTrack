"""
Ground Track Visualizer - Dinamica del volo aerospaziale
Applicazione GUI per la visualizzazione di ground track di satelliti.

Esegui con:  python main.py
"""
import sys
import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QSpinBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QSplitter, QGroupBox, QCheckBox, QSlider, QFormLayout,
    QTextEdit, QScrollArea, QFrame, QSizePolicy, QMessageBox,
)

import orbital as orb
from satellites import SATELLITES, list_satellites, get_satellite_params
from view_2d import GroundTrackView
from view_3d import View3D


# ======================================================================
#  Stile grafico globale
# ======================================================================
DARK_STYLE = """
QMainWindow, QWidget { background-color: #0f1420; color: #dce6f0; }
QGroupBox {
    border: 1px solid #2a3a55; border-radius: 6px;
    margin-top: 12px; padding-top: 6px;
    color: #7aaaff; font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 5px;
}
QLabel { color: #dce6f0; }
QLabel#title { color: #6ab0ff; font-size: 14pt; font-weight: bold; }
QLabel#subtitle { color: #9ab8d8; font-size: 9pt; }
QLabel#info { color: #c8d8e8; font-family: 'Consolas', monospace; font-size: 9pt; }
QLabel#value { color: #ffd866; font-family: 'Consolas', monospace; }
QPushButton {
    background-color: #1f3050; color: #ffffff;
    border: 1px solid #3a5580; border-radius: 4px;
    padding: 6px 14px; font-weight: bold;
}
QPushButton:hover   { background-color: #2a4070; border-color: #5a7ab0; }
QPushButton:pressed { background-color: #3355a0; }
QPushButton#play    { background-color: #1a6640; border-color: #2a9060; }
QPushButton#play:hover { background-color: #2a8050; }
QPushButton#pause   { background-color: #8a4a10; border-color: #b06020; }
QPushButton#pause:hover { background-color: #a05a18; }
QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #1a2438; color: #ffffff;
    border: 1px solid #3a5580; border-radius: 3px; padding: 3px;
}
QComboBox QAbstractItemView {
    background-color: #1a2438; color: #ffffff;
    selection-background-color: #3a5580;
}
QSlider::groove:horizontal {
    background: #1a2438; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #5a7ab0; width: 14px; margin: -6px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #3a5580; }
QTabWidget::pane { border: 1px solid #2a3a55; background: #0a0e18; }
QTabBar::tab {
    background: #1a2438; color: #aacce8; padding: 7px 18px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    font-weight: bold; min-width: 130px;
}
QTabBar::tab:selected { background: #2a4070; color: #ffffff; }
QTextEdit {
    background-color: #080c14; color: #aaccee;
    border: 1px solid #2a3a55; font-family: 'Consolas', monospace;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #3a5580; border-radius: 3px; background: #1a2438;
}
QCheckBox::indicator:checked { background: #4090d0; border: 1px solid #60a0d8; }
QScrollBar:vertical {
    background: #0f1420; width: 10px; border: none;
}
QScrollBar::handle:vertical { background: #2a3a55; border-radius: 5px; min-height: 25px; }
QFrame#separator { background: #2a3a55; max-height: 1px; }
"""


# ======================================================================
#  Control Panel: input parametri Kepleriani + preset + animazione
# ======================================================================
class ControlPanel(QWidget):
    parameters_changed = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    reset_clicked = pyqtSignal()
    frame_changed = pyqtSignal(str)          # 'ECI' o 'ECEF'
    j2_toggled = pyqtSignal(bool)
    duration_changed = pyqtSignal(float)
    ecef_3d_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Titolo
        title = QLabel('Ground Track Visualizer')
        title.setObjectName('title')
        layout.addWidget(title)
        subtitle = QLabel('Dinamica del volo aerospaziale\nVisualizzazione orbitale interattiva')
        subtitle.setObjectName('subtitle')
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setObjectName('separator')
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # Preset satelliti
        preset_box = QGroupBox('Preset satellite')
        pl = QVBoxLayout(preset_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem('— custom —')
        for name in list_satellites():
            self.preset_combo.addItem(name)
        pl.addWidget(self.preset_combo)
        self.preset_desc = QLabel('Scegli un satellite o usa parametri personalizzati.')
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet('color: #88aadd; font-style: italic; font-size: 9pt;')
        pl.addWidget(self.preset_desc)
        layout.addWidget(preset_box)

        # Parametri Kepleriani
        kep_box = QGroupBox('6 Parametri Kepleriani')
        form = QFormLayout(kep_box)
        form.setLabelAlignment(Qt.AlignRight)

        self.spin_a = self._mkspin(6500, 45000, 7000, 1, ' km')
        self.spin_e = self._mkspin(0.0, 0.99, 0.1, 3, '')
        self.spin_i = self._mkspin(0.0, 180.0, 45.0, 2, ' °')
        self.spin_raan = self._mkspin(0.0, 360.0, 0.0, 2, ' °')
        self.spin_argp = self._mkspin(0.0, 360.0, 0.0, 2, ' °')
        self.spin_theta = self._mkspin(0.0, 360.0, 0.0, 2, ' °')

        form.addRow('a  (semiasse maggiore):', self.spin_a)
        form.addRow('e  (eccentricità):', self.spin_e)
        form.addRow('i  (inclinazione):', self.spin_i)
        form.addRow('Ω  (RAAN):', self.spin_raan)
        form.addRow('ω  (arg. pericentro):', self.spin_argp)
        form.addRow('θ₀ (anomalia vera iniz.):', self.spin_theta)

        self.chk_j2 = QCheckBox('Applica perturbazione J2 (SSO, Molniya)')
        form.addRow(self.chk_j2)

        layout.addWidget(kep_box)

        # Animazione
        anim_box = QGroupBox('Animazione')
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
        speed_row.addWidget(QLabel('Velocità:'))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 5000)     # *1x ... *5000x simulato
        self.slider_speed.setValue(500)
        speed_row.addWidget(self.slider_speed)
        self.lbl_speed = QLabel('500 x')
        self.lbl_speed.setMinimumWidth(60)
        self.lbl_speed.setObjectName('value')
        speed_row.addWidget(self.lbl_speed)
        al.addLayout(speed_row)

        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel('Durata (periodi):'))
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 50.0)
        self.spin_duration.setValue(3.0)
        self.spin_duration.setSingleStep(0.5)
        self.spin_duration.setSuffix(' T')
        dur_row.addWidget(self.spin_duration)
        al.addLayout(dur_row)

        layout.addWidget(anim_box)

        # Visualizzazione
        vis_box = QGroupBox('Visualizzazione')
        vl = QVBoxLayout(vis_box)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel('Ground Track 2D:'))
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(['ECEF (terrestre)', 'ECI (inerziale)'])
        frame_row.addWidget(self.frame_combo)
        vl.addLayout(frame_row)

        self.chk_ecef_3d = QCheckBox('Rotazione Terra nel 3D (ECEF)')
        vl.addWidget(self.chk_ecef_3d)

        layout.addWidget(vis_box)

        layout.addStretch(1)

        # Collegamenti
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
            self.preset_desc.setText('Parametri personalizzati.')
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
        # se l'utente modifica manualmente, torna a "custom"
        if self.preset_combo.currentIndex() != 0:
            self._building = True
            self.preset_combo.setCurrentIndex(0)
            self.preset_desc.setText('Parametri personalizzati.')
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
        self.lbl_speed.setText(f'{value} x')
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
            return '#FF4444'
        name = self.preset_combo.itemText(idx)
        params = SATELLITES.get(name, {})
        return params.get('colore', '#FF4444')

    def get_name(self):
        idx = self.preset_combo.currentIndex()
        if idx <= 0:
            return 'Custom'
        return self.preset_combo.itemText(idx)


# ======================================================================
#  Info Panel: grandezze derivate
# ======================================================================
class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(210)
        layout.addWidget(self.text)

    def update_info(self, params, info_dict, sim_time):
        a = params['a']; e = params['e']
        i = np.degrees(params['i']); raan = np.degrees(params['raan'])
        argp = np.degrees(params['argp']); theta = np.degrees(params['theta'])
        lines = []
        lines.append(f"<b style='color:#6ab0ff'>Grandezze derivate</b><br>")
        lines.append(f"<span style='color:#88aadd'>Tipo orbita:</span> "
                     f"<b style='color:#ffd866'>{info_dict['tipo_orbita']}</b><br>")
        lines.append(f"<span style='color:#88aadd'>Periodo T:</span> "
                     f"<b style='color:#ffd866'>{info_dict['periodo']:.1f} s "
                     f"= {info_dict['periodo']/60:.2f} min "
                     f"= {info_dict['periodo']/3600:.3f} h</b><br>")
        lines.append(f"<span style='color:#88aadd'>Pericentro r<sub>p</sub>:</span> "
                     f"<b style='color:#ffd866'>{info_dict['pericentro']:.1f} km</b> "
                     f"(altitudine: {info_dict['pericentro']-orb.R_EARTH:.1f} km)<br>")
        lines.append(f"<span style='color:#88aadd'>Apocentro r<sub>a</sub>:</span> "
                     f"<b style='color:#ffd866'>{info_dict['apocentro']:.1f} km</b> "
                     f"(altitudine: {info_dict['apocentro']-orb.R_EARTH:.1f} km)<br>")
        lines.append(f"<span style='color:#88aadd'>Semilato retto p:</span> "
                     f"{info_dict['semilato_retto']:.1f} km<br>")
        lines.append(f"<span style='color:#88aadd'>Momento ang. h:</span> "
                     f"{info_dict['momento_angolare']:.1f} km²/s<br>")
        lines.append(f"<span style='color:#88aadd'>Energia ξ:</span> "
                     f"{info_dict['energia_specifica']:.3f} km²/s²<br>")
        lines.append(f"<br><b style='color:#6ab0ff'>Stato corrente</b><br>")
        lines.append(f"<span style='color:#88aadd'>Tempo simulato:</span> "
                     f"<b style='color:#ffd866'>{sim_time:.1f} s "
                     f"= {sim_time/60:.2f} min</b><br>")
        lines.append(f"<span style='color:#88aadd'>Altitudine:</span> "
                     f"<b style='color:#ffd866'>{info_dict['altitudine']:.1f} km</b><br>")
        lines.append(f"<span style='color:#88aadd'>|r|:</span> "
                     f"{info_dict['raggio_corrente']:.1f} km<br>")
        lines.append(f"<span style='color:#88aadd'>|v|:</span> "
                     f"<b style='color:#ffd866'>{info_dict['velocita_corrente']:.3f} km/s</b> "
                     f"(v_esc: {info_dict['vel_fuga']:.3f} km/s)<br>")
        lines.append(f"<span style='color:#88aadd'>v<sub>⊥</sub>:</span> "
                     f"{info_dict['vel_ortogonale']:.3f} km/s &nbsp; "
                     f"<span style='color:#88aadd'>v<sub>r</sub>:</span> "
                     f"{info_dict['vel_radiale']:.3f} km/s<br>")
        lines.append(f"<span style='color:#88aadd'>Flight Path Angle γ:</span> "
                     f"<b style='color:#ffd866'>{info_dict['flight_path_angle']:+.3f}°</b><br>")
        self.text.setHtml(''.join(lines))


# ======================================================================
#  MainWindow
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ground Track Visualizer - Dinamica del volo aerospaziale')
        self.resize(1500, 900)

        # Stato simulazione
        self.sim_time = 0.0            # tempo simulato in secondi
        self.speed = 500.0             # moltiplicatore tempo (1x = realtime)
        self.playing = False
        self._precomputed_track = None
        self._track_time_array = None

        # Widgets
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Pannello controlli (sinistra) dentro scroll area
        self.control = ControlPanel()
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidget(self.control)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(320)
        ctrl_scroll.setMaximumWidth(420)
        splitter.addWidget(ctrl_scroll)

        # Area centrale: Tab widget
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        # Ground Track 2D
        self.view_2d = GroundTrackView()
        tab_2d = QWidget()
        l2 = QVBoxLayout(tab_2d)
        l2.setContentsMargins(2, 2, 2, 2)
        l2.addWidget(self.view_2d)
        self.tabs.addTab(tab_2d, 'Ground Track 2D')

        # 3D view
        self.view_3d = View3D()
        tab_3d = QWidget()
        l3 = QVBoxLayout(tab_3d)
        l3.setContentsMargins(2, 2, 2, 2)
        l3.addWidget(self.view_3d)
        self.tabs.addTab(tab_3d, 'Vista 3D Orbitale')

        # Pannello info
        self.info_panel = InfoPanel()
        tab_info = QWidget()
        li = QVBoxLayout(tab_info)
        li.setContentsMargins(6, 6, 6, 6)
        li.addWidget(self.info_panel)
        self.tabs.addTab(tab_info, 'Info Orbita')

        splitter.setSizes([350, 1150])

        # Timer di animazione
        self.timer = QTimer()
        self.timer.setInterval(60)          # ~16-17 fps
        self.timer.timeout.connect(self._on_tick)

        # Segnali
        self.control.parameters_changed.connect(self._recompute_orbit)
        self.control.play_toggled.connect(self._toggle_play)
        self.control.reset_clicked.connect(self._reset_simulation)
        self.control.speed_changed.connect(self._set_speed)
        self.control.frame_changed.connect(self.view_2d.set_frame)
        self.control.duration_changed.connect(lambda _: self._recompute_orbit())
        self.control.ecef_3d_toggled.connect(self.view_3d.set_show_ecef)

        # Preset iniziale: ISS
        self.control.preset_combo.setCurrentIndex(1)

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
            QMessageBox.warning(self, 'Orbita aperta',
                                'Il ground track ha senso solo per orbite chiuse (e<1).')
        if e < 1.0:
            T = orb.orbital_period(a)
        else:
            T = 3600.0
        duration = params['duration_periods'] * T
        N = 900
        t_arr = np.linspace(0, duration, N)
        r_eci, v_eci, theta_arr, raan_arr, argp_arr = orb.propagate_keplerian(
            a, e, i, raan, argp, theta0, t_arr,
            j2=params['j2'],
        )
        r_ecef = orb.eci_array_to_ecef(r_eci, t_arr, theta_g0=0.0)

        lats_ecef, lons_ecef = orb.compute_groundtrack(r_ecef)
        lats_eci, lons_eci = orb.compute_groundtrack(r_eci)

        # Memorizza per animazione
        self._precomputed_track = {
            'r_eci': r_eci, 'r_ecef': r_ecef,
            'theta_arr': theta_arr,
            'raan_arr': raan_arr, 'argp_arr': argp_arr,
            't_arr': t_arr,
            'T': T, 'a': a, 'e': e, 'i': i,
            'raan0': raan, 'argp0': argp, 'theta0': theta0,
            'duration': duration, 'j2': params['j2'],
        }

        # Aggiorna view 2D
        self.view_2d.set_orbit_data(lats_ecef, lons_ecef, lats_eci, lons_eci,
                                    color=color, name=name)

        # Aggiorna view 3D
        self.view_3d.set_orbit(a, e, i, raan, argp, theta0,
                                color=color, name=name)

        # Info corrente
        self._update_info_panel()

    def _update_info_panel(self):
        params = self.control.get_parameters()
        track = self._precomputed_track
        if track is None:
            return
        # Find sim time -> idx
        idx = self._current_index()
        a, e, i = track['a'], track['e'], track['i']
        raan_now = track['raan_arr'][idx]
        argp_now = track['argp_arr'][idx]
        theta_now = track['theta_arr'][idx]
        info = orb.orbital_info(a, e, i, raan_now, argp_now, theta_now)
        self.info_panel.update_info(
            {'a': a, 'e': e, 'i': i, 'raan': raan_now,
             'argp': argp_now, 'theta': theta_now},
            info, self.sim_time)

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
        self.view_2d.set_current_index(idx)
        self.view_3d.set_time(self.sim_time)
        self.view_3d.set_current_anomaly(track['theta_arr'][idx])
        # Aggiorna pannello info a tasso ridotto (5 fps) per non saturare
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
    app.setStyleSheet(DARK_STYLE)
    app.setFont(QFont('Segoe UI', 9))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
