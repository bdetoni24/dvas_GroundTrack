"""
2D orbit simulator around a user-selected central body.

The orbit shape depends on (r_p, a, e, ω, θ₀):
    e = 0      -> circular
    0 < e < 1  -> elliptic
    e = 1      -> parabolic
    e > 1      -> hyperbolic (shows v∞ and θ∞)

The satellite is animated using Kepler's equation (M = M0 + n·t).
Hovering over the orbit reveals local state (TA, r, v, fpa).
Zoom slider keeps the central body centred.
"""
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox,
    QPushButton, QSlider, QGroupBox, QFormLayout, QTextEdit, QScrollArea,
    QFrame,
)

from orbital import mean_to_true_anomaly


# Celestial bodies (μ in km³/s², R in km)
PLANETS = {
    'Earth':   {'mu': 398600.4418,      'R': 6378.137, 'color': '#58a6ff'},
    'Moon':    {'mu': 4902.8000,        'R': 1737.4,   'color': '#bbbbbb'},
    'Mars':    {'mu': 42828.314,        'R': 3389.5,   'color': '#cc5533'},
    'Venus':   {'mu': 324858.59,        'R': 6051.8,   'color': '#d9a066'},
    'Mercury': {'mu': 22031.78,         'R': 2439.7,   'color': '#888888'},
    'Jupiter': {'mu': 126686534.0,      'R': 69911.0,  'color': '#d7a875'},
    'Saturn':  {'mu': 37931187.0,       'R': 58232.0,  'color': '#c9b580'},
    'Sun':     {'mu': 1.32712440018e11, 'R': 695700.0, 'color': '#e3b341'},
}


class OrbitSimulator2D(QWidget):
    """2D orbit simulator around a user-selectable planet."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.planet = 'Earth'
        self.mu = PLANETS[self.planet]['mu']
        self.R_p = PLANETS[self.planet]['R']

        self.e = 0.5
        self.rp = 8000.0
        self.a = self.rp / (1 - self.e)
        self.argp = 0.0
        self.theta0 = 0.0
        self.zoom = 1.0
        self._updating = False

        self.playing = False
        self.sim_time = 0.0
        self.speed = 50.0

        self._theta_arr = None
        self._xy_arr = None
        self._sat_marker = None
        self._sat_radius_line = None
        self.tooltip = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ----------------- Left sidebar (matches main style) -----------
        side_container = QWidget()
        side_container.setObjectName('sim_sidebar')
        side = QVBoxLayout(side_container)
        side.setContentsMargins(10, 10, 10, 10)
        side.setSpacing(10)

        title = QLabel('2D Orbit Simulator')
        title.setObjectName('title')
        side.addWidget(title)
        subtitle = QLabel('Interactive conic-section analyzer')
        subtitle.setObjectName('subtitle')
        side.addWidget(subtitle)

        sep = QFrame()
        sep.setObjectName('separator')
        sep.setFrameShape(QFrame.HLine)
        side.addWidget(sep)

        # Planet
        pg = QGroupBox('Central body')
        pfl = QVBoxLayout(pg)
        self.planet_combo = QComboBox()
        for name in PLANETS:
            self.planet_combo.addItem(name)
        pfl.addWidget(self.planet_combo)
        self.planet_info = QLabel('')
        self.planet_info.setObjectName('info')
        self.planet_info.setWordWrap(True)
        pfl.addWidget(self.planet_info)
        side.addWidget(pg)

        # Orbital parameters
        og = QGroupBox('Orbital parameters')
        form = QFormLayout(og)
        form.setLabelAlignment(Qt.AlignRight)

        self.spin_rp = QDoubleSpinBox()
        self.spin_rp.setDecimals(1)
        self.spin_rp.setRange(100.0, 1e9)
        self.spin_rp.setValue(8000.0)
        self.spin_rp.setSuffix(' km')
        self.spin_rp.setMinimumWidth(110)
        form.addRow('r_p  (pericenter):', self.spin_rp)

        self.spin_a = QDoubleSpinBox()
        self.spin_a.setDecimals(1)
        self.spin_a.setRange(-1e9, 1e9)
        self.spin_a.setValue(self.a)
        self.spin_a.setSuffix(' km')
        self.spin_a.setMinimumWidth(110)
        form.addRow('a   (semi-major axis):', self.spin_a)

        self.spin_e = QDoubleSpinBox()
        self.spin_e.setDecimals(3)
        self.spin_e.setRange(0.0, 3.0)
        self.spin_e.setSingleStep(0.05)
        self.spin_e.setValue(0.5)
        form.addRow('e   (eccentricity):', self.spin_e)

        self.spin_argp = QDoubleSpinBox()
        self.spin_argp.setDecimals(1)
        self.spin_argp.setRange(0.0, 360.0)
        self.spin_argp.setSingleStep(5.0)
        self.spin_argp.setSuffix(' °')
        form.addRow('ω   (arg. of pericenter):', self.spin_argp)

        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setDecimals(1)
        self.spin_theta.setRange(-180.0, 180.0)
        self.spin_theta.setSingleStep(5.0)
        self.spin_theta.setSuffix(' °')
        form.addRow('θ₀  (initial true anom.):', self.spin_theta)

        side.addWidget(og)

        # View (zoom)
        vg = QGroupBox('View')
        vv = QVBoxLayout(vg)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel('Zoom:'))
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(10, 500)    # 0.10× … 5.00×
        self.slider_zoom.setValue(100)        # 1.00×
        zoom_row.addWidget(self.slider_zoom)
        self.lbl_zoom = QLabel('1.00 ×')
        self.lbl_zoom.setObjectName('value')
        self.lbl_zoom.setMinimumWidth(55)
        zoom_row.addWidget(self.lbl_zoom)
        vv.addLayout(zoom_row)
        zbtn_row = QHBoxLayout()
        self.btn_zoom_reset = QPushButton('Reset view')
        zbtn_row.addWidget(self.btn_zoom_reset)
        vv.addLayout(zbtn_row)
        side.addWidget(vg)

        # Animation
        ag = QGroupBox('Animation')
        av = QVBoxLayout(ag)
        row = QHBoxLayout()
        self.btn_play = QPushButton('▶  Play')
        self.btn_play.setObjectName('play')
        self.btn_play.setCheckable(True)
        self.btn_reset = QPushButton('⟲  Reset')
        row.addWidget(self.btn_play); row.addWidget(self.btn_reset)
        av.addLayout(row)
        sr = QHBoxLayout()
        sr.addWidget(QLabel('Speed:'))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 1000)
        self.slider_speed.setValue(50)
        sr.addWidget(self.slider_speed)
        self.lbl_speed = QLabel('50 ×')
        self.lbl_speed.setObjectName('value')
        self.lbl_speed.setMinimumWidth(55)
        sr.addWidget(self.lbl_speed)
        av.addLayout(sr)
        side.addWidget(ag)

        # Info
        info_box = QGroupBox('Orbit info')
        il = QVBoxLayout(info_box)
        il.setContentsMargins(4, 4, 4, 4)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(200)
        il.addWidget(self.info_text)
        side.addWidget(info_box)

        side.addStretch(1)

        credits = QLabel('Made by De Toni Bernardo & Da Ros Nicola')
        credits.setObjectName('credits')
        credits.setAlignment(Qt.AlignCenter)
        side.addWidget(credits)

        # Scroll area wrapping the sidebar, matching the main app look
        side_scroll = QScrollArea()
        side_scroll.setWidget(side_container)
        side_scroll.setWidgetResizable(True)
        side_scroll.setMinimumWidth(320)
        side_scroll.setMaximumWidth(400)
        root.addWidget(side_scroll)

        # ----------------- Plot -----------------
        self.fig = Figure(figsize=(7, 7), facecolor='#0d1117')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0d1117')
        root.addWidget(self.canvas, 1)

        # Connections
        self.planet_combo.currentIndexChanged.connect(self._on_planet_changed)
        self.spin_rp.valueChanged.connect(self._on_rp_changed)
        self.spin_a.valueChanged.connect(self._on_a_changed)
        self.spin_e.valueChanged.connect(self._on_e_changed)
        self.spin_argp.valueChanged.connect(lambda _: self._recompute())
        self.spin_theta.valueChanged.connect(lambda _: self._recompute())
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.btn_reset.clicked.connect(self._on_reset)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        self.slider_zoom.valueChanged.connect(self._on_zoom_changed)
        self.btn_zoom_reset.clicked.connect(self._on_zoom_reset)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._on_tick)

        self._on_planet_changed(0)

    # ==================================================================
    #   Parameter coupling  (rp, a, e)
    # ==================================================================
    def _sync_a_from_rp_e(self):
        e = self.spin_e.value()
        rp = self.spin_rp.value()
        if abs(e - 1.0) < 1e-3:
            a_val = np.inf
            self.spin_a.setEnabled(False)
            self.spin_a.blockSignals(True)
            self.spin_a.setValue(0.0)
            self.spin_a.blockSignals(False)
        else:
            a_val = rp / (1 - e)   # positive for ellipse, negative for hyp.
            self.spin_a.setEnabled(True)
            self.spin_a.blockSignals(True)
            self.spin_a.setValue(a_val)
            self.spin_a.blockSignals(False)

    def _sync_rp_from_a_e(self):
        e = self.spin_e.value()
        a = self.spin_a.value()
        if abs(e - 1.0) < 1e-3:
            return  # parabola: rp is authoritative
        rp = a * (1 - e)
        if rp <= 0:
            return
        self.spin_rp.blockSignals(True)
        self.spin_rp.setValue(rp)
        self.spin_rp.blockSignals(False)

    def _on_rp_changed(self, _v):
        if self._updating: return
        self._updating = True
        self._sync_a_from_rp_e()
        self._updating = False
        self._recompute()

    def _on_a_changed(self, _v):
        if self._updating: return
        self._updating = True
        self._sync_rp_from_a_e()
        self._updating = False
        self._recompute()

    def _on_e_changed(self, _v):
        if self._updating: return
        self._updating = True
        # Keep rp as the authoritative quantity and update a
        self._sync_a_from_rp_e()
        self._updating = False
        self._recompute()

    # ==================================================================
    #   Slots
    # ==================================================================
    def _on_planet_changed(self, _idx):
        name = self.planet_combo.currentText()
        self.planet = name
        self.mu = PLANETS[name]['mu']
        self.R_p = PLANETS[name]['R']
        self.planet_info.setText(
            f'μ = {self.mu:.4g} km³/s²\nR = {self.R_p:.1f} km'
        )
        default_rp = self.R_p * 1.4
        self._updating = True
        self.spin_rp.setRange(self.R_p * 1.01, self.R_p * 500)
        self.spin_rp.setValue(default_rp)
        self.spin_rp.setSingleStep(max(self.R_p * 0.05, 10.0))
        self.spin_a.setRange(-self.R_p * 1000, self.R_p * 1000)
        self.spin_a.setSingleStep(max(self.R_p * 0.1, 10.0))
        self._sync_a_from_rp_e()
        self._updating = False
        self.sim_time = 0.0
        self._recompute()

    def _on_play_toggled(self, checked):
        self.playing = checked
        if checked:
            self.btn_play.setText('❚❚  Pause')
            self.btn_play.setObjectName('pause')
            self.timer.start()
        else:
            self.btn_play.setText('▶  Play')
            self.btn_play.setObjectName('play')
            self.timer.stop()
        self.btn_play.style().polish(self.btn_play)

    def _on_reset(self):
        self.sim_time = 0.0
        self._update_satellite()

    def _on_speed_changed(self, v):
        self.lbl_speed.setText(f'{v} ×')
        self.speed = float(v)

    def _on_zoom_changed(self, v):
        self.zoom = v / 100.0
        self.lbl_zoom.setText(f'{self.zoom:.2f} ×')
        self._apply_zoom()

    def _on_zoom_reset(self):
        self.slider_zoom.setValue(100)

    # ==================================================================
    #   Geometry
    # ==================================================================
    def _semi_latus_rectum(self):
        e = self.e; rp = self.rp
        return 2 * rp if abs(e - 1.0) < 1e-3 else rp * (1 + e)

    def _theta_inf(self):
        return np.arccos(-1.0 / self.e) if self.e > 1.0 else None

    def _build_orbit_points(self):
        e = self.e
        if e < 1.0:
            theta_arr = np.linspace(0, 2 * np.pi, 360)
        elif abs(e - 1.0) < 1e-3:
            theta_max = 2 * np.arctan(np.sqrt(24))
            theta_arr = np.linspace(-theta_max, theta_max, 400)
        else:
            theta_inf = self._theta_inf()
            theta_arr = np.linspace(-0.985 * theta_inf,
                                    0.985 * theta_inf, 400)
        p = self._semi_latus_rectum()
        r = p / (1 + e * np.cos(theta_arr))
        ang = theta_arr + self.argp
        x = r * np.cos(ang); y = r * np.sin(ang)
        return theta_arr, np.column_stack([x, y])

    def _xy_at(self, theta):
        p = self._semi_latus_rectum()
        r = p / (1 + self.e * np.cos(theta))
        ang = theta + self.argp
        return np.array([r * np.cos(ang), r * np.sin(ang)])

    # ==================================================================
    #   Time parametrization (Kepler)
    # ==================================================================
    def _mean_motion_and_M0(self):
        e = self.e; mu = self.mu; rp = self.rp
        if e < 1.0:
            a = rp / (1 - e) if e < 1 else rp
            E0 = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(self.theta0 / 2),
                                np.sqrt(1 + e) * np.cos(self.theta0 / 2))
            M0 = E0 - e * np.sin(E0)
            n = np.sqrt(mu / a ** 3)
        elif abs(e - 1.0) < 1e-3:
            tan_h = np.tan(self.theta0 / 2)
            M0 = 0.5 * tan_h + (1 / 6) * tan_h ** 3
            n = np.sqrt(mu / (2 * rp ** 3))
        else:
            arg = np.sqrt((e - 1) / (e + 1)) * np.tan(self.theta0 / 2)
            arg = np.clip(arg, -0.999, 0.999)
            F0 = 2 * np.arctanh(arg)
            M0 = e * np.sinh(F0) - F0
            a = rp / (e - 1)
            n = np.sqrt(mu / a ** 3)
        return n, M0

    def _theta_at_time(self, t):
        n, M0 = self._mean_motion_and_M0()
        M = M0 + n * t
        try:
            return float(mean_to_true_anomaly(M, self.e))
        except Exception:
            return self.theta0

    # ==================================================================
    #   Drawing
    # ==================================================================
    def _recompute(self):
        self.e = self.spin_e.value()
        self.rp = self.spin_rp.value()
        self.argp = np.radians(self.spin_argp.value())
        self.theta0 = np.radians(self.spin_theta.value())
        self._theta_arr, self._xy_arr = self._build_orbit_points()
        self._redraw()

    def _natural_extent(self):
        """Intrinsic viewport size (zoom = 1)."""
        e = self.e; rp = self.rp
        if e < 1.0:
            ra_v = rp * (1 + e) / (1 - e) if e > 0 else rp
            base = ra_v * 1.15
        else:
            base = float(np.max(np.linalg.norm(self._xy_arr, axis=1))) * 1.05
        return max(base, self.R_p * 2.5)

    def _apply_zoom(self):
        base = self._natural_extent()
        half = base / self.zoom
        self.ax.set_xlim(-half, half)
        self.ax.set_ylim(-half, half)
        self.canvas.draw_idle()

    def _redraw(self):
        self.ax.clear()
        self.ax.set_facecolor('#0d1117')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(color='#30363d', alpha=0.4, linewidth=0.5)
        for s in self.ax.spines.values():
            s.set_color('#30363d')
        self.ax.tick_params(colors='#8b949e')

        e = self.e; rp = self.rp

        # Central body
        circle = Circle((0, 0), self.R_p,
                        facecolor=PLANETS[self.planet]['color'],
                        edgecolor='white', linewidth=0.8,
                        alpha=0.85, zorder=2)
        self.ax.add_patch(circle)
        # Primary focus (central body center)
        self.ax.plot(0, 0, marker='+', color='white',
                     markersize=9, markeredgewidth=1.4, zorder=3)

        # Second focus (ellipses / hyperbolas)
        if abs(e - 1.0) > 1e-3 and e > 1e-4:
            a_val = rp / (1 - e)        # signed
            f2_dist = -2 * a_val * e
            fx = f2_dist * np.cos(self.argp)
            fy = f2_dist * np.sin(self.argp)
            self.ax.plot(fx, fy, marker='o', color='#8b949e',
                         markersize=5, markeredgecolor='white',
                         markeredgewidth=0.8, zorder=5)
            self.ax.annotate('F₂', (fx, fy), xytext=(8, 8),
                             textcoords='offset points',
                             color='#8b949e', fontsize=9)

        # Orbit
        x = self._xy_arr[:, 0]; y = self._xy_arr[:, 1]
        self.ax.plot(x, y, color='#f85149', linewidth=1.8,
                     alpha=0.95, zorder=4)

        # Asymptotes (hyperbola)
        if e > 1.0 and abs(e - 1.0) > 1e-3:
            self._draw_asymptotes()

        # Pericenter
        pp = self._xy_at(0.0)
        self.ax.plot(pp[0], pp[1], marker='o', color='#d29922',
                     markersize=8, markeredgecolor='white', zorder=6)
        self.ax.annotate('  P', (pp[0], pp[1]), color='#e3b341',
                         fontsize=10, fontweight='bold', zorder=7)

        # Apocenter (elliptic only)
        if 1e-3 < e < 1.0:
            ap = self._xy_at(np.pi)
            self.ax.plot(ap[0], ap[1], marker='o', color='#58a6ff',
                         markersize=7, markeredgecolor='white', zorder=6)
            self.ax.annotate('  A', (ap[0], ap[1]), color='#58a6ff',
                             fontsize=10, fontweight='bold', zorder=7)
            self.ax.plot([pp[0], ap[0]], [pp[1], ap[1]],
                         color='#58a6ff', linestyle='--',
                         linewidth=0.6, alpha=0.4, zorder=3)

        # Satellite
        sat_xy = self._xy_at(self._theta_at_time(self.sim_time))
        self._sat_radius_line, = self.ax.plot(
            [0, sat_xy[0]], [0, sat_xy[1]],
            color='#56d4dd', linewidth=0.7, alpha=0.5, zorder=3)
        self._sat_marker, = self.ax.plot(
            [sat_xy[0]], [sat_xy[1]],
            marker='o', markersize=11, markerfacecolor='#e3b341',
            markeredgecolor='white', zorder=10)

        self.tooltip = self.ax.annotate(
            '', xy=(0, 0), xytext=(12, 12),
            textcoords='offset points', fontsize=9,
            color='#e6edf3',
            bbox=dict(facecolor='#161b22', edgecolor='#58a6ff',
                      alpha=0.95, boxstyle='round,pad=0.4'),
            visible=False, zorder=20,
        )

        self.ax.set_title(
            f'{self._orbit_type_name().capitalize()} orbit around {self.planet}',
            color='#58a6ff', fontsize=11, pad=6,
        )
        self.ax.set_xlabel('x [km]', color='#e6edf3')
        self.ax.set_ylabel('y [km]', color='#e6edf3')

        self._apply_zoom()
        self._update_info_panel()

    def _draw_asymptotes(self):
        e = self.e
        theta_inf = self._theta_inf()
        base = self._natural_extent()
        for sign in (+1, -1):
            th_far = sign * 0.985 * theta_inf
            far = self._xy_at(th_far)
            ang = sign * theta_inf + self.argp
            d = np.array([np.cos(ang), np.sin(ang)])
            t_vals = np.linspace(-base * 3, base * 3, 2)
            xs = far[0] + t_vals * d[0]
            ys = far[1] + t_vals * d[1]
            self.ax.plot(xs, ys, color='#8b949e',
                         linestyle=':', linewidth=0.9,
                         alpha=0.55, zorder=3)

    def _update_satellite(self):
        if self._sat_marker is None:
            return
        th = self._theta_at_time(self.sim_time)
        sat_xy = self._xy_at(th)
        self._sat_marker.set_data([sat_xy[0]], [sat_xy[1]])
        if self._sat_radius_line is not None:
            self._sat_radius_line.set_data([0, sat_xy[0]], [0, sat_xy[1]])
        self._update_info_panel()
        self.canvas.draw_idle()

    # ==================================================================
    #   Tick
    # ==================================================================
    def _on_tick(self):
        dt_real = self.timer.interval() / 1000.0
        self.sim_time += dt_real * self.speed
        if self.e > 1.0 and abs(self.e - 1.0) > 1e-3:
            th = self._theta_at_time(self.sim_time)
            theta_inf = self._theta_inf()
            if abs(th) > 0.97 * theta_inf:
                self.sim_time = 0.0
        elif abs(self.e - 1.0) < 1e-3:
            th = self._theta_at_time(self.sim_time)
            if abs(th) > np.radians(150):
                self.sim_time = 0.0
        self._update_satellite()

    # ==================================================================
    #   Info
    # ==================================================================
    def _orbit_type_name(self):
        e = self.e
        if e < 1e-3:                 return 'circular'
        if e < 1.0:                  return 'elliptic'
        if abs(e - 1.0) < 1e-3:      return 'parabolic'
        return 'hyperbolic'

    def _orbit_props(self, theta):
        e = self.e; mu = self.mu; rp = self.rp
        p = self._semi_latus_rectum()
        if abs(e - 1.0) < 1e-3:
            a = np.inf
        else:
            a = rp / (1 - e)   # signed
        h = np.sqrt(mu * abs(p))
        r = abs(p) / (1 + e * np.cos(theta))
        if abs(e - 1.0) < 1e-3:
            v = np.sqrt(2 * mu / r); energy = 0.0
        else:
            v = np.sqrt(mu * (2 / r - 1 / a)); energy = -mu / (2 * a)
        v_perp = h / r
        v_r = mu / h * e * np.sin(theta)
        gamma = np.degrees(np.arctan2(v_r, v_perp))
        return dict(r=r, v=v, v_r=v_r, v_perp=v_perp,
                    gamma=gamma, h=h, p=p, a=a, energy=energy)

    def _update_info_panel(self):
        th = self._theta_at_time(self.sim_time)
        pr = self._orbit_props(th)
        e = self.e; rp = self.rp; mu = self.mu

        L = []
        L.append(f"<b style='color:#58a6ff'>{self.planet}</b><br>")
        L.append(f"<span style='color:#8b949e'>Type:</span> "
                 f"<b style='color:#e3b341'>{self._orbit_type_name()}</b><br>")
        L.append(f"<span style='color:#8b949e'>r<sub>p</sub>:</span> "
                 f"<b style='color:#e3b341'>{rp:.1f} km</b> "
                 f"(h={rp - self.R_p:.1f} km)<br>")
        if e < 1.0:
            ra = rp * (1 + e) / (1 - e) if e > 0 else rp
            L.append(f"<span style='color:#8b949e'>r<sub>a</sub>:</span> "
                     f"<b style='color:#e3b341'>{ra:.1f} km</b><br>")
            L.append(f"<span style='color:#8b949e'>a:</span> "
                     f"<b style='color:#e3b341'>{pr['a']:.1f} km</b><br>")
            T = 2 * np.pi * np.sqrt(pr['a'] ** 3 / mu)
            L.append(f"<span style='color:#8b949e'>Period T:</span> "
                     f"{T:.1f} s = {T / 60:.2f} min = {T / 3600:.3f} h<br>")
        elif abs(e - 1.0) < 1e-3:
            L.append(f"<span style='color:#8b949e'>r<sub>a</sub>:</span> "
                     f"∞ (parabola)<br>")
            L.append(f"<span style='color:#8b949e'>v<sub>esc</sub> at r<sub>p</sub>:</span> "
                     f"{np.sqrt(2 * mu / rp):.3f} km/s<br>")
        else:
            theta_inf = self._theta_inf()
            v_inf = np.sqrt(mu * (e - 1) / rp)
            delta = 2 * np.arcsin(1.0 / e)
            L.append(f"<span style='color:#8b949e'>a:</span> "
                     f"<b style='color:#e3b341'>{pr['a']:.1f} km</b> (negative)<br>")
            L.append(f"<span style='color:#8b949e'>v<sub>∞</sub>:</span> "
                     f"<b style='color:#e3b341'>{v_inf:.3f} km/s</b><br>")
            L.append(f"<span style='color:#8b949e'>θ<sub>∞</sub>:</span> "
                     f"<b style='color:#e3b341'>±{np.degrees(theta_inf):.2f}°</b><br>")
            L.append(f"<span style='color:#8b949e'>δ (deflection):</span> "
                     f"{np.degrees(delta):.2f}°<br>")

        L.append(f"<span style='color:#8b949e'>Energy ξ:</span> "
                 f"<b style='color:#e3b341'>{pr['energy']:+.4f} km²/s²</b><br>")
        L.append(f"<span style='color:#8b949e'>h:</span> "
                 f"{pr['h']:.1f} km²/s &nbsp; "
                 f"<span style='color:#8b949e'>p:</span> {pr['p']:.1f} km<br>")
        L.append(f"<br><b style='color:#58a6ff'>Current state</b><br>")
        L.append(f"<span style='color:#8b949e'>θ:</span> "
                 f"<b style='color:#e3b341'>{np.degrees(th):+.2f}°</b><br>")
        L.append(f"<span style='color:#8b949e'>r:</span> {pr['r']:.1f} km<br>")
        L.append(f"<span style='color:#8b949e'>|v|:</span> "
                 f"<b style='color:#e3b341'>{pr['v']:.3f} km/s</b><br>")
        L.append(f"<span style='color:#8b949e'>v<sub>⊥</sub>:</span> "
                 f"{pr['v_perp']:.3f} km/s &nbsp; "
                 f"<span style='color:#8b949e'>v<sub>r</sub>:</span> "
                 f"{pr['v_r']:.3f} km/s<br>")
        L.append(f"<span style='color:#8b949e'>γ (FPA):</span> "
                 f"<b style='color:#e3b341'>{pr['gamma']:+.3f}°</b><br>")
        self.info_text.setHtml(''.join(L))

    # ==================================================================
    #   Mouse tooltip
    # ==================================================================
    def _on_mouse_move(self, event):
        if self.tooltip is None or self._xy_arr is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.canvas.draw_idle()
            return
        diffs = self._xy_arr - np.array([event.xdata, event.ydata])
        d2 = np.sum(diffs ** 2, axis=1)
        idx = int(np.argmin(d2))
        pt = self._xy_arr[idx]
        xl = self.ax.get_xlim()
        tol = (xl[1] - xl[0]) * 0.025
        if np.sqrt(d2[idx]) > tol:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.canvas.draw_idle()
            return
        theta = float(self._theta_arr[idx])
        pr = self._orbit_props(theta)
        txt = (f'θ = {np.degrees(theta):+.2f}°\n'
               f'r = {pr["r"]:.1f} km\n'
               f'|v| = {pr["v"]:.3f} km/s\n'
               f'v⊥ = {pr["v_perp"]:.3f}, v_r = {pr["v_r"]:.3f}\n'
               f'γ = {pr["gamma"]:+.2f}°')
        self.tooltip.xy = (pt[0], pt[1])
        self.tooltip.set_text(txt)
        self.tooltip.set_visible(True)
        self.canvas.draw_idle()
