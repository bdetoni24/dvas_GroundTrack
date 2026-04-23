"""
Simulatore di orbita 2D attorno a un corpo centrale generico.

L'orbita varia in funzione di e, r_p, ω, θ₀:
  e = 0           -> circolare
  0 < e < 1       -> ellittica
  e = 1           -> parabolica
  e > 1           -> iperbolica (vengono mostrati v∞ e θ∞)

Il satellite è animato seguendo la legge di Keplero (M = M0 + n t).
Passando il mouse sopra l'orbita compaiono i dati locali (TA, r, v, fpa).
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
    QPushButton, QSlider, QGroupBox, QFormLayout, QTextEdit, QSizePolicy,
)

from orbital import mean_to_true_anomaly


# Costanti corpi celesti (μ in km³/s², R in km)
PLANETS = {
    'Terra':    {'mu': 398600.4418,      'R': 6378.137, 'color': '#4488ff'},
    'Luna':     {'mu': 4902.8000,        'R': 1737.4,   'color': '#cccccc'},
    'Marte':    {'mu': 42828.314,        'R': 3389.5,   'color': '#cc5533'},
    'Venere':   {'mu': 324858.59,        'R': 6051.8,   'color': '#ddaa66'},
    'Mercurio': {'mu': 22031.78,         'R': 2439.7,   'color': '#999999'},
    'Giove':    {'mu': 126686534.0,      'R': 69911.0,  'color': '#ddaa77'},
    'Saturno':  {'mu': 37931187.0,       'R': 58232.0,  'color': '#ccbb88'},
    'Sole':     {'mu': 1.32712440018e11, 'R': 695700.0, 'color': '#ffaa33'},
}


class OrbitSimulator2D(QWidget):
    """Simulatore di orbita 2D attorno a un pianeta selezionabile."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.planet = 'Terra'
        self.mu = PLANETS[self.planet]['mu']
        self.R_p = PLANETS[self.planet]['R']

        self.e = 0.5
        self.rp = 8000.0
        self.argp = 0.0
        self.theta0 = 0.0

        self.playing = False
        self.sim_time = 0.0
        self.speed = 50.0

        self._theta_arr = None
        self._xy_arr = None
        self._sat_marker = None
        self._sat_radius_line = None
        self.tooltip = None

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ----------------- Pannello sinistro -----------------
        side = QWidget()
        side.setFixedWidth(300)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)

        # Pianeta
        pg = QGroupBox('Corpo centrale')
        pfl = QVBoxLayout(pg)
        self.planet_combo = QComboBox()
        for name in PLANETS:
            self.planet_combo.addItem(name)
        pfl.addWidget(self.planet_combo)
        self.planet_info = QLabel('')
        self.planet_info.setWordWrap(True)
        self.planet_info.setStyleSheet('color:#88aadd; font-size:9pt;')
        pfl.addWidget(self.planet_info)
        sl.addWidget(pg)

        # Parametri orbita
        og = QGroupBox('Parametri orbita')
        form = QFormLayout(og)
        form.setLabelAlignment(Qt.AlignRight)

        self.spin_rp = QDoubleSpinBox()
        self.spin_rp.setDecimals(1)
        self.spin_rp.setRange(100.0, 1e9)
        self.spin_rp.setValue(8000.0)
        self.spin_rp.setSuffix(' km')
        form.addRow('r_p (pericentro):', self.spin_rp)

        self.spin_e = QDoubleSpinBox()
        self.spin_e.setDecimals(3)
        self.spin_e.setRange(0.0, 3.0)
        self.spin_e.setSingleStep(0.05)
        self.spin_e.setValue(0.5)
        form.addRow('e (eccentricità):', self.spin_e)

        self.spin_argp = QDoubleSpinBox()
        self.spin_argp.setDecimals(1)
        self.spin_argp.setRange(0.0, 360.0)
        self.spin_argp.setSingleStep(5.0)
        self.spin_argp.setSuffix(' °')
        form.addRow('ω (arg. peric.):', self.spin_argp)

        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setDecimals(1)
        self.spin_theta.setRange(-180.0, 180.0)
        self.spin_theta.setSingleStep(5.0)
        self.spin_theta.setSuffix(' °')
        form.addRow('θ₀ (anom. iniz.):', self.spin_theta)

        sl.addWidget(og)

        # Animazione
        ag = QGroupBox('Animazione')
        av = QVBoxLayout(ag)
        row = QHBoxLayout()
        self.btn_play = QPushButton('▶  Play')
        self.btn_play.setObjectName('play')
        self.btn_play.setCheckable(True)
        self.btn_reset = QPushButton('⟲  Reset')
        row.addWidget(self.btn_play); row.addWidget(self.btn_reset)
        av.addLayout(row)
        sr = QHBoxLayout()
        sr.addWidget(QLabel('Velocità:'))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 1000)
        self.slider_speed.setValue(50)
        sr.addWidget(self.slider_speed)
        self.lbl_speed = QLabel('50 x')
        self.lbl_speed.setObjectName('value')
        self.lbl_speed.setMinimumWidth(60)
        sr.addWidget(self.lbl_speed)
        av.addLayout(sr)
        sl.addWidget(ag)

        # Info
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(220)
        sl.addWidget(self.info_text)
        sl.addStretch(1)
        root.addWidget(side)

        # ----------------- Plot -----------------
        self.fig = Figure(figsize=(7, 7), facecolor='#02030a')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#02030a')
        root.addWidget(self.canvas, 1)

        # Connessioni
        self.planet_combo.currentIndexChanged.connect(self._on_planet_changed)
        self.spin_rp.valueChanged.connect(lambda _: self._recompute())
        self.spin_e.valueChanged.connect(lambda _: self._recompute())
        self.spin_argp.valueChanged.connect(lambda _: self._recompute())
        self.spin_theta.valueChanged.connect(lambda _: self._recompute())
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.btn_reset.clicked.connect(self._on_reset)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        # Timer animazione
        self.timer = QTimer()
        self.timer.setInterval(50)   # 20 fps
        self.timer.timeout.connect(self._on_tick)

        self._on_planet_changed(0)

    # ==================================================================
    #   Slot
    # ==================================================================
    def _on_planet_changed(self, _idx):
        name = self.planet_combo.currentText()
        self.planet = name
        self.mu = PLANETS[name]['mu']
        self.R_p = PLANETS[name]['R']
        self.planet_info.setText(
            f"μ = {self.mu:.4g} km³/s²\nR = {self.R_p:.1f} km"
        )
        # Default: rp = 1.4 R
        default_rp = self.R_p * 1.4
        self.spin_rp.blockSignals(True)
        self.spin_rp.setRange(self.R_p * 1.01, self.R_p * 200)
        self.spin_rp.setValue(default_rp)
        self.spin_rp.setSingleStep(max(self.R_p * 0.05, 10.0))
        self.spin_rp.blockSignals(False)
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
        self.lbl_speed.setText(f'{v} x')
        self.speed = float(v)

    # ==================================================================
    #   Geometria
    # ==================================================================
    def _semi_latus_rectum(self):
        e = self.e; rp = self.rp
        return 2 * rp if abs(e - 1.0) < 1e-3 else rp * (1 + e)

    def _theta_inf(self):
        e = self.e
        if e <= 1.0:
            return None
        return np.arccos(-1.0 / e)

    def _build_orbit_points(self):
        """Campiona θ lungo l'orbita per il plot e calcola (x, y)."""
        e = self.e
        if e < 1.0:
            theta_arr = np.linspace(0, 2 * np.pi, 360)
        elif abs(e - 1.0) < 1e-3:
            # Parabola: r = p/(1+cos θ) → limita r a ~25 r_p
            theta_max = 2 * np.arctan(np.sqrt(24))
            theta_arr = np.linspace(-theta_max, theta_max, 400)
        else:
            theta_inf = self._theta_inf()
            theta_arr = np.linspace(-0.985 * theta_inf,
                                    0.985 * theta_inf, 400)
        p = self._semi_latus_rectum()
        r = p / (1 + e * np.cos(theta_arr))
        ang = theta_arr + self.argp
        x = r * np.cos(ang)
        y = r * np.sin(ang)
        return theta_arr, np.column_stack([x, y])

    def _xy_at(self, theta):
        p = self._semi_latus_rectum()
        r = p / (1 + self.e * np.cos(theta))
        ang = theta + self.argp
        return np.array([r * np.cos(ang), r * np.sin(ang)])

    # ==================================================================
    #   Tempo / animazione
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
    #   Disegno
    # ==================================================================
    def _recompute(self):
        self.e = self.spin_e.value()
        self.rp = self.spin_rp.value()
        self.argp = np.radians(self.spin_argp.value())
        self.theta0 = np.radians(self.spin_theta.value())
        self._theta_arr, self._xy_arr = self._build_orbit_points()
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.set_facecolor('#02030a')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(color='#223355', alpha=0.4, linewidth=0.5)
        for s in self.ax.spines.values():
            s.set_color('#335577')
        self.ax.tick_params(colors='#aaccee')

        e = self.e; rp = self.rp
        # Limiti
        if e < 1.0:
            ra_v = rp * (1 + e) / (1 - e) if e > 0 else rp
            max_r = ra_v * 1.15
        else:
            # estensione massima dell'orbita campionata
            max_r = float(np.max(np.linalg.norm(self._xy_arr, axis=1))) * 1.05
        max_r = max(max_r, self.R_p * 2.5)
        self.ax.set_xlim(-max_r, max_r)
        self.ax.set_ylim(-max_r, max_r)

        # Pianeta
        circle = Circle((0, 0), self.R_p,
                        facecolor=PLANETS[self.planet]['color'],
                        edgecolor='white', linewidth=0.8,
                        alpha=0.85, zorder=2)
        self.ax.add_patch(circle)
        self.ax.plot(0, 0, marker='+', color='white', markersize=8, zorder=3)

        # Orbita
        x = self._xy_arr[:, 0]; y = self._xy_arr[:, 1]
        self.ax.plot(x, y, color='#ff7744', linewidth=1.8,
                     alpha=0.95, zorder=4)

        # Asintoti per iperbole
        if e > 1.0 and abs(e - 1.0) > 1e-3:
            self._draw_asymptotes(max_r)

        # Pericentro
        pp = self._xy_at(0.0)
        self.ax.plot(pp[0], pp[1], marker='o', color='#ff8800',
                     markersize=8, markeredgecolor='white', zorder=6)
        self.ax.annotate('  P', (pp[0], pp[1]), color='#ffaa44',
                         fontsize=11, fontweight='bold', zorder=7)

        # Apocentro (solo se ellittica)
        if 1e-3 < e < 1.0:
            ap = self._xy_at(np.pi)
            self.ax.plot(ap[0], ap[1], marker='o', color='#4488ff',
                         markersize=7, markeredgecolor='white', zorder=6)
            self.ax.annotate('  A', (ap[0], ap[1]), color='#88bbff',
                             fontsize=11, fontweight='bold', zorder=7)

        # Asse del pericentro (linea degli apsidi)
        if e < 1.0 and e > 1e-3:
            ap = self._xy_at(np.pi)
            self.ax.plot([pp[0], ap[0]], [pp[1], ap[1]],
                         color='#88aaff', linestyle='--',
                         linewidth=0.6, alpha=0.4, zorder=3)

        # Satellite
        sat_xy = self._xy_at(self._theta_at_time(self.sim_time))
        self._sat_radius_line, = self.ax.plot(
            [0, sat_xy[0]], [0, sat_xy[1]],
            color='#88ccff', linewidth=0.7, alpha=0.5, zorder=3)
        self._sat_marker, = self.ax.plot(
            [sat_xy[0]], [sat_xy[1]],
            marker='o', markersize=11, markerfacecolor='#ffd866',
            markeredgecolor='white', zorder=10)

        # Tooltip (creato dopo clear)
        self.tooltip = self.ax.annotate(
            '', xy=(0, 0), xytext=(12, 12),
            textcoords='offset points', fontsize=9,
            color='white',
            bbox=dict(facecolor='#1a2438', edgecolor='#88aadd',
                      alpha=0.92, boxstyle='round,pad=0.4'),
            visible=False, zorder=20,
        )

        # Titolo / etichette
        self.ax.set_title(
            f'Orbita {self._orbit_type_name()} attorno a {self.planet}',
            color='#aaddff', fontsize=11, pad=6,
        )
        self.ax.set_xlabel('x [km]', color='white')
        self.ax.set_ylabel('y [km]', color='white')

        self._update_info_panel()
        self.canvas.draw_idle()

    def _draw_asymptotes(self, max_r):
        """Disegna gli asintoti dell'orbita iperbolica."""
        e = self.e
        theta_inf = self._theta_inf()
        # Direzioni asintoti nel frame del pericentro
        for sign in (+1, -1):
            th_far = sign * 0.985 * theta_inf
            far = self._xy_at(th_far)
            # direzione asintoto = (cos(th_inf*sign + argp), sin(...))
            ang = sign * theta_inf + self.argp
            d = np.array([np.cos(ang), np.sin(ang)])
            t_vals = np.linspace(-max_r * 1.5, max_r * 1.5, 2)
            xs = far[0] + t_vals * d[0]
            ys = far[1] + t_vals * d[1]
            self.ax.plot(xs, ys, color='#aaaaff',
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
        # Reset per orbite aperte quando il sat si avvicina all'asintoto
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
        if e < 1e-3: return 'circolare'
        if e < 1.0: return 'ellittica'
        if abs(e - 1.0) < 1e-3: return 'parabolica'
        return 'iperbolica'

    def _orbit_props(self, theta):
        e = self.e; mu = self.mu; rp = self.rp
        p = self._semi_latus_rectum()
        if abs(e - 1.0) < 1e-3:
            a = np.inf
        else:
            a = rp / (1 - e) if e < 1 else rp / (1 - e)  # negativo per iperb.
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
        L.append(f"<b style='color:#6ab0ff'>Pianeta: {self.planet}</b><br>")
        L.append(f"<span style='color:#88aadd'>Tipo:</span> "
                 f"<b style='color:#ffd866'>{self._orbit_type_name()}</b><br>")
        L.append(f"<span style='color:#88aadd'>r<sub>p</sub>:</span> "
                 f"<b style='color:#ffd866'>{rp:.1f} km</b> "
                 f"(h={rp - self.R_p:.1f} km)<br>")
        if e < 1.0:
            ra = rp * (1 + e) / (1 - e) if e > 0 else rp
            L.append(f"<span style='color:#88aadd'>r<sub>a</sub>:</span> "
                     f"<b style='color:#ffd866'>{ra:.1f} km</b><br>")
            T = 2 * np.pi * np.sqrt(pr['a'] ** 3 / mu)
            L.append(f"<span style='color:#88aadd'>Periodo T:</span> "
                     f"{T:.1f} s = {T / 60:.2f} min = {T / 3600:.3f} h<br>")
        elif abs(e - 1.0) < 1e-3:
            L.append(f"<span style='color:#88aadd'>r<sub>a</sub>:</span> "
                     f"∞ (parabola)<br>")
            L.append(f"<span style='color:#88aadd'>v fuga al r<sub>p</sub>:</span> "
                     f"{np.sqrt(2 * mu / rp):.3f} km/s<br>")
        else:
            theta_inf = self._theta_inf()
            v_inf = np.sqrt(mu * (e - 1) / rp)
            delta = 2 * np.arcsin(1.0 / e)
            L.append(f"<span style='color:#88aadd'>v<sub>∞</sub>:</span> "
                     f"<b style='color:#ffd866'>{v_inf:.3f} km/s</b><br>")
            L.append(f"<span style='color:#88aadd'>θ<sub>∞</sub>:</span> "
                     f"<b style='color:#ffd866'>±{np.degrees(theta_inf):.2f}°</b><br>")
            L.append(f"<span style='color:#88aadd'>δ (deflessione):</span> "
                     f"{np.degrees(delta):.2f}°<br>")

        L.append(f"<span style='color:#88aadd'>Energia ξ:</span> "
                 f"<b style='color:#ffd866'>{pr['energy']:+.4f} km²/s²</b><br>")
        L.append(f"<span style='color:#88aadd'>h:</span> "
                 f"{pr['h']:.1f} km²/s &nbsp; "
                 f"<span style='color:#88aadd'>p:</span> {pr['p']:.1f} km<br>")
        L.append(f"<br><b style='color:#6ab0ff'>Stato corrente</b><br>")
        L.append(f"<span style='color:#88aadd'>θ:</span> "
                 f"<b style='color:#ffd866'>{np.degrees(th):+.2f}°</b><br>")
        L.append(f"<span style='color:#88aadd'>r:</span> {pr['r']:.1f} km<br>")
        L.append(f"<span style='color:#88aadd'>|v|:</span> "
                 f"<b style='color:#ffd866'>{pr['v']:.3f} km/s</b><br>")
        L.append(f"<span style='color:#88aadd'>v<sub>⊥</sub>:</span> "
                 f"{pr['v_perp']:.3f} km/s &nbsp; "
                 f"<span style='color:#88aadd'>v<sub>r</sub>:</span> "
                 f"{pr['v_r']:.3f} km/s<br>")
        L.append(f"<span style='color:#88aadd'>γ (FPA):</span> "
                 f"<b style='color:#ffd866'>{pr['gamma']:+.3f}°</b><br>")
        self.info_text.setHtml(''.join(L))

    # ==================================================================
    #   Tooltip mouse
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
        txt = (f"θ = {np.degrees(theta):+.2f}°\n"
               f"r = {pr['r']:.1f} km\n"
               f"|v| = {pr['v']:.3f} km/s\n"
               f"v⊥ = {pr['v_perp']:.3f}, v_r = {pr['v_r']:.3f}\n"
               f"γ = {pr['gamma']:+.2f}°")
        self.tooltip.xy = (pt[0], pt[1])
        self.tooltip.set_text(txt)
        self.tooltip.set_visible(True)
        self.canvas.draw_idle()
