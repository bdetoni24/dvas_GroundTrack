"""
Ground Track view — self-contained tab.

Supports two frames:
  * ECEF : Earth-fixed (tiene conto della rotazione terrestre ω_E)
  * ECI  : inerziale  (stesso calcolo ma senza rotazione terrestre →
           "ECEF senza ritardo"; mostra ascensione retta / declinazione)

Sulla mappa vengono evidenziati i punti notevoli dell'orbita:
pericentro (P), apocentro (A), nodo ascendente (NA) e nodo discendente (ND).
"""
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox,
    QPushButton, QSlider, QGroupBox, QFormLayout, QTextEdit, QScrollArea,
    QFrame, QCheckBox, QMessageBox, QRadioButton, QButtonGroup,
)

import orbital as orb
from satellites import SATELLITES, list_satellites, get_satellite_params


# ---------------------------------------------------------------------------
def _split_wrapped(lons, lats, threshold=180.0):
    """Spezza la traccia ai salti ±180°."""
    if len(lons) == 0:
        return []
    breaks = np.where(np.abs(np.diff(lons)) > threshold)[0] + 1
    idx = np.concatenate(([0], breaks, [len(lons)]))
    return [(lons[a:b], lats[a:b]) for a, b in zip(idx[:-1], idx[1:]) if b - a > 1]


def _lat_lon_at(theta_target, r_eci_array, theta_arr, r_ecef_array):
    """Trova lat/lon (ECI e ECEF) del punto sull'orbita con true anomaly
    più vicina a theta_target (mod 2π)."""
    if len(theta_arr) == 0:
        return None
    d = np.abs(((theta_arr - theta_target + np.pi) % (2 * np.pi)) - np.pi)
    k = int(np.argmin(d))
    x, y, z = r_eci_array[k]
    r = np.sqrt(x * x + y * y + z * z)
    lat_eci = np.degrees(np.arcsin(z / r))
    lon_eci = np.degrees(np.arctan2(y, x))
    x, y, z = r_ecef_array[k]
    r = np.sqrt(x * x + y * y + z * z)
    lat_ecef = np.degrees(np.arcsin(z / r))
    lon_ecef = np.degrees(np.arctan2(y, x))
    return lat_eci, lon_eci, lat_ecef, lon_ecef


# ===========================================================================
class GroundTrackView(QWidget):
    """Ground Track tab — sidebar + grafico, autonomo."""

    LANDMARKS = [
        # (label, theta_target, marker, color)
        ('P',  0.0,      'o', '#d29922'),          # pericentro
        ('A',  np.pi,    'o', '#58a6ff'),          # apocentro
        ('NA', None,     '^', '#3fb950'),          # nodo ascendente (θ = -ω)
        ('ND', None,     'v', '#f85149'),          # nodo discendente (θ = π-ω)
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = True
        self._track = None
        self.frame = 'ECEF'
        self.current_idx = 0
        self.sim_time = 0.0
        self.speed = 500.0
        self.playing = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------------- Sidebar ----------------
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(10, 10, 10, 10)
        sv.setSpacing(10)

        title = QLabel('Ground Track')
        title.setObjectName('title'); sv.addWidget(title)
        sub = QLabel('Proiezione sub-satellite (ECEF / ECI)')
        sub.setObjectName('subtitle'); sv.addWidget(sub)

        sep = QFrame(); sep.setObjectName('separator'); sep.setFrameShape(QFrame.HLine)
        sv.addWidget(sep)

        # Frame selector
        frame_box = QGroupBox('Reference frame')
        fl = QVBoxLayout(frame_box)
        self.rb_ecef = QRadioButton('ECEF — Earth-fixed (con rotazione ω_E)')
        self.rb_eci  = QRadioButton('ECI — inerziale (ECEF senza ritardo)')
        self.rb_ecef.setChecked(True)
        self._frame_group = QButtonGroup(self)
        self._frame_group.addButton(self.rb_ecef, 0)
        self._frame_group.addButton(self.rb_eci, 1)
        fl.addWidget(self.rb_ecef); fl.addWidget(self.rb_eci)
        sv.addWidget(frame_box)

        # Preset
        pbox = QGroupBox('Satellite preset')
        pl = QVBoxLayout(pbox)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem('— custom —')
        for name in list_satellites():
            self.preset_combo.addItem(name)
        pl.addWidget(self.preset_combo)
        self.preset_desc = QLabel('Choose a preset or enter custom parameters.')
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet('color:#8b949e; font-style:italic; font-size:8pt;')
        pl.addWidget(self.preset_desc)
        sv.addWidget(pbox)

        # --- Parametri del piano orbitale (i, Ω) — riquadro principale in alto
        plane_box = QGroupBox('Orbit plane  —  key parameters')
        plane_box.setStyleSheet(
            'QGroupBox { border:2px solid #58a6ff; color:#58a6ff; '
            'background-color:#0f1a2a; }'
        )
        pf = QFormLayout(plane_box); pf.setLabelAlignment(Qt.AlignRight)
        self.spin_i    = self._mkspin(-180.0, 180.0, 45.0, 2, ' °')
        self.spin_raan = self._mkspin(-360.0, 360.0,  0.0, 2, ' °')
        pf.addRow('i   (inclination):', self.spin_i)
        pf.addRow('Ω   (RAAN):',        self.spin_raan)
        sv.addWidget(plane_box)

        # --- Altri elementi orbitali
        kbox = QGroupBox('Other elements')
        form = QFormLayout(kbox); form.setLabelAlignment(Qt.AlignRight)
        self.spin_rp    = self._mkspin(orb.R_EARTH + 100, 200000, 6778, 1, ' km')
        self.spin_e     = self._mkspin(0.0,  0.99,   0.01, 3, '')
        self.spin_argp  = self._mkspin(-360.0, 360.0, 0.0, 2, ' °')
        self.spin_theta = self._mkspin(-360.0, 360.0, 0.0, 2, ' °')
        form.addRow('r_p (pericenter):',   self.spin_rp)
        form.addRow('e  (eccentricity):',  self.spin_e)
        form.addRow('ω  (arg. perigee):',  self.spin_argp)
        form.addRow('θ₀ (initial TA):',    self.spin_theta)
        self.chk_j2 = QCheckBox('Apply J2 perturbation')
        form.addRow(self.chk_j2)
        self.spin_dur = QDoubleSpinBox()
        self.spin_dur.setRange(0.1, 50.0); self.spin_dur.setValue(3.0)
        self.spin_dur.setSingleStep(0.5); self.spin_dur.setSuffix(' T')
        form.addRow('Duration:', self.spin_dur)
        sv.addWidget(kbox)

        # Animation
        abox = QGroupBox('Animation')
        av = QVBoxLayout(abox)
        row = QHBoxLayout()
        self.btn_play = QPushButton('▶  Play'); self.btn_play.setObjectName('play')
        self.btn_play.setCheckable(True)
        self.btn_reset = QPushButton('⟲  Reset')
        row.addWidget(self.btn_play); row.addWidget(self.btn_reset)
        av.addLayout(row)
        sr = QHBoxLayout()
        sr.addWidget(QLabel('Speed:'))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 5000); self.slider_speed.setValue(500)
        sr.addWidget(self.slider_speed)
        self.lbl_speed = QLabel('500 ×'); self.lbl_speed.setObjectName('value')
        self.lbl_speed.setMinimumWidth(55)
        sr.addWidget(self.lbl_speed)
        av.addLayout(sr)
        sv.addWidget(abox)

        # Info
        ibox = QGroupBox('Orbit info')
        il = QVBoxLayout(ibox); il.setContentsMargins(4, 4, 4, 4)
        self.info_text = QTextEdit(); self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(200)
        il.addWidget(self.info_text)
        sv.addWidget(ibox)

        sv.addStretch(1)
        credits = QLabel('De Toni Bernardo · Da Ros Nicola')
        credits.setObjectName('credits'); credits.setAlignment(Qt.AlignCenter)
        sv.addWidget(credits)

        scroll = QScrollArea()
        scroll.setWidget(side); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320); scroll.setMaximumWidth(400)
        root.addWidget(scroll)

        # ---------------- Plot ----------------
        self.fig = Figure(figsize=(10, 5), facecolor='#0d1117')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        root.addWidget(self.canvas, 1)
        self._setup_axes()

        # artisti dinamici
        self._track_lines = []
        self._sat_marker = None
        self._sat_label = None
        self._landmark_artists = []
        self._info_text = None

        # ---------------- Connections ----------------
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        for sp in (self.spin_rp, self.spin_e, self.spin_i,
                   self.spin_raan, self.spin_argp, self.spin_theta):
            sp.valueChanged.connect(self._on_param_changed)
        self.chk_j2.toggled.connect(lambda _: self._recompute())
        self.spin_dur.valueChanged.connect(lambda _: self._recompute())
        self._frame_group.idClicked.connect(self._on_frame_changed)
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.btn_reset.clicked.connect(self._on_reset)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)

        self.timer = QTimer(); self.timer.setInterval(60)
        self.timer.timeout.connect(self._on_tick)

        self._building = False
        self.preset_combo.setCurrentIndex(1)   # ISS

    # ------------------------------------------------------------------
    def _mkspin(self, low, high, val, dec, suf=''):
        sp = QDoubleSpinBox()
        sp.setRange(low, high); sp.setDecimals(dec); sp.setValue(val)
        sp.setSingleStep(0.01 if dec > 2 else 1.0 if dec == 2 else 10.0)
        if suf: sp.setSuffix(suf)
        sp.setMinimumWidth(110)
        return sp

    # ------------------------------------------------------------------
    def _setup_axes(self):
        self.ax.set_facecolor('#0d1117')
        self.ax.set_xlim(-180, 180); self.ax.set_ylim(-90, 90)
        self.ax.tick_params(colors='#8b949e')
        for s in self.ax.spines.values(): s.set_color('#30363d')
        self.ax.set_xticks(np.arange(-180, 181, 30))
        self.ax.set_yticks(np.arange(-90, 91, 30))
        self.ax.grid(color='#30363d', alpha=0.45, linewidth=0.6)
        self.ax.axhline(0, color='#58a6ff', alpha=0.45, linewidth=0.9)
        self.ax.axvline(0, color='#58a6ff', alpha=0.45, linewidth=0.9)

    def _apply_frame_labels(self):
        if self.frame == 'ECI':
            self.ax.set_xlabel('Right ascension [°]', color='#e6edf3')
            self.ax.set_ylabel('Declination [°]',      color='#e6edf3')
            self.ax.set_title('Ground Track — ECI (inertial, no Earth rotation)',
                              color='#58a6ff', fontsize=11, pad=8)
        else:
            self.ax.set_xlabel('Longitude [°]', color='#e6edf3')
            self.ax.set_ylabel('Latitude [°]',  color='#e6edf3')
            self.ax.set_title('Ground Track — ECEF (Earth-fixed)',
                              color='#58a6ff', fontsize=11, pad=8)

    # ------------------------------------------------------------------
    # Sidebar callbacks
    def _on_preset_changed(self, idx):
        if self._building: return
        if idx <= 0:
            self.preset_desc.setText('Custom parameters.'); return
        p = get_satellite_params(self.preset_combo.itemText(idx))
        if p is None: return
        self.preset_desc.setText(p.get('descrizione', ''))
        self._building = True
        # preset fornisce 'a'; converto in r_p = a(1-e)
        rp = p['a'] * (1 - p['e'])
        self.spin_rp.setValue(rp)
        self.spin_e.setValue(p['e'])
        self.spin_i.setValue(p['i']);     self.spin_raan.setValue(p['raan'])
        self.spin_argp.setValue(p['argp']); self.spin_theta.setValue(p['theta'])
        self._building = False
        self._recompute()

    def _on_param_changed(self, _v):
        if self._building: return
        if self.preset_combo.currentIndex() != 0:
            self._building = True
            self.preset_combo.setCurrentIndex(0)
            self.preset_desc.setText('Custom parameters.')
            self._building = False
        self._recompute()

    def _on_frame_changed(self, idx):
        self.frame = 'ECEF' if idx == 0 else 'ECI'
        self._redraw()

    def _on_play_toggled(self, checked):
        self.playing = checked
        if checked:
            self.btn_play.setText('❚❚  Pause'); self.btn_play.setObjectName('pause')
            self.timer.start()
        else:
            self.btn_play.setText('▶  Play');   self.btn_play.setObjectName('play')
            self.timer.stop()
        self.btn_play.style().polish(self.btn_play)

    def _on_reset(self):
        self.sim_time = 0.0; self.current_idx = 0
        self._update_marker(); self._update_info()
        self.canvas.draw_idle()

    def _on_speed_changed(self, v):
        self.speed = float(v); self.lbl_speed.setText(f'{v} ×')

    # ------------------------------------------------------------------
    def _get_params(self):
        rp = self.spin_rp.value(); e = self.spin_e.value()
        a = rp / (1 - e) if e < 1.0 else rp
        return dict(
            a=a, rp=rp, e=e,
            i=np.radians(self.spin_i.value()),
            raan=np.radians(self.spin_raan.value()),
            argp=np.radians(self.spin_argp.value()),
            theta=np.radians(self.spin_theta.value()),
            j2=self.chk_j2.isChecked(),
            duration_periods=self.spin_dur.value(),
        )

    def _color(self):
        idx = self.preset_combo.currentIndex()
        if idx <= 0: return '#f85149'
        return SATELLITES.get(self.preset_combo.itemText(idx), {}).get('colore', '#f85149')

    def _name(self):
        idx = self.preset_combo.currentIndex()
        return 'Custom' if idx <= 0 else self.preset_combo.itemText(idx)

    # ------------------------------------------------------------------
    def _recompute(self):
        p = self._get_params()
        if p['e'] >= 1.0:
            QMessageBox.warning(self, 'Open orbit',
                                'Ground track only applies to closed orbits (e<1).')
            return
        T = orb.orbital_period(p['a'])
        duration = p['duration_periods'] * T
        t_arr = np.linspace(0, duration, 720)
        r_eci, _, theta_arr, raan_arr, argp_arr = orb.propagate_keplerian(
            p['a'], p['e'], p['i'], p['raan'], p['argp'], p['theta'], t_arr, j2=p['j2'],
        )
        r_ecef = orb.eci_array_to_ecef(r_eci, t_arr, theta_g0=0.0)
        lats_ecef, lons_ecef = orb.compute_groundtrack(r_ecef)
        lats_eci,  lons_eci  = orb.compute_groundtrack(r_eci)

        # Landmark positions su una singola orbita (primo periodo).
        one_period_mask = t_arr <= T
        r_eci_1 = r_eci[one_period_mask]
        r_ecef_1 = r_ecef[one_period_mask]
        theta_1 = theta_arr[one_period_mask]

        argp0 = p['argp']
        landmark_theta = {
            'P':  0.0,
            'A':  np.pi,
            'NA': (-argp0) % (2 * np.pi),
            'ND': (np.pi - argp0) % (2 * np.pi),
        }
        landmark_pts = {}
        for lbl, th in landmark_theta.items():
            pt = _lat_lon_at(th, r_eci_1, theta_1, r_ecef_1)
            if pt is not None:
                landmark_pts[lbl] = pt   # (lat_eci, lon_eci, lat_ecef, lon_ecef)

        self._track = dict(
            a=p['a'], e=p['e'], i=p['i'],
            theta_arr=theta_arr, raan_arr=raan_arr, argp_arr=argp_arr,
            t_arr=t_arr, T=T, duration=duration,
            lats_ecef=lats_ecef, lons_ecef=lons_ecef,
            lats_eci=lats_eci,   lons_eci=lons_eci,
            landmark_pts=landmark_pts,
        )
        self.sim_time = 0.0; self.current_idx = 0
        self._redraw()

    # ------------------------------------------------------------------
    def _redraw(self):
        for ln in self._track_lines: ln.remove()
        self._track_lines = []
        for art in self._landmark_artists:
            try: art.remove()
            except Exception: pass
        self._landmark_artists = []
        if self._sat_marker is not None: self._sat_marker.remove(); self._sat_marker = None
        if self._sat_label  is not None: self._sat_label.remove();  self._sat_label  = None
        if self._info_text  is not None:
            try: self._info_text.remove()
            except Exception: pass
            self._info_text = None

        if self._track is None:
            self._apply_frame_labels(); self.canvas.draw_idle(); return

        lats, lons = self._cur_track()
        color = self._color()

        # traccia + frecce direzionali
        segments = _split_wrapped(lons, lats)
        for lon_seg, lat_seg in segments:
            (ln,) = self.ax.plot(lon_seg, lat_seg, color=color,
                                 linewidth=1.4, alpha=0.95, zorder=3)
            self._track_lines.append(ln)
        self._draw_direction_arrows(segments, color)

        # landmarks + lat max/min + prograde/retrograde badge
        self._draw_landmarks()
        self._draw_lat_limits()
        self._draw_motion_badge()

        # posizione corrente del satellite
        sat_lon, sat_lat = lons[self.current_idx], lats[self.current_idx]
        (sat,) = self.ax.plot(sat_lon, sat_lat, marker='o', markersize=10,
                              markerfacecolor=color, markeredgecolor='white',
                              markeredgewidth=1.5, zorder=6)
        self._sat_marker = sat
        self._sat_label = self.ax.annotate(
            self._name(), xy=(sat_lon, sat_lat),
            xytext=(8, 8), textcoords='offset points',
            color='#e6edf3', fontsize=9,
            bbox=dict(facecolor='#161b22', edgecolor=color,
                      alpha=0.85, boxstyle='round,pad=0.3'),
            zorder=7,
        )

        self._apply_frame_labels()
        self._draw_frame_note()
        self._update_info()
        self.canvas.draw_idle()

    def _draw_landmarks(self):
        pts = self._track['landmark_pts']
        for label, theta_target, marker, col in self.LANDMARKS:
            if label not in pts:
                continue
            lat_eci, lon_eci, lat_ecef, lon_ecef = pts[label]
            if self.frame == 'ECI':
                lat, lon = lat_eci, lon_eci
            else:
                lat, lon = lat_ecef, lon_ecef
            (art,) = self.ax.plot(lon, lat, marker=marker, markersize=10,
                                  markerfacecolor=col, markeredgecolor='white',
                                  markeredgewidth=1.2, linestyle='none', zorder=5)
            self._landmark_artists.append(art)
            ann = self.ax.annotate(
                label, xy=(lon, lat), xytext=(6, 6),
                textcoords='offset points', color=col,
                fontsize=9, fontweight='bold', zorder=5,
            )
            self._landmark_artists.append(ann)

    def _draw_direction_arrows(self, segments, color, n_per_seg=3):
        """Mette frecce lungo la traccia per indicarne il verso di percorrenza."""
        for lon_seg, lat_seg in segments:
            L = len(lon_seg)
            if L < 5:
                continue
            # frecce a posizioni equispaziate nel segmento
            positions = np.linspace(L * 0.2, L * 0.8, n_per_seg).astype(int)
            for k in positions:
                k = int(np.clip(k, 1, L - 2))
                x0, y0 = lon_seg[k - 1], lat_seg[k - 1]
                x1, y1 = lon_seg[k + 1], lat_seg[k + 1]
                ann = self.ax.annotate(
                    '', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=color,
                                    lw=1.4, mutation_scale=14,
                                    shrinkA=0, shrinkB=0, alpha=0.95),
                    zorder=4,
                )
                self._track_lines.append(ann)

    def _draw_motion_badge(self):
        """Box in alto a destra: prograda / polare / retrograda."""
        i_deg = np.degrees(self._track['i'])
        # normalizzo in [0, 180] per la classificazione
        i_mod = abs(((i_deg + 180) % 360) - 180)
        if abs(i_mod - 90.0) < 1.0:
            kind, col = 'POLAR', '#d29922'
        elif i_mod < 90.0:
            kind, col = 'PROGRADE', '#3fb950'
        else:
            kind, col = 'RETROGRADE', '#f85149'
        txt = f'{kind}   i = {i_deg:+.2f}°'
        art = self.ax.text(
            178, 84, txt, color=col, fontsize=10, fontweight='bold',
            ha='right', va='top',
            bbox=dict(facecolor='#0d1117', edgecolor=col,
                      linewidth=1.5, boxstyle='round,pad=0.4'),
            zorder=8,
        )
        self._landmark_artists.append(art)

    def _draw_lat_limits(self):
        """Linee orizzontali a ±lat_max (= min(i, 180°-i) in gradi)."""
        i_deg = np.degrees(self._track['i'])
        lat_max = min(i_deg, 180.0 - i_deg)
        for y, label in ((lat_max, f'lat_max = +{lat_max:.2f}°'),
                         (-lat_max, f'lat_min = −{lat_max:.2f}°')):
            ln = self.ax.axhline(y, color='#e3b341', linestyle='--',
                                 linewidth=0.9, alpha=0.65, zorder=2)
            self._landmark_artists.append(ln)
            ann = self.ax.annotate(
                label, xy=(178, y), xytext=(-4, 4 if y > 0 else -10),
                textcoords='offset points', ha='right',
                color='#e3b341', fontsize=8, fontweight='bold', zorder=5,
            )
            self._landmark_artists.append(ann)

    def _draw_frame_note(self):
        if self._track is None: return
        if self.frame == 'ECEF':
            T = self._track['T']
            dlam = -np.degrees(orb.OMEGA_EARTH * T)
            dlam = ((dlam + 180) % 360) - 180
            km_eq = abs(dlam) * 111.32
            txt = f'Δλ = {dlam:+.2f}° / orbit  (≈ {km_eq:.0f} km at equator)'
        else:
            txt = 'ECI = ECEF senza rotazione terrestre (ω_E = 0)'
        self._info_text = self.ax.text(
            -178, 84, txt, color='#e3b341', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='#0d1117', edgecolor='#30363d',
                      alpha=0.9, boxstyle='round,pad=0.3'), zorder=8,
        )

    def _cur_track(self):
        t = self._track
        if self.frame == 'ECI':
            return t['lats_eci'], t['lons_eci']
        return t['lats_ecef'], t['lons_ecef']

    def _update_marker(self):
        if self._track is None or self._sat_marker is None: return
        lats, lons = self._cur_track()
        lat = lats[self.current_idx]; lon = lons[self.current_idx]
        self._sat_marker.set_data([lon], [lat])
        if self._sat_label is not None:
            self._sat_label.xy = (lon, lat)

    # ------------------------------------------------------------------
    def _on_tick(self):
        if self._track is None: return
        dt = self.timer.interval() / 1000.0
        self.sim_time = (self.sim_time + dt * self.speed) % self._track['duration']
        self.current_idx = min(
            int(np.searchsorted(self._track['t_arr'], self.sim_time)),
            len(self._track['t_arr']) - 1,
        )
        self._update_marker()
        if not hasattr(self, '_tick_i'): self._tick_i = 0
        self._tick_i += 1
        if self._tick_i >= 5:
            self._tick_i = 0; self._update_info()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _update_info(self):
        t = self._track
        if t is None: return
        k = self.current_idx
        info = orb.orbital_info(
            t['a'], t['e'], t['i'],
            t['raan_arr'][k], t['argp_arr'][k], t['theta_arr'][k],
        )
        L = []
        L.append("<b style='color:#58a6ff'>Orbit</b><br>")
        L.append(f"<span style='color:#8b949e'>Type:</span> "
                 f"<b style='color:#e3b341'>{info['tipo_orbita']}</b><br>")
        L.append(f"<span style='color:#8b949e'>a:</span> "
                 f"<b style='color:#e3b341'>{t['a']:.1f} km</b><br>")
        L.append(f"<span style='color:#8b949e'>T:</span> "
                 f"<b style='color:#e3b341'>{info['periodo']/60:.2f} min "
                 f"({info['periodo']/3600:.3f} h)</b><br>")
        L.append(f"<span style='color:#8b949e'>r_p:</span> {info['pericentro']:.1f} km &nbsp;"
                 f"<span style='color:#8b949e'>r_a:</span> {info['apocentro']:.1f} km<br>")
        L.append(f"<span style='color:#8b949e'>h:</span> {info['momento_angolare']:.0f} km²/s "
                 f"&nbsp;<span style='color:#8b949e'>ξ:</span> {info['energia_specifica']:.3f} km²/s²<br>")

        # Landmarks (lat / lon nel frame corrente)
        L.append("<br><b style='color:#58a6ff'>Landmarks</b> "
                 f"<span style='color:#8b949e'>({self.frame})</span><br>")
        for lbl, _, _, col in self.LANDMARKS:
            if lbl not in t['landmark_pts']: continue
            lat_eci, lon_eci, lat_ecef, lon_ecef = t['landmark_pts'][lbl]
            lat, lon = (lat_eci, lon_eci) if self.frame == 'ECI' else (lat_ecef, lon_ecef)
            L.append(f"<span style='color:{col}'>● {lbl}:</span> "
                     f"lat {lat:+.2f}°, lon {lon:+.2f}°<br>")

        L.append("<br><b style='color:#58a6ff'>Current state</b><br>")
        L.append(f"<span style='color:#8b949e'>t:</span> "
                 f"<b style='color:#e3b341'>{self.sim_time/60:.2f} min</b><br>")
        L.append(f"<span style='color:#8b949e'>Alt:</span> "
                 f"<b style='color:#e3b341'>{info['altitudine']:.1f} km</b><br>")
        L.append(f"<span style='color:#8b949e'>|v|:</span> "
                 f"<b style='color:#e3b341'>{info['velocita_corrente']:.3f} km/s</b><br>")
        L.append(f"<span style='color:#8b949e'>γ:</span> "
                 f"<b style='color:#e3b341'>{info['flight_path_angle']:+.3f}°</b><br>")
        self.info_text.setHtml(''.join(L))
