"""
Widget 3D con Terra texturata, orbita del satellite e visualizzazione
dei 6 parametri Kepleriani (a, e, i, Omega, omega, theta).

Ottimizzazioni:
  - Mesh Terra a bassa risoluzione e disegnata con colore uniforme in ECI
  - Redraw completo solo quando cambiano parametri orbitali o scatta
    la soglia di rotazione ECEF; altrimenti si aggiorna solo il marker.
  - Occlusione orbita: i tratti dietro al pianeta sono sbiaditi.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from PyQt5.QtWidgets import QWidget, QVBoxLayout

from orbital import (
    keplerian_to_eci, perifocal_to_eci_matrix, eci_to_ecef_matrix,
    R_EARTH, OMEGA_EARTH,
)

EARTH_MAP_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'earth_map.jpg')


class View3D(QWidget):
    """Visualizzazione 3D dell'orbita."""

    # Soglia minima di rotazione (in radianti) per ritracciare la Terra in ECEF
    ECEF_REDRAW_THRESHOLD = np.radians(12.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.a = 7000.0
        self.e = 0.1
        self.i = np.radians(45.0)
        self.raan = np.radians(0.0)
        self.argp = np.radians(0.0)
        self.theta = 0.0
        self.orbit_color = '#FF4444'
        self.sat_name = 'Satellite'
        self.show_ecef = False
        self.current_time = 0.0
        self._earth_rotation = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 8), facecolor='#02030a')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#02030a')

        self._earth_img = None
        self._earth_facecolors = None
        self._load_earth_texture()

        self._sat_marker = None

        self._setup_view()
        self.redraw()

    # ------------------------------------------------------------------
    def _load_earth_texture(self):
        try:
            img = Image.open(EARTH_MAP_PATH).resize((180, 90))
            self._earth_img = np.array(img) / 255.0
        except Exception as exc:
            print(f'[View3D] Texture Terra non caricata: {exc}')
            self._earth_img = None

    def _setup_view(self):
        self.ax.grid(False)
        self.ax.set_axis_off()
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        self.fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)

    # ------------------------------------------------------------------
    def set_orbit(self, a, e, i, raan, argp, theta, color='#FF4444', name='Satellite'):
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.argp = argp
        self.theta = theta
        self.orbit_color = color
        self.sat_name = name
        self._adjust_limits()
        self.redraw()

    def set_time(self, t):
        self.current_time = t
        if self.show_ecef:
            new_rot = OMEGA_EARTH * t
            if abs(new_rot - self._earth_rotation) > self.ECEF_REDRAW_THRESHOLD:
                self._earth_rotation = new_rot
                self.redraw()

    def set_current_anomaly(self, theta):
        self.theta = theta
        self._update_satellite()
        self.canvas.draw_idle()

    def set_show_ecef(self, flag):
        self.show_ecef = bool(flag)
        self._earth_rotation = OMEGA_EARTH * self.current_time if flag else 0.0
        self.redraw()

    # ------------------------------------------------------------------
    def _adjust_limits(self):
        if self.e < 1.0:
            r_max = self.a * (1 + self.e) * 1.15
        else:
            r_max = self.a * 3
        r_max = max(r_max, R_EARTH * 2)
        self.ax.set_xlim(-r_max, r_max)
        self.ax.set_ylim(-r_max, r_max)
        self.ax.set_zlim(-r_max, r_max)

    def _view_direction(self):
        """Versore (approx.) che punta dall'origine verso la camera."""
        el = np.radians(self.ax.elev if self.ax.elev is not None else 25)
        az = np.radians(self.ax.azim if self.ax.azim is not None else -60)
        return np.array([np.cos(el) * np.cos(az),
                         np.cos(el) * np.sin(az),
                         np.sin(el)])

    def _occlusion_mask(self, pts):
        """
        True dove il punto è visibile (davanti alla sfera) o fuori silhouette;
        False se nascosto dalla sfera.
        """
        view = self._view_direction()
        dots = pts @ view
        perp = np.linalg.norm(pts - dots[:, None] * view, axis=1)
        # nascosto se proiezione dietro il centro E dentro silhouette
        hidden = (dots < 0) & (perp < R_EARTH)
        return ~hidden

    # ------------------------------------------------------------------
    def _draw_earth(self):
        Nu, Nv = 32, 18
        u = np.linspace(-np.pi, np.pi, Nu)
        v = np.linspace(0, np.pi, Nv)
        U, V = np.meshgrid(u, v)
        lon_eff = U + (self._earth_rotation if self.show_ecef else 0.0)
        X = R_EARTH * np.sin(V) * np.cos(lon_eff)
        Y = R_EARTH * np.sin(V) * np.sin(lon_eff)
        Z = R_EARTH * np.cos(V)

        if self._earth_img is not None:
            img = self._earth_img
            col = ((U + np.pi) / (2 * np.pi)) * (img.shape[1] - 1)
            row = (V / np.pi) * (img.shape[0] - 1)
            fc = img[row.astype(int), col.astype(int)]
            self.ax.plot_surface(
                X, Y, Z,
                facecolors=fc,
                rstride=1, cstride=1,
                linewidth=0, antialiased=False, shade=False,
                zorder=1,
            )
        else:
            self.ax.plot_surface(X, Y, Z, color='#2288ff', alpha=0.75,
                                 linewidth=0, antialiased=False)

    def _draw_equatorial_plane(self):
        r = max(self.a * 1.2, R_EARTH * 1.8)
        N = 48
        t = np.linspace(0, 2 * np.pi, N)
        xs = r * np.cos(t); ys = r * np.sin(t); zs = np.zeros_like(xs)
        verts = [list(zip(xs, ys, zs))]
        poly = Poly3DCollection(verts, alpha=0.06, facecolor='#66aaff',
                                edgecolor='#335577', linewidth=0.5)
        self.ax.add_collection3d(poly)
        # Punto d'Ariete
        self.ax.plot([0, r], [0, 0], [0, 0],
                     color='#ff3366', linewidth=1.2, zorder=3, alpha=0.85)
        self.ax.text(r * 1.05, 0, 0, r' $\gamma$',
                     color='#ff6688', fontsize=11, fontweight='bold')
        # Asse z
        z_axis_len = max(R_EARTH * 1.5, self.a * 0.12)
        self.ax.plot([0, 0], [0, 0], [-z_axis_len, z_axis_len],
                     color='#aaffaa', linewidth=1.2, zorder=3, alpha=0.8)

    def _plot_with_occlusion(self, pts, color, linewidth=1.5, zorder=4):
        """Traccia una polilinea 3D sbiadita nei tratti dietro il pianeta."""
        if len(pts) < 2:
            return
        visible = self._occlusion_mask(pts)
        # Segmenti consecutivi con stessa visibilità
        changes = np.where(np.diff(visible.astype(int)) != 0)[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(pts)]))
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            seg = pts[s:e]
            vis = visible[s]
            if vis:
                self.ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                             color=color, linewidth=linewidth,
                             alpha=0.95, zorder=zorder)
            else:
                self.ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                             color=color, linewidth=linewidth * 0.7,
                             linestyle=(0, (3, 3)),
                             alpha=0.25, zorder=zorder - 1)

    def _draw_orbit(self):
        if self.e >= 1.0:
            theta_arr = np.linspace(-np.radians(140), np.radians(140), 90)
        else:
            theta_arr = np.linspace(0, 2 * np.pi, 120)

        p = self.a * (1 - self.e ** 2) if self.e < 1.0 else self.a * abs(1 - self.e ** 2)
        r_mag = p / (1 + self.e * np.cos(theta_arr))
        pts_per = np.column_stack([
            r_mag * np.cos(theta_arr),
            r_mag * np.sin(theta_arr),
            np.zeros_like(theta_arr),
        ])
        T_per = perifocal_to_eci_matrix(self.raan, self.i, self.argp)
        r_orbit = pts_per @ T_per.T
        if self.show_ecef:
            r_orbit = r_orbit @ eci_to_ecef_matrix(self._earth_rotation).T

        self._plot_with_occlusion(r_orbit, self.orbit_color,
                                  linewidth=1.6, zorder=4)

        # Pericentro e apocentro
        rp_vec, _ = keplerian_to_eci(self.a, self.e, self.i, self.raan,
                                     self.argp, 0.0)
        if self.show_ecef:
            rp_vec = eci_to_ecef_matrix(self._earth_rotation) @ rp_vec
        self.ax.plot([rp_vec[0]], [rp_vec[1]], [rp_vec[2]],
                     marker='o', color='#ff8800', markersize=7,
                     markeredgecolor='white', zorder=6)
        self.ax.text(rp_vec[0] * 1.08, rp_vec[1] * 1.08,
                     rp_vec[2] * 1.08 + R_EARTH * 0.15,
                     'P', color='#ffaa44', fontsize=10, fontweight='bold')

        if self.e < 1.0 and self.e > 1e-3:
            ra_vec, _ = keplerian_to_eci(self.a, self.e, self.i, self.raan,
                                         self.argp, np.pi)
            if self.show_ecef:
                ra_vec = eci_to_ecef_matrix(self._earth_rotation) @ ra_vec
            self.ax.plot([ra_vec[0]], [ra_vec[1]], [ra_vec[2]],
                         marker='o', color='#4488ff', markersize=6,
                         markeredgecolor='white', zorder=6)
            self.ax.text(ra_vec[0] * 1.08, ra_vec[1] * 1.08,
                         ra_vec[2] * 1.08 + R_EARTH * 0.15,
                         'A', color='#88bbff', fontsize=10, fontweight='bold')

        # Linea dei nodi
        if abs(self.i) > 1e-3:
            N_vec = np.array([np.cos(self.raan), np.sin(self.raan), 0])
            r_N = max(self.a * 1.05, R_EARTH * 1.5)
            Na = N_vec * r_N; Nd = -N_vec * r_N
            if self.show_ecef:
                Rmat = eci_to_ecef_matrix(self._earth_rotation)
                Na = Rmat @ Na; Nd = Rmat @ Nd
            self.ax.plot([Nd[0], Na[0]], [Nd[1], Na[1]], [Nd[2], Na[2]],
                         color='#ffff66', linewidth=0.8, alpha=0.55, zorder=3)

    def _draw_satellite(self):
        r, _ = keplerian_to_eci(self.a, self.e, self.i, self.raan,
                                self.argp, self.theta)
        if self.show_ecef:
            r = eci_to_ecef_matrix(self._earth_rotation) @ r

        (sat,) = self.ax.plot([r[0]], [r[1]], [r[2]],
                              marker='o', markersize=10,
                              markerfacecolor=self.orbit_color,
                              markeredgecolor='white',
                              zorder=10)
        self._sat_marker = sat
        offset = max(R_EARTH * 0.15, np.linalg.norm(r) * 0.05)
        self.ax.text(r[0], r[1], r[2] + offset,
                     self.sat_name, color='white', fontsize=9,
                     ha='center', fontweight='bold')

    def _update_satellite(self):
        if self._sat_marker is None:
            return
        r, _ = keplerian_to_eci(self.a, self.e, self.i, self.raan,
                                self.argp, self.theta)
        if self.show_ecef:
            r = eci_to_ecef_matrix(self._earth_rotation) @ r
        self._sat_marker.set_data_3d([r[0]], [r[1]], [r[2]])

    # ------------------------------------------------------------------
    def redraw(self):
        try:
            elev = self.ax.elev
            azim = self.ax.azim
        except Exception:
            elev, azim = 25, 45

        self.ax.clear()
        self._setup_view()
        self._adjust_limits()
        self._draw_earth()
        self._draw_equatorial_plane()
        self._draw_orbit()
        self._draw_satellite()

        mode_txt = 'ECEF (ruotante)' if self.show_ecef else 'ECI (inerziale)'
        self.ax.set_title(
            f'Sistema {mode_txt} - {self.sat_name}',
            color='#aaddff', fontsize=11, pad=6,
        )

        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()
