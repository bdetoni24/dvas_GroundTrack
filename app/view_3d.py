"""
Widget 3D con Terra texturata, orbita del satellite e visualizzazione
dei 6 parametri Kepleriani (a, e, i, Omega, omega, theta).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
from PIL import Image

from PyQt5.QtWidgets import QWidget, QVBoxLayout

from orbital import (
    keplerian_to_eci, perifocal_to_eci_matrix, eci_to_ecef_matrix,
    R_EARTH, OMEGA_EARTH,
)

EARTH_MAP_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'earth_map.jpg')


class View3D(QWidget):
    """
    Visualizzazione 3D: Terra texturata con asse di rotazione,
    piano equatoriale, punto d'Ariete, piano orbitale, orbita,
    satellite animato e indicatori dei 6 parametri Kepleriani.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Stato orbita corrente
        self.a = 7000.0
        self.e = 0.1
        self.i = np.radians(45.0)
        self.raan = np.radians(0.0)
        self.argp = np.radians(0.0)
        self.theta = 0.0
        self.orbit_color = '#FF4444'
        self.sat_name = 'Satellite'
        self.show_ecef = False            # se True, l'orbita ruota con la Terra
        self.current_time = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 8), facecolor='#02030a')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#02030a')

        # Texture caricata in memoria (lazy)
        self._earth_img = None
        self._load_earth_texture()

        # Handles dinamici
        self._earth_surface = None
        self._earth_rotation = 0.0
        self._orbit_line = None
        self._sat_marker = None
        self._sat_trail = None
        self._annotations = []

        self._setup_view()
        self.redraw()

    # ------------------------------------------------------------------
    def _load_earth_texture(self):
        try:
            img = Image.open(EARTH_MAP_PATH).resize((360, 180))
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
        # Posizione del subplot: full panel, lieve margine
        self.fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)

    # ------------------------------------------------------------------
    def set_orbit(self, a, e, i, raan, argp, theta, color='#FF4444', name='Satellite'):
        """Imposta l'orbita (angoli in radianti) e richiede un redraw completo."""
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
        """
        Aggiorna il tempo. In modalità ECI non serve redraw completo
        (la Terra è statica), solo l'aggiornamento del satellite.
        In modalità ECEF la Terra ruota: rifa redraw solo se la rotazione
        è cambiata di almeno 1 grado per non saturare le risorse.
        """
        self.current_time = t
        new_rot = OMEGA_EARTH * t
        if self.show_ecef:
            if abs(new_rot - self._earth_rotation) > np.radians(5.0):
                self._earth_rotation = new_rot
                self.redraw()
        else:
            self._earth_rotation = new_rot

    def set_current_anomaly(self, theta):
        """Aggiorna solo la posizione del satellite (theta in radianti)."""
        self.theta = theta
        self._update_satellite()
        self.canvas.draw_idle()

    def set_show_ecef(self, flag):
        self.show_ecef = bool(flag)
        self.redraw()

    # ------------------------------------------------------------------
    def _adjust_limits(self):
        """Imposta i limiti degli assi in funzione dell'orbita attuale."""
        if self.e < 1.0:
            r_max = self.a * (1 + self.e) * 1.15
        else:
            r_max = self.a * 3
        r_max = max(r_max, R_EARTH * 2)
        self.ax.set_xlim(-r_max, r_max)
        self.ax.set_ylim(-r_max, r_max)
        self.ax.set_zlim(-r_max, r_max)

    # ------------------------------------------------------------------
    def _draw_earth(self):
        """Disegna la Terra texturata come sfera."""
        Nu, Nv = 60, 30
        u = np.linspace(-np.pi, np.pi, Nu)
        v = np.linspace(0, np.pi, Nv)
        U, V = np.meshgrid(u, v)
        # Longitudine 0 (Greenwich) ruotata di _earth_rotation in ECI
        lon_eff = U + (self._earth_rotation if self.show_ecef else 0.0)
        # Per far ruotare visivamente la Terra, modifichiamo x,y
        X = R_EARTH * np.cos(V) * 0 + R_EARTH * np.sin(V) * np.cos(lon_eff)
        Y = R_EARTH * np.sin(V) * np.sin(lon_eff)
        Z = R_EARTH * np.cos(V)

        if self._earth_img is not None:
            img = self._earth_img
            # Sample texture alla griglia
            # mappatura u [-pi,pi] -> colonna [0,Nu-1], v [0,pi] -> riga [0,Nv-1]
            col = ((U + np.pi) / (2 * np.pi)) * (img.shape[1] - 1)
            row = (V / np.pi) * (img.shape[0] - 1)
            col = col.astype(int)
            row = row.astype(int)
            fc = img[row, col]
            self.ax.plot_surface(
                X, Y, Z,
                facecolors=fc,
                rstride=1, cstride=1,
                linewidth=0, antialiased=False, shade=False,
                zorder=1,
            )
        else:
            self.ax.plot_surface(X, Y, Z, color='#2288ff', alpha=0.6)

    def _draw_equatorial_plane(self):
        """Disegna il piano equatoriale come disco trasparente."""
        r = max(self.a * 1.25, R_EARTH * 1.8)
        N = 64
        t = np.linspace(0, 2 * np.pi, N)
        xs = r * np.cos(t)
        ys = r * np.sin(t)
        zs = np.zeros_like(xs)
        verts = [list(zip(xs, ys, zs))]
        poly = Poly3DCollection(verts, alpha=0.07, facecolor='#66aaff',
                                edgecolor='#335577', linewidth=0.5)
        self.ax.add_collection3d(poly)
        # Linea del punto d'Ariete (asse x ECI)
        self.ax.plot([0, r], [0, 0], [0, 0],
                     color='#ff3366', linewidth=1.3, zorder=3, alpha=0.85)
        self.ax.text(r * 1.05, 0, 0, r' $\gamma$',
                     color='#ff6688', fontsize=11, fontweight='bold')
        # Asse di rotazione terrestre (z)
        z_axis_len = max(R_EARTH * 1.6, self.a * 0.15)
        self.ax.plot([0, 0], [0, 0], [-z_axis_len, z_axis_len],
                     color='#aaffaa', linewidth=1.3, zorder=3, alpha=0.85)
        self.ax.text(0, 0, z_axis_len * 1.05, 'z',
                     color='#88ff88', fontsize=10, fontweight='bold')

    def _draw_orbit(self):
        """
        Disegna l'ellisse orbitale completa, nodo ascendente,
        linea degli apsidi (vettore eccentricità) e pericentro.
        """
        if self.e >= 1.0:
            # orbita aperta: disegna solo un ramo
            theta_arr = np.linspace(-np.radians(150), np.radians(150), 120)
        else:
            theta_arr = np.linspace(0, 2 * np.pi, 120)

        p = self.a * (1 - self.e**2) if self.e < 1.0 else self.a * abs(1 - self.e**2)
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

        self.ax.plot(r_orbit[:, 0], r_orbit[:, 1], r_orbit[:, 2],
                     color=self.orbit_color, linewidth=1.5,
                     alpha=0.9, zorder=4)

        # Pericentro (theta=0) e apocentro (theta=pi)
        rp_vec, _ = keplerian_to_eci(self.a, self.e, self.i, self.raan,
                                     self.argp, 0.0)
        if self.show_ecef:
            rp_vec = eci_to_ecef_matrix(self._earth_rotation) @ rp_vec
        self.ax.plot([rp_vec[0]], [rp_vec[1]], [rp_vec[2]],
                     marker='o', color='#ff8800', markersize=8,
                     markeredgecolor='white', zorder=6)
        self.ax.text(rp_vec[0] * 1.08, rp_vec[1] * 1.08, rp_vec[2] * 1.08 + R_EARTH * 0.15,
                     'P', color='#ffaa44', fontsize=11, fontweight='bold')

        if self.e < 1.0 and self.e > 1e-3:
            ra_vec, _ = keplerian_to_eci(self.a, self.e, self.i, self.raan,
                                         self.argp, np.pi)
            if self.show_ecef:
                ra_vec = eci_to_ecef_matrix(self._earth_rotation) @ ra_vec
            self.ax.plot([ra_vec[0]], [ra_vec[1]], [ra_vec[2]],
                         marker='o', color='#4488ff', markersize=7,
                         markeredgecolor='white', zorder=6)
            self.ax.text(ra_vec[0] * 1.08, ra_vec[1] * 1.08, ra_vec[2] * 1.08 + R_EARTH * 0.15,
                         'A', color='#88bbff', fontsize=11, fontweight='bold')

            # Linea degli apsidi (congiunge peri- e apo-centro)
            self.ax.plot([rp_vec[0], ra_vec[0]],
                         [rp_vec[1], ra_vec[1]],
                         [rp_vec[2], ra_vec[2]],
                         color='#aa6622', linestyle='--', linewidth=0.7,
                         alpha=0.55, zorder=3)

        # Linea dei nodi (vettore N = k x h) nel piano equatoriale, solo se i!=0
        if abs(self.i) > 1e-3:
            N_vec = np.array([np.cos(self.raan), np.sin(self.raan), 0])
            r_N = max(self.a * 1.10, R_EARTH * 1.5)
            Na = N_vec * r_N
            Nd = -N_vec * r_N
            if self.show_ecef:
                Rmat = eci_to_ecef_matrix(self._earth_rotation)
                Na = Rmat @ Na
                Nd = Rmat @ Nd
            self.ax.plot([Nd[0], Na[0]], [Nd[1], Na[1]], [Nd[2], Na[2]],
                         color='#ffff66', linestyle='-', linewidth=0.9,
                         alpha=0.65, zorder=3)
            self.ax.text(Na[0] * 1.04, Na[1] * 1.04, Na[2] * 1.04,
                         'NA', color='#ffff66', fontsize=10, fontweight='bold')
            self.ax.text(Nd[0] * 1.04, Nd[1] * 1.04, Nd[2] * 1.04,
                         'ND', color='#aaaa44', fontsize=9)
        else:
            N_vec = np.array([1.0, 0.0, 0.0])  # placeholder per i=0

        # Arco che visualizza RAAN (Omega)
        self._draw_angle_arc(
            center=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1]),
            start_dir=np.array([1, 0, 0]),    # asse X ECI
            angle=self.raan,
            radius=max(self.a * 0.25, R_EARTH * 1.1),
            color='#ffaa33',
            label=r'$\Omega$',
        )

        # Arco che visualizza inclinazione i (tra piano equatoriale e piano orbitale, all'NA)
        if abs(self.i) > 1e-3:
            h_dir_eci = perifocal_to_eci_matrix(self.raan, self.i, self.argp) @ np.array([0, 0, 1])
            self._draw_angle_arc(
                center=np.array([0, 0, 0]),
                normal=N_vec,
                start_dir=np.array([0, 0, 1]),   # asse Z (asse rotazione)
                angle=self.i,
                radius=max(self.a * 0.18, R_EARTH * 1.05),
                color='#66ffaa',
                label='i',
                rotate_about=N_vec,
            )

        # Arco che visualizza argomento di pericentro omega (nel piano orbitale, dal NA al pericentro)
        if abs(self.argp) > 1e-3:
            T = perifocal_to_eci_matrix(self.raan, self.i, 0.0)
            self._draw_angle_arc_plane(
                T_per=T,
                angle=self.argp,
                radius=max(self.a * 0.3, R_EARTH * 1.2),
                color='#ff88ff',
                label=r'$\omega$',
            )

    def _draw_angle_arc(self, center, normal, start_dir, angle, radius,
                        color='#ffffff', label='', rotate_about=None):
        """
        Disegna un arco che rappresenta un angolo.
        `start_dir` è la direzione iniziale, e l'arco viene disegnato
        ruotando attorno a `normal` di un angolo `angle`.
        """
        if abs(angle) < 1e-3:
            return
        # Crea base ortonormale
        u = start_dir / np.linalg.norm(start_dir)
        # Proietta normal per essere ortogonale a u
        n = normal - np.dot(normal, u) * u
        if np.linalg.norm(n) < 1e-6:
            return
        n = n / np.linalg.norm(n)
        v = np.cross(n, u)
        ts = np.linspace(0, angle, 40)
        pts = np.array([center + radius * (np.cos(t) * u + np.sin(t) * v) for t in ts])
        if self.show_ecef:
            R = eci_to_ecef_matrix(self._earth_rotation)
            pts = pts @ R.T
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=color, linewidth=2.0, alpha=0.9, zorder=4)
        mid = pts[len(pts) // 2]
        self.ax.text(mid[0], mid[1], mid[2], label,
                     color=color, fontsize=11, fontweight='bold')

    def _draw_angle_arc_plane(self, T_per, angle, radius, color, label):
        """Arco nel piano orbitale (usando matrice di rotazione PER->ECI)."""
        if abs(angle) < 1e-3:
            return
        ts = np.linspace(0, angle, 40)
        pts_per = np.array([[radius * np.cos(t), radius * np.sin(t), 0] for t in ts])
        pts = pts_per @ T_per.T
        if self.show_ecef:
            R = eci_to_ecef_matrix(self._earth_rotation)
            pts = pts @ R.T
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=color, linewidth=2.0, alpha=0.9, zorder=4)
        mid = pts[len(pts) // 2]
        self.ax.text(mid[0], mid[1], mid[2], label,
                     color=color, fontsize=11, fontweight='bold')

    def _draw_satellite(self):
        """Disegna il satellite nella sua posizione attuale."""
        r, _ = keplerian_to_eci(self.a, self.e, self.i, self.raan,
                                self.argp, self.theta)
        if self.show_ecef:
            r = eci_to_ecef_matrix(self._earth_rotation) @ r

        # Linea dal centro al satellite (raggio vettore)
        self.ax.plot([0, r[0]], [0, r[1]], [0, r[2]],
                     color='#88ccff', linewidth=0.8, alpha=0.5, zorder=3)
        # Satellite
        (sat,) = self.ax.plot([r[0]], [r[1]], [r[2]],
                              marker='o', markersize=11,
                              markerfacecolor=self.orbit_color,
                              markeredgecolor='white',
                              zorder=10)
        self._sat_marker = sat
        # Etichetta satellite spostata di un offset che dipende dalla scala
        offset = max(R_EARTH * 0.15, np.linalg.norm(r) * 0.05)
        self.ax.text(r[0], r[1], r[2] + offset,
                     self.sat_name, color='white', fontsize=9,
                     ha='center', fontweight='bold')

        # Arco di anomalia vera theta (dal pericentro al satellite)
        if abs(self.theta) > 1e-3:
            T = perifocal_to_eci_matrix(self.raan, self.i, self.argp)
            self._draw_angle_arc_plane(
                T_per=T,
                angle=self.theta,
                radius=max(np.linalg.norm(r) * 0.35, R_EARTH * 0.8),
                color='#00ffff',
                label=r'$\theta$',
            )

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
        """Ridisegna l'intera scena."""
        # Salva il punto di vista corrente
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

        # Titolo info
        mode_txt = 'ECEF (ruotante)' if self.show_ecef else 'ECI (inerziale)'
        self.ax.set_title(
            f'Sistema {mode_txt} - {self.sat_name}',
            color='#aaddff', fontsize=11, pad=6,
        )

        # Ripristina vista
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()
