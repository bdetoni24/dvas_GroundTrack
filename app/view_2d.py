"""
Widget 2D per il Ground Track (ECI ed ECEF) con mappa della Terra.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PIL import Image

from PyQt5.QtWidgets import QWidget, QVBoxLayout


EARTH_MAP_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'earth_map.jpg')


def _split_wrapped(lons, lats, threshold=180.0):
    """
    Spezza la linea del ground track quando la longitudine "salta"
    da +180 a -180 o viceversa, per evitare traversate della mappa.
    Restituisce lista di segmenti (lon_seg, lat_seg).
    """
    if len(lons) == 0:
        return []
    dl = np.abs(np.diff(lons))
    breaks = np.where(dl > threshold)[0] + 1
    indices = np.concatenate(([0], breaks, [len(lons)]))
    segments = []
    for a, b in zip(indices[:-1], indices[1:]):
        if b - a > 1:
            segments.append((lons[a:b], lats[a:b]))
    return segments


class GroundTrackView(QWidget):
    """
    Widget con mappa della Terra + ground track.
    Ha due modalità: ECI (su coordinate celesti: declinazione/ascensione retta)
    ed ECEF (su latitudine/longitudine terrestre).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = 'ECEF'                # 'ECI' o 'ECEF'
        self.lats = np.array([])
        self.lons = np.array([])
        self.lats_eci = np.array([])
        self.lons_eci = np.array([])
        self.track_color = '#FF4444'
        self.sat_name = ''
        self.current_idx = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 5), facecolor='#0a0a14')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self._setup_axes()

        # Oggetti grafici riutilizzati durante l'animazione
        self._bg_image = None
        self._track_lines = []
        self._sat_marker = None
        self._sat_label = None
        self._orbit_info = []

        self._load_background()

    # ------------------------------------------------------------------
    def _setup_axes(self):
        self.ax.set_facecolor('#0a0a14')
        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)
        self.ax.set_xlabel('Longitudine [°]', color='white')
        self.ax.set_ylabel('Latitudine [°]', color='white')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('#444466')

        # Griglia
        self.ax.set_xticks(np.arange(-180, 181, 30))
        self.ax.set_yticks(np.arange(-90, 91, 30))
        self.ax.grid(color='#335577', alpha=0.35, linewidth=0.5)

        # Equatore e greenwich in evidenza
        self.ax.axhline(0, color='#66ddff', alpha=0.5, linewidth=0.8)
        self.ax.axvline(0, color='#66ddff', alpha=0.5, linewidth=0.8)

    def _load_background(self):
        """Carica la texture terrestre come sfondo."""
        try:
            img = Image.open(EARTH_MAP_PATH)
            img_arr = np.array(img)
            self._bg_image = self.ax.imshow(
                img_arr,
                extent=[-180, 180, -90, 90],
                aspect='auto',
                alpha=0.85,
                zorder=0,
            )
        except Exception as exc:
            print(f'[GroundTrackView] Texture non caricata: {exc}')

    # ------------------------------------------------------------------
    def set_frame(self, frame):
        assert frame in ('ECI', 'ECEF')
        self.frame = frame
        self._apply_frame_labels()
        self._redraw()

    def _apply_frame_labels(self):
        if self.frame == 'ECI':
            self.ax.set_xlabel('Ascensione retta [°]', color='white')
            self.ax.set_ylabel('Declinazione [°]', color='white')
            self.ax.set_title('Ground Track - Sistema ECI (inerziale)',
                              color='#aaddff', fontsize=11, pad=8)
        else:
            self.ax.set_xlabel('Longitudine [°]', color='white')
            self.ax.set_ylabel('Latitudine [°]', color='white')
            self.ax.set_title('Ground Track - Sistema ECEF (terrestre)',
                              color='#aaddff', fontsize=11, pad=8)

    # ------------------------------------------------------------------
    def set_orbit_data(self, lats_ecef, lons_ecef, lats_eci, lons_eci,
                       color='#FF4444', name='Satellite'):
        """Imposta i dati dell'orbita da visualizzare."""
        self.lats = np.asarray(lats_ecef)
        self.lons = np.asarray(lons_ecef)
        self.lats_eci = np.asarray(lats_eci)
        self.lons_eci = np.asarray(lons_eci)
        self.track_color = color
        self.sat_name = name
        self.current_idx = 0
        self._redraw()

    def set_current_index(self, idx):
        """Aggiorna la posizione corrente del satellite."""
        lats = self.lats_eci if self.frame == 'ECI' else self.lats
        if len(lats) == 0:
            return
        self.current_idx = int(idx) % len(lats)
        self._update_marker()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _redraw(self):
        """Ridisegna completamente il ground track."""
        # Rimuovi vecchie linee
        for ln in self._track_lines:
            ln.remove()
        self._track_lines = []
        if self._sat_marker is not None:
            self._sat_marker.remove()
            self._sat_marker = None
        if self._sat_label is not None:
            self._sat_label.remove()
            self._sat_label = None

        lats = self.lats_eci if self.frame == 'ECI' else self.lats
        lons = self.lons_eci if self.frame == 'ECI' else self.lons
        if len(lats) == 0:
            self.canvas.draw_idle()
            return

        # Spezza in segmenti per evitare wrap
        segments = _split_wrapped(lons, lats)
        for lon_seg, lat_seg in segments:
            (ln,) = self.ax.plot(lon_seg, lat_seg,
                                 color=self.track_color,
                                 linewidth=1.4, alpha=0.95,
                                 zorder=2)
            self._track_lines.append(ln)

        # Primo punto (t0)
        (start,) = self.ax.plot(lons[0], lats[0], 'o',
                                markersize=6,
                                color='#22ff22',
                                markeredgecolor='white',
                                zorder=4)
        self._track_lines.append(start)

        # Marker del satellite (posizione corrente)
        (sat,) = self.ax.plot(lons[self.current_idx], lats[self.current_idx],
                              marker='o', markersize=10,
                              markerfacecolor=self.track_color,
                              markeredgecolor='white',
                              markeredgewidth=1.5,
                              zorder=5)
        self._sat_marker = sat
        self._sat_label = self.ax.annotate(
            self.sat_name,
            xy=(lons[self.current_idx], lats[self.current_idx]),
            xytext=(8, 8), textcoords='offset points',
            color='white', fontsize=9,
            bbox=dict(facecolor='#222244', edgecolor=self.track_color,
                      alpha=0.85, boxstyle='round,pad=0.3'),
            zorder=6,
        )

        self._apply_frame_labels()
        self.canvas.draw_idle()

    def _update_marker(self):
        lats = self.lats_eci if self.frame == 'ECI' else self.lats
        lons = self.lons_eci if self.frame == 'ECI' else self.lons
        if len(lats) == 0 or self._sat_marker is None:
            return
        self._sat_marker.set_data([lons[self.current_idx]],
                                  [lats[self.current_idx]])
        if self._sat_label is not None:
            self._sat_label.xy = (lons[self.current_idx], lats[self.current_idx])

    # ------------------------------------------------------------------
    def clear(self):
        self.lats = np.array([])
        self.lons = np.array([])
        self.lats_eci = np.array([])
        self.lons_eci = np.array([])
        self._redraw()
