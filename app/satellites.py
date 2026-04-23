"""
Database di satelliti / orbite di esempio.
I parametri sono valori indicativi per visualizzazione didattica
(non sono TLE realtime, ma rappresentano fedelmente la classe orbitale).
"""
import numpy as np

from orbital import MU_EARTH, R_EARTH, T_SIDEREAL, OMEGA_SUN, J2_EARTH


def _sso_inclination(a, e):
    """
    Calcola l'inclinazione necessaria per un'orbita Sun-Synchronous (SSO)
    a partire da semiasse maggiore ed eccentricità (appunti pag. 24-25).
        Omega_dot = -3/2 * J2 * sqrt(mu)/((1-e^2)^2 * a^(7/2)) * R^2 * cos(i) = OMEGA_SUN
    """
    n = np.sqrt(MU_EARTH / a ** 3)
    p = a * (1 - e ** 2)
    factor = -1.5 * n * J2_EARTH * (R_EARTH / p) ** 2
    cos_i = OMEGA_SUN / factor
    return np.arccos(np.clip(cos_i, -1.0, 1.0))


# ----------------------------------------------------------------------
#  Definizione satelliti (a, e, i, raan, argp, theta) - angoli in gradi
# ----------------------------------------------------------------------
SATELLITES = {
    'ISS (Stazione Spaziale)': {
        'a':     6781.137,        # km (~408 km altitudine)
        'e':     0.0006,
        'i':     51.64,
        'raan':  120.0,
        'argp':  60.0,
        'theta': 0.0,
        'descrizione': 'Stazione Spaziale Internazionale, orbita LEO bassa.',
        'colore': '#FF4444',
    },
    'Hubble Space Telescope': {
        'a':     6918.0,
        'e':     0.0003,
        'i':     28.47,
        'raan':  85.0,
        'argp':  0.0,
        'theta': 0.0,
        'descrizione': 'Telescopio spaziale, orbita LEO equatoriale.',
        'colore': '#44CCFF',
    },
    'GPS (semisincrona MEO)': {
        'a':     26560.0,
        'e':     0.012,
        'i':     55.0,
        'raan':  0.0,
        'argp':  0.0,
        'theta': 0.0,
        'descrizione': 'Costellazione GPS, periodo ~12h, orbita semisincrona.',
        'colore': '#FFAA00',
    },
    'Geostazionaria (GEO)': {
        'a':     42164.0,
        'e':     0.0,
        'i':     0.0,
        'raan':  0.0,
        'argp':  0.0,
        'theta': 0.0,
        'descrizione': 'Orbita geostazionaria: i=0, e=0, T=T_sidereo. Appare ferma in ECEF.',
        'colore': '#FFFF00',
    },
    'Geosincrona inclinata': {
        'a':     42164.0,
        'e':     0.0,
        'i':     30.0,
        'raan':  60.0,
        'argp':  0.0,
        'theta': 0.0,
        'descrizione': 'T=T_sidereo ma i!=0: ground track a forma di "8".',
        'colore': '#FFD700',
    },
    'Molniya (Russia)': {
        'a':     26600.0,
        'e':     0.74,
        'i':     63.4,
        'raan':  90.0,
        'argp':  270.0,           # apogeo sull'emisfero nord
        'theta': 0.0,
        'descrizione': 'Orbita altamente eccentrica, T~12h, i=63.4 (no rotazione apsidi).',
        'colore': '#FF66CC',
    },
    'SSO (Sun-Synchronous, ~700km)': {
        'a':     7078.0,
        'e':     0.001,
        'i':     np.degrees(_sso_inclination(7078.0, 0.001)),
        'raan':  0.0,
        'argp':  0.0,
        'theta': 0.0,
        'descrizione': 'Orbita eliosincrona: passa sempre sopra ogni punto alla stessa ora locale.',
        'colore': '#88FF88',
    },
    'Polare LEO': {
        'a':     7000.0,
        'e':     0.001,
        'i':     90.0,
        'raan':  45.0,
        'argp':  0.0,
        'theta': 0.0,
        'descrizione': 'Orbita polare classica.',
        'colore': '#CC88FF',
    },
    'GTO (Geostationary Transfer)': {
        'a':     24400.0,
        'e':     0.73,
        'i':     27.0,
        'raan':  0.0,
        'argp':  180.0,
        'theta': 0.0,
        'descrizione': "Orbita di trasferimento dal LEO al GEO (Hohmann).",
        'colore': '#FF8800',
    },
    'Tundra': {
        'a':     42164.0,
        'e':     0.27,
        'i':     63.4,
        'raan':  0.0,
        'argp':  270.0,
        'theta': 0.0,
        'descrizione': 'Geosincrona altamente ellittica, ground track a goccia.',
        'colore': '#00FFCC',
    },
}


def get_satellite_params(name):
    """Restituisce un dizionario con i parametri di un satellite (angoli in gradi)."""
    if name not in SATELLITES:
        return None
    return SATELLITES[name].copy()


def list_satellites():
    return list(SATELLITES.keys())
