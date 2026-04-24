"""
Library of example satellites / reference orbits for the simulator.
Values are representative for educational visualization (not live TLEs),
but correctly reproduce each orbital class.
"""
import numpy as np

from orbital import MU_EARTH, R_EARTH, OMEGA_SUN, J2_EARTH


def _sso_inclination(a, e):
    """
    Inclination that makes an orbit Sun-synchronous given (a, e):
        Ω_dot = -3/2 · J2 · √μ / ((1-e²)² · a^(7/2)) · R² · cos(i)  =  Ω_sun
    """
    n = np.sqrt(MU_EARTH / a ** 3)
    p = a * (1 - e ** 2)
    factor = -1.5 * n * J2_EARTH * (R_EARTH / p) ** 2
    cos_i = OMEGA_SUN / factor
    return np.arccos(np.clip(cos_i, -1.0, 1.0))


# ----------------------------------------------------------------------
#   Satellite presets (angles in degrees)
# ----------------------------------------------------------------------
SATELLITES = {
    'ISS (International Space Station)': {
        'a': 6781.137, 'e': 0.0006,
        'i': 51.64, 'raan': 120.0, 'argp': 60.0, 'theta': 0.0,
        'descrizione': 'Low Earth Orbit (~408 km altitude) hosting the ISS.',
        'colore': '#f85149',
    },
    'Hubble Space Telescope': {
        'a': 6918.0, 'e': 0.0003,
        'i': 28.47, 'raan': 85.0, 'argp': 0.0, 'theta': 0.0,
        'descrizione': 'Space telescope in low, near-equatorial orbit.',
        'colore': '#56d4dd',
    },
    'GPS (semi-synchronous MEO)': {
        'a': 26560.0, 'e': 0.012,
        'i': 55.0, 'raan': 0.0, 'argp': 0.0, 'theta': 0.0,
        'descrizione': 'GPS constellation: ~12 h period, MEO.',
        'colore': '#d29922',
    },
    'Geostationary (GEO)': {
        'a': 42164.0, 'e': 0.0,
        'i': 0.0, 'raan': 0.0, 'argp': 0.0, 'theta': 0.0,
        'descrizione': 'i=0, e=0, T = sidereal day → fixed in ECEF.',
        'colore': '#e3b341',
    },
    'Inclined geosynchronous': {
        'a': 42164.0, 'e': 0.0,
        'i': 30.0, 'raan': 60.0, 'argp': 0.0, 'theta': 0.0,
        'descrizione': 'T = sidereal but i≠0 → figure-8 ground track.',
        'colore': '#f0c674',
    },
    'Molniya (Russia)': {
        'a': 26600.0, 'e': 0.74,
        'i': 63.4, 'raan': 90.0, 'argp': 270.0, 'theta': 0.0,
        'descrizione': 'Highly elliptic, T~12 h, critical i=63.4° (frozen apsides).',
        'colore': '#ff7ac6',
    },
    'SSO (Sun-Synchronous, ~700 km)': {
        'a': 7078.0, 'e': 0.001,
        'i': np.degrees(_sso_inclination(7078.0, 0.001)),
        'raan': 0.0, 'argp': 0.0, 'theta': 0.0,
        'descrizione': 'Sun-synchronous orbit: fixed local solar time at passes.',
        'colore': '#3fb950',
    },
    'Polar LEO': {
        'a': 7000.0, 'e': 0.001,
        'i': 90.0, 'raan': 45.0, 'argp': 0.0, 'theta': 0.0,
        'descrizione': 'Classic polar orbit.',
        'colore': '#bf7af0',
    },
    'GTO (Geostationary Transfer)': {
        'a': 24400.0, 'e': 0.73,
        'i': 27.0, 'raan': 0.0, 'argp': 180.0, 'theta': 0.0,
        'descrizione': 'Hohmann transfer orbit from LEO to GEO.',
        'colore': '#ff9f40',
    },
    'Tundra': {
        'a': 42164.0, 'e': 0.27,
        'i': 63.4, 'raan': 0.0, 'argp': 270.0, 'theta': 0.0,
        'descrizione': 'Geosynchronous highly-elliptic orbit, tear-drop ground track.',
        'colore': '#00d4b1',
    },
}


def get_satellite_params(name):
    """Return a dict of parameters for a satellite preset (angles in degrees)."""
    if name not in SATELLITES:
        return None
    return SATELLITES[name].copy()


def list_satellites():
    return list(SATELLITES.keys())
