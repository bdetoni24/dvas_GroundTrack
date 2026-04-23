"""
Modulo di meccanica orbitale per il Ground Track Visualizer.
Implementa le formule del corso "Dinamica del volo aerospaziale".

Convenzioni (da appunti pag. 21):
    PER -> ECI: T = R3(-Omega) * R1(-i) * R3(-omega)
    ECI -> PER: T = R3(omega) * R1(i) * R3(Omega)
    R3(a) = [[cos a, sin a, 0],[-sin a, cos a, 0],[0, 0, 1]]
    R1(a) = [[1, 0, 0],[0, cos a, sin a],[0, -sin a, cos a]]
"""
import numpy as np


# Costanti della Terra (appunti pag. 10)
MU_EARTH = 398600.4418       # km^3/s^2 - parametro gravitazionale standard
R_EARTH = 6378.137           # km - raggio equatoriale
J2_EARTH = 1.08262668e-3     # coefficiente J2
T_SIDEREAL = 86164.0905      # s - giorno siderale
OMEGA_EARTH = 2 * np.pi / T_SIDEREAL   # rad/s ~ 7.2921159e-5
OMEGA_EARTH_DEG_H = np.degrees(OMEGA_EARTH) * 3600   # ~15.04 deg/h
OMEGA_SUN = 2 * np.pi / (365.2421897 * 86400)        # rad/s (per SSO)


# =====================================================================
#  1. Equazione di Keplero e conversioni tra anomalie
# =====================================================================

def solve_kepler_elliptic(M, e, tol=1e-12, max_iter=100):
    """
    Risolve M = E - e*sin(E) con Newton-Raphson (orbita ellittica).
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    # Riduci nel range [-pi, pi]
    M = ((M + np.pi) % (2 * np.pi)) - np.pi
    # Initial guess
    E = np.where(np.abs(M) < 0.1, M, np.pi * np.sign(M))
    if e >= 0.6:
        E = np.pi * np.sign(M)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break
    return E if E.size > 1 else float(E[0])


def solve_kepler_hyperbolic(M, e, tol=1e-12, max_iter=200):
    """Risolve M = e*sinh(F) - F (orbita iperbolica)."""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    F = np.arcsinh(M / e) if np.all(M != 0) else np.asarray(M)
    for _ in range(max_iter):
        f = e * np.sinh(F) - F - M
        fp = e * np.cosh(F) - 1
        dF = -f / fp
        F = F + dF
        if np.all(np.abs(dF) < tol):
            break
    return F if F.size > 1 else float(F[0])


def eccentric_to_true_anomaly(E, e):
    """E -> theta per orbita ellittica."""
    return 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                            np.sqrt(1 - e) * np.cos(E / 2))


def true_to_eccentric_anomaly(theta, e):
    """theta -> E per orbita ellittica."""
    return 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(theta / 2),
                            np.sqrt(1 + e) * np.cos(theta / 2))


def hyperbolic_to_true_anomaly(F, e):
    """F -> theta per orbita iperbolica."""
    return 2.0 * np.arctan2(np.sqrt(e + 1) * np.sinh(F / 2),
                            np.sqrt(e - 1) * np.cosh(F / 2))


def mean_to_true_anomaly(M, e):
    """M -> theta per qualunque tipo di orbita (ellittica, parabolica, iperbolica)."""
    if e < 1.0:
        E = solve_kepler_elliptic(M, e)
        return eccentric_to_true_anomaly(E, e)
    if abs(e - 1.0) < 1e-10:
        # Equazione di Barker (risoluzione analitica della cubica)
        Mp = np.atleast_1d(np.asarray(M, dtype=float))
        A = 1.5 * Mp
        B = np.cbrt(A + np.sqrt(A ** 2 + 1))
        theta = 2 * np.arctan(B - 1.0 / B)
        return theta if theta.size > 1 else float(theta[0])
    # Iperbolica
    F = solve_kepler_hyperbolic(M, e)
    return hyperbolic_to_true_anomaly(F, e)


# =====================================================================
#  2. Trasformazioni tra sistemi di riferimento
# =====================================================================

def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def perifocal_to_eci_matrix(raan, i, argp):
    """T_PER->ECI = R3(-Omega) R1(-i) R3(-omega)"""
    return rot_z(-raan) @ rot_x(-i) @ rot_z(-argp)


def eci_to_ecef_matrix(theta_g):
    """Rotazione attorno a z di un angolo theta_g (tempo siderale di Greenwich)."""
    return rot_z(theta_g)


def eci_to_latlon(r_eci):
    """
    Dalla posizione ECI calcola declinazione (latitudine celeste) e ascensione retta.
    Usata per il "ground track ECI".
    """
    x, y, z = r_eci
    r = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def ecef_to_latlon(r_ecef):
    """
    Dalla posizione ECEF calcola latitudine e longitudine geocentriche (gradi).
    (approssimazione sferica: adeguata per ground track visualization)
    """
    x, y, z = r_ecef
    r = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


# =====================================================================
#  3. Conversioni Keplerian <-> cartesiane
# =====================================================================

def keplerian_to_eci(a, e, i, raan, argp, theta, mu=MU_EARTH):
    """
    Converte i 6 parametri Kepleriani in posizione e velocità nel sistema ECI.
    Parametri:
        a: semiasse maggiore [km] (per parabola: a=rp, tratteremo come p/2)
        e: eccentricità
        i, raan, argp, theta: angoli in radianti
    Returns:
        r_eci, v_eci (ciascuno numpy array shape (3,))
    """
    if abs(e - 1.0) < 1e-10:
        p = 2.0 * a           # per parabola: p = 2*rp (appunti pag. 11)
    else:
        p = a * (1 - e ** 2)

    r_mag = p / (1 + e * np.cos(theta))
    h = np.sqrt(mu * p)

    # Posizione e velocità nel sistema perifocale (appunti pag. 21)
    r_per = np.array([r_mag * np.cos(theta),
                      r_mag * np.sin(theta),
                      0.0])
    v_per = np.array([-mu / h * np.sin(theta),
                       mu / h * (e + np.cos(theta)),
                       0.0])

    T = perifocal_to_eci_matrix(raan, i, argp)
    return T @ r_per, T @ v_per


def eci_to_keplerian(r, v, mu=MU_EARTH):
    """
    Algoritmo "classico" per ricavare i 6 parametri da r,v (appunti pag. 20).
    Restituisce (a, e, i, raan, argp, theta) con angoli in radianti.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)

    # Vettore eccentricità (appunti pag. 4)
    e_vec = np.cross(v, h_vec) / mu - r / r_mag
    e = np.linalg.norm(e_vec)

    # Inclinazione
    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    # Linea dei nodi
    k_hat = np.array([0, 0, 1])
    N_vec = np.cross(k_hat, h_vec)
    N = np.linalg.norm(N_vec)

    # RAAN
    if N < 1e-10:
        raan = 0.0
    else:
        raan = np.arccos(np.clip(N_vec[0] / N, -1.0, 1.0))
        if N_vec[1] < 0:
            raan = 2 * np.pi - raan

    # Argomento di pericentro
    if N < 1e-10 or e < 1e-10:
        argp = 0.0
    else:
        argp = np.arccos(np.clip(np.dot(N_vec, e_vec) / (N * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp

    # Anomalia vera
    if e < 1e-10:
        if N < 1e-10:
            theta = np.arccos(np.clip(r[0] / r_mag, -1.0, 1.0))
            if r[1] < 0:
                theta = 2 * np.pi - theta
        else:
            theta = np.arccos(np.clip(np.dot(N_vec, r) / (N * r_mag), -1.0, 1.0))
            if r[2] < 0:
                theta = 2 * np.pi - theta
    else:
        theta = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
        if np.dot(r, v) < 0:
            theta = 2 * np.pi - theta

    # Semiasse maggiore (da energia specifica - appunti pag. 4)
    xi = v_mag ** 2 / 2 - mu / r_mag
    if abs(xi) > 1e-10:
        a = -mu / (2 * xi)
    else:
        a = np.inf

    return a, e, i, raan, argp, theta


# =====================================================================
#  4. Propagazione
# =====================================================================

def orbital_period(a, mu=MU_EARTH):
    """Terza legge di Keplero: T^2 = 4*pi^2*a^3/mu (appunti pag. 9)."""
    if a <= 0:
        return np.inf
    return 2 * np.pi * np.sqrt(a ** 3 / mu)


def mean_motion(a, mu=MU_EARTH):
    return np.sqrt(mu / abs(a) ** 3)


def propagate_keplerian(a, e, i, raan, argp, theta0, t_array,
                        mu=MU_EARTH, j2=False):
    """
    Propaga un'orbita kepleriana nel tempo.
    Se j2=True applica le derive secolari su Omega e omega (appunti pag. 24).

    Returns:
        r_eci: array shape (N, 3)
        v_eci: array shape (N, 3)
        theta_arr: array shape (N,)
        raan_arr, argp_arr: array shape (N,) - elementi perturbati
    """
    t_array = np.atleast_1d(np.asarray(t_array, dtype=float))
    N = len(t_array)

    # Anomalia media iniziale
    if e < 1.0:
        E0 = true_to_eccentric_anomaly(theta0, e)
        M0 = E0 - e * np.sin(E0)
    elif abs(e - 1) < 1e-10:
        tan_half = np.tan(theta0 / 2)
        M0 = 0.5 * tan_half + (1.0 / 6.0) * tan_half ** 3
    else:  # iperbolica
        F0 = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(theta0 / 2))
        M0 = e * np.sinh(F0) - F0

    # Moto medio
    if e < 1.0:
        n = np.sqrt(mu / a ** 3)
    elif abs(e - 1) < 1e-10:
        # parabola: n = sqrt(mu/rp^3) with convention where M = 1/2*tan(th/2) + 1/6*tan^3
        # più semplicemente dM/dt = h/r_p^2 scaled; qui usiamo:
        rp = a  # stored as rp for parabola
        n = np.sqrt(mu / (2 * rp ** 3))
    else:
        n = np.sqrt(mu / abs(a) ** 3)

    # Derive secolari J2 (se richieste)
    if j2 and e < 1.0:
        p = a * (1 - e ** 2)
        factor = -1.5 * n * J2_EARTH * (R_EARTH / p) ** 2
        raan_dot = factor * np.cos(i)
        argp_dot = -factor * (2.5 * np.sin(i) ** 2 - 2.0)
    else:
        raan_dot = 0.0
        argp_dot = 0.0

    r_eci = np.zeros((N, 3))
    v_eci = np.zeros((N, 3))
    theta_arr = np.zeros(N)
    raan_arr = raan + raan_dot * t_array
    argp_arr = argp + argp_dot * t_array

    for k, t in enumerate(t_array):
        M = M0 + n * t
        th = mean_to_true_anomaly(M, e)
        theta_arr[k] = th
        r, v = keplerian_to_eci(a, e, i, raan_arr[k], argp_arr[k], th, mu)
        r_eci[k] = r
        v_eci[k] = v

    return r_eci, v_eci, theta_arr, raan_arr, argp_arr


def eci_array_to_ecef(r_eci_array, t_array, theta_g0=0.0, omega_e=OMEGA_EARTH):
    """
    Converte un array di vettori ECI in ECEF.
    theta_g(t) = theta_g0 + omega_e * t
    """
    r_eci_array = np.asarray(r_eci_array)
    t_array = np.asarray(t_array)
    theta_g = theta_g0 + omega_e * t_array
    r_ecef = np.zeros_like(r_eci_array)
    for k in range(len(t_array)):
        r_ecef[k] = rot_z(theta_g[k]) @ r_eci_array[k]
    return r_ecef


def compute_groundtrack(r_ecef_array):
    """Ritorna (lats, lons) in gradi per un array di posizioni ECEF."""
    x = r_ecef_array[:, 0]
    y = r_ecef_array[:, 1]
    z = r_ecef_array[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    # wrap nel range [-180, 180]
    lon = ((lon + 180) % 360) - 180
    return lat, lon


# =====================================================================
#  5. Grandezze derivate utili (appunti vari)
# =====================================================================

def orbital_info(a, e, i, raan, argp, theta, mu=MU_EARTH):
    """Calcola un dizionario di grandezze di interesse per un'orbita."""
    info = {}
    if abs(e - 1.0) < 1e-10:
        p = 2 * a
        rp = a
        ra = np.inf
    else:
        p = a * (1 - e ** 2)
        rp = a * (1 - e) if e < 1 else a * (e - 1)
        ra = a * (1 + e) if e < 1 else np.inf

    h = np.sqrt(mu * abs(p))
    r = abs(p) / (1 + e * np.cos(theta))
    v = np.sqrt(mu * (2 / r - 1 / a)) if e != 1 else np.sqrt(2 * mu / r)
    v_perp = h / r
    v_r = mu / h * e * np.sin(theta)
    gamma = np.degrees(np.arctan2(v_r, v_perp))
    energy = -mu / (2 * a) if e != 1 else 0.0

    info['periodo'] = orbital_period(a, mu) if e < 1 else np.inf
    info['semilato_retto'] = p
    info['pericentro'] = rp
    info['apocentro'] = ra
    info['momento_angolare'] = h
    info['energia_specifica'] = energy
    info['raggio_corrente'] = r
    info['velocita_corrente'] = v
    info['vel_radiale'] = v_r
    info['vel_ortogonale'] = v_perp
    info['flight_path_angle'] = gamma
    info['altitudine'] = r - R_EARTH
    info['vel_pericentro'] = np.sqrt(mu / rp) * np.sqrt(1 + e) if rp > 0 else np.nan
    info['vel_apocentro'] = np.sqrt(mu / ra) * np.sqrt(1 - e) if np.isfinite(ra) else 0.0
    info['vel_fuga'] = np.sqrt(2 * mu / r)
    info['tipo_orbita'] = (
        'circolare' if e < 1e-3 else
        'ellittica' if e < 1.0 else
        'parabolica' if abs(e - 1) < 1e-6 else
        'iperbolica'
    )
    return info
