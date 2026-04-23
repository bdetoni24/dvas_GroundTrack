# Ground Track Visualizer

Applicazione interattiva in Python/PyQt5 per la visualizzazione del *ground track* di satelliti e della loro orbita 3D attorno alla Terra. Sviluppata come progetto didattico per il corso di **Dinamica del volo aerospaziale** (A.A. 2025/26, prof. Carlo Bettanini Fecia Di Cossato).

## Funzionalità

- Inserimento dei **6 parametri Kepleriani** (a, e, i, Ω, ω, θ₀).
- **Preset** di satelliti notevoli (ISS, GPS, GEO, Molniya, SSO, ecc.).
- **Ground track 2D** su mappa terrestre in sistema ECEF (terrestre) o ECI (inerziale).
- **Vista 3D orbitale** con Terra texturata, piano equatoriale, piano orbitale, linea dei nodi, linea degli apsidi e indicatori angolari dei parametri kepleriani.
- **Animazione temporale** con velocità configurabile e durata selezionabile in numero di periodi.
- Opzione di **perturbazione J₂** (secolare su Ω e ω) per orbite SSO / Molniya.
- Pannello con **grandezze derivate** (periodo, pericentro, apocentro, momento angolare, energia, flight path angle, velocità, altitudine, ecc.).

## Requisiti

- Python 3.9+
- Dipendenze: PyQt5, NumPy, Matplotlib, SciPy, Pillow

## Installazione

```bash
pip install -r app/requirements.txt
```

## Avvio

```bash
cd app
python main.py
```

Su Windows è disponibile anche `app/run.bat`.

## Struttura del progetto

```
app/
├── main.py           # GUI principale (PyQt5)
├── orbital.py        # Meccanica orbitale (Keplero, propagazione, trasformazioni)
├── satellites.py     # Preset di satelliti
├── view_2d.py        # Vista ground track 2D
├── view_3d.py        # Vista 3D orbitale
├── assets/           # Texture della Terra
├── requirements.txt
└── run.bat
```

## Crediti

Appunti originali del corso di Dinamica del volo aerospaziale 2025/26 di **De Toni Bernardo** e **Da Ros Nicola**.
