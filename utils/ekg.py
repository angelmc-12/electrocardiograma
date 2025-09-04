# utils/ekg.py
import os
import numpy as np
import wfdb
import neurokit2 as nk
from matplotlib.ticker import MultipleLocator
import hashlib

def load_wfdb_record(basepath: str):
    """
    basepath: ruta sin extensión (.hea/.mat), p.ej. 'data/JS00001'
    return: record (WFDB), signal (np.ndarray n_muestras x n_deriv), fs (int), sig_names (list)
    """
    record = wfdb.rdrecord(basepath)
    signal = record.p_signal.astype(float)   # señales físicas (normalmente mV)
    fs = int(record.fs)
    sig_names = getattr(record, "sig_name", [f"Lead {i}" for i in range(signal.shape[1])])
    return record, signal, fs, sig_names

def pick_lead(signal, idx: int):
    assert 0 <= idx < signal.shape[1], "Lead index fuera de rango"
    return signal[:, idx].astype(float)

def clean_ecg(ecg, fs: int):
    return nk.ecg_clean(ecg, sampling_rate=fs)

def detect_r_and_hr(ecg_clean, fs: int):
    # Devuelve indices de R, HR instantánea y media (lpm), junto con un mensaje de estado
    signals_df, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
    r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)

    if r_idx.size == 0:
        hr_inst = np.full(len(ecg_clean), np.nan)
        hr_mean = np.nan
        status = "⚠ No se detectaron picos R; prueba otra derivación o revisa la señal."
        return r_idx, hr_inst, hr_mean, status

    # Compatibilidad entre versiones
    try:
        hr_inst = nk.ecg_rate(peaks=r_idx, sampling_rate=fs, desired_length=len(ecg_clean))
    except TypeError:
        hr_inst = nk.signal_rate(peaks=r_idx, sampling_rate=fs, desired_length=len(ecg_clean))

    hr_mean = float(np.nanmean(hr_inst))
    status = f"FC media: {hr_mean:.1f} lpm"
    return r_idx, hr_inst, hr_mean, status

def apply_ekg_grid(ax):
    """
    Aplica la cuadrícula tipo papel EKG:
      - menor: 0.04 s (x), 0.1 mV (y)
      - mayor: 0.20 s (x), 0.5 mV (y)
    """
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which='minor', color='#ffcccc', linewidth=0.5)

    ax.xaxis.set_major_locator(MultipleLocator(0.20))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(which='major', color='#ff9999', linewidth=1.0)

def list_local_records(data_dir="data"):
    """
    Devuelve lista de rutas base (sin extensión) para todos los .hea en data_dir
    """
    bases = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".hea"):
            base = os.path.splitext(fname)[0]
            mat = os.path.join(data_dir, base + ".mat")
            if os.path.exists(mat):
                bases.append(os.path.join(data_dir, base))
    bases.sort()
    return bases

def apply_ekg_grid_shapes(x0, x1, y0, y1,
                          x_minor=0.04, x_major=0.20,
                          y_minor=0.1, y_major=0.5,
                          minor_color="#ffcccc", major_color="#ff9999",
                          minor_w=0.5, major_w=1.0):
    """
    Devuelve una lista de shapes (Plotly) que dibujan la cuadrícula tipo papel EKG.
    - Líneas menores: 0.04 s (x) y 0.1 mV (y)
    - Líneas mayores: 0.20 s (x) y 0.5 mV (y)
    """
    shapes = []
    # Verticales menores y mayores
    x = x0
    while x <= x1 + 1e-9:
        is_major = (abs((x - x0) / x_major - round((x - x0) / x_major)) < 1e-6)
        shapes.append(dict(
            type="line", x0=x, x1=x, y0=y0, y1=y1,
            line=dict(color=major_color if is_major else minor_color,
                      width=major_w if is_major else minor_w)
        ))
        x += x_minor
    # Horizontales menores y mayores
    y = y0
    while y <= y1 + 1e-9:
        is_major = (abs((y - y0) / y_major - round((y - y0) / y_major)) < 1e-6)
        shapes.append(dict(
            type="line", x0=x0, x1=x1, y0=y, y1=y,
            line=dict(color=major_color if is_major else minor_color,
                      width=major_w if is_major else minor_w)
        ))
        y += y_minor
    return shapes

def compute_rr_intervals(r_idx, fs: int):
    """Devuelve RR en muestras y en segundos (np.arrays)."""
    if r_idx is None or len(r_idx) < 2:
        return np.array([]), np.array([])
    rr = np.diff(r_idx).astype(float)
    return rr, rr / fs

def nice_ylim(y, pad_ratio=0.15, min_pad=0.4):
    """Devuelve (ymin, ymax) con padding agradable y redondeo a múltiplos de 0.5 mV."""
    y = np.asarray(y)
    y_min = float(np.nanmin(y)); y_max = float(np.nanmax(y))
    rng = y_max - y_min
    pad = max(min_pad, pad_ratio * (rng if rng > 0 else 1.0))
    y0 = y_min - pad
    y1 = y_max + pad
    # Redondear a múltiplos de 0.5 mV para que el papel mayor quede alineado
    def round_to_half(v, up=False):
        return (np.floor(v*2)/2.0) if not up else (np.ceil(v*2)/2.0)
    return round_to_half(y0), round_to_half(y1, up=True)

def build_speed_gain_badge(speed_mm_s=25, gain_mm_mV=10):
    return f'<span class="badge">Velocidad: {speed_mm_s} mm/s</span><span class="badge">Ganancia: {gain_mm_mV} mm/mV</span>'


def downsample_xy(x, y, max_points=2000):
    """Downsampling uniforme a ~max_points."""
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]

def hash_array(a: np.ndarray) -> str:
    """Hash rápido para cache (no criptográfico)."""
    return hashlib.md5(a.tobytes()).hexdigest()

def precompute_shapes_grid(x0, x1, y0, y1,
                           x_minor=0.04, x_major=0.20,
                           y_minor=0.1, y_major=0.5,
                           minor_color="#ffebee", major_color="#ffcdd2",
                           minor_w=0.4, major_w=0.8):
    """Igual a apply_ekg_grid_shapes pero pensado para reutilizar shapes."""
    from .ekg import apply_ekg_grid_shapes  # reutiliza tu versión
    return apply_ekg_grid_shapes(
        x0, x1, y0, y1,
        x_minor, x_major, y_minor, y_major,
        minor_color, major_color, minor_w, major_w
    )
