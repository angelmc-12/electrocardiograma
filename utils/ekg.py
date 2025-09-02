# utils/ekg.py
import os
import numpy as np
import wfdb
import neurokit2 as nk
from matplotlib.ticker import MultipleLocator

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
