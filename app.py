# app.py
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.ekg import (
    load_wfdb_record, pick_lead, clean_ecg,
    detect_r_and_hr, apply_ekg_grid, list_local_records
)

st.set_page_config(page_title="ECG Explorer", layout="wide")
st.title("ECG Explorer — 12 derivaciones")

# ---------- Sidebar: selección de registro y opciones ----------
records = list_local_records("data")
if not records:
    st.error("No se encontraron registros WFDB en ./data (parejas .hea/.mat).")
    st.stop()

rec_base = st.sidebar.selectbox("Registro", records, index=0)
lead_names_default = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# Carga registro
record, signal, fs, sig_names = load_wfdb_record(rec_base)
if len(sig_names) != signal.shape[1]:
    sig_names = lead_names_default[:signal.shape[1]]

lead_name = st.sidebar.selectbox("Derivación", sig_names, index=1 if 'II' in sig_names else 0)
lead_idx = sig_names.index(lead_name)

seconds = st.sidebar.slider("Segundos a mostrar", min_value=2, max_value=10, value=5)
show_grid = st.sidebar.checkbox("Mostrar papel EKG", value=True)
clean_toggle = st.sidebar.checkbox("Limpiar señal (neurokit2.ecg_clean)", value=True)
range_ok = st.sidebar.slider("Rango normal FC (lpm)", 40, 140, (60,100))

# --- Clasificación (inferencia) ---
import torch
import torch.nn as nn

# Las 4 clases que entrenaste
LABEL_NAMES = ["Sinus Bradycardia", "Sinus Rhythm", "Atrial Fibrillation", "Sinus Tachycardia"]
MODEL_PATH = "models/best_ecgnet.pt"  # ruta al modelo entrenado

# Define la misma arquitectura que usaste para entrenar
class ECGNet(nn.Module):
    def __init__(self, in_ch=12, n_cls=4):
        super().__init__()
        def block(ci, co, k=7, s=2, p=3):
            return nn.Sequential(
                nn.Conv1d(ci, co, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm1d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            )
        self.feat = nn.Sequential(
            block(in_ch, 32),
            block(32, 64),
            block(64, 128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, n_cls))
    def forward(self, x):
        x = self.feat(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)

@st.cache_resource
def load_classifier():
    device = torch.device("cpu")
    model = ECGNet(in_ch=12, n_cls=len(LABEL_NAMES)).to(device).float()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

def preprocess_12lead_for_model_from_base(rec_base):
    """
    Lee el registro completo en 12 derivaciones desde rec_base (sin extensión),
    lo normaliza (z-score por canal) y devuelve tensor (1,12,5000).
    """
    sig, fields = wfdb.rdsamp(rec_base)  # (N, 12)
    target_len = 5000
    n = sig.shape[0]
    if n == target_len:
        x = sig
    elif n > target_len:
        start = (n - target_len)//2
        x = sig[start:start+target_len, :]
    else:
        pad = target_len - n
        x = np.pad(sig, ((0,pad),(0,0)), mode="constant")
    # normalización por canal
    x = x.astype(np.float32)
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-7
    x = (x - m) / s
    x = x.T  # (12,5000)
    x = np.expand_dims(x, 0)  # (1,12,5000)
    return torch.from_numpy(x).float()

# ---------- Procesamiento ----------
ecg = pick_lead(signal, lead_idx)
ecg_proc = clean_ecg(ecg, fs) if clean_toggle else ecg

r_idx, hr_inst, hr_mean, status = detect_r_and_hr(ecg_proc, fs)

# Alerta
if np.isnan(hr_mean):
    alerta = "⚠ No se pudo calcular FC."
elif range_ok[0] <= hr_mean <= range_ok[1]:
    alerta = f"✅ {status} (OK)"
else:
    alerta = f"⚠ {status} (fuera de rango {range_ok[0]}–{range_ok[1]} lpm)"

st.subheader(alerta)

# ---------- Plot ECG + picos R ----------
N = int(seconds * fs)
t = np.arange(len(ecg_proc)) / fs
t_win, y_win = t[:N], ecg_proc[:N]
r_win = r_idx[(r_idx >= 0) & (r_idx < N)]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_win, y_win, color="black", linewidth=1, label="ECG")
if r_win.size > 0:
    ax.scatter(t_win[r_win], y_win[r_win], s=15, color="red", zorder=3, label="Picos R")

if show_grid:
    apply_ekg_grid(ax)

ax.set_xlim(0, seconds)
# márgenes verticales agradables
pad = max(0.5, 0.1 * (np.nanmax(y_win) - np.nanmin(y_win)))
ax.set_ylim(np.nanmin(y_win)-pad, np.nanmax(y_win)+pad)

ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Amplitud (mV)")
title = f"{os.path.basename(rec_base)} — {lead_name} — FS={fs} Hz"
ax.set_title(title)
ax.legend(loc="upper right")
st.pyplot(fig, use_container_width=True)

# ---------- Curva de FC (opcional) ----------
if r_win.size > 0:
    fig2, ax2 = plt.subplots(figsize=(12, 2.5))
    ax2.plot(t[:N], hr_inst[:N], linewidth=1.2)
    ax2.set_ylabel("FC (lpm)")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_title("Frecuencia cardíaca instantánea")
    st.pyplot(fig2, use_container_width=True)

# ---------- Metadatos rápidos ----------
with st.expander("Metadatos del registro"):
    st.write({
        "Frecuencia de muestreo (Hz)": fs,
        "Nº muestras (lead seleccionado)": int(len(ecg)),
        "Nº derivaciones": int(signal.shape[1]),
        "Derivaciones": sig_names
    })

st.caption("Nota: picos R y FC calculados con neurokit2; papel EKG: 1 mm=0.04 s y 0.1 mV; cuadros grandes=0.20 s y 0.5 mV.")

# ---------- Clasificación automática ----------
st.markdown("---")
st.subheader("Clasificación automática (4 clases)")

if st.button("Clasificar este registro"):
    try:
        model, device = load_classifier()
        x = preprocess_12lead_for_model_from_base(rec_base)
        with torch.no_grad():
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        pred_idx = int(np.argmax(probs))
        pred_label = LABEL_NAMES[pred_idx]
        st.success(f"Predicción: **{pred_label}**")
        # Mostrar probabilidades en gráfico de barras
        import pandas as pd
        dfp = pd.DataFrame({"Clase": LABEL_NAMES, "Probabilidad": probs})
        st.bar_chart(dfp.set_index("Clase"))
    except Exception as e:
        st.error(f"Ocurrió un error durante la clasificación: {e}")
