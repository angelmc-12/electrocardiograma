# app.py
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import wfdb

from utils.ekg import (
    load_wfdb_record, pick_lead, clean_ecg,
    detect_r_and_hr, apply_ekg_grid_shapes, list_local_records,
    compute_rr_intervals, nice_ylim, build_speed_gain_badge
)

# ------------------ Config & Título ------------------
st.set_page_config(page_title="ECG Explorer — 12 derivaciones", layout="wide")
st.markdown("""
<style>
/* Badge estilo “pill” */
.badge {display:inline-block;padding:4px 10px;border-radius:999px;font-weight:600;font-size:12px;margin-right:6px;}
.badge-ok {background:#e8f5e9;color:#256029;border:1px solid #c8e6c9;}
.badge-warn {background:#fff3e0;color:#8a4f00;border:1px solid #ffe0b2;}
.badge-alert {background:#ffebee;color:#b71c1c;border:1px solid #ffcdd2;}
.kpi-card {border:1px solid #eaeaea;border-radius:14px;padding:12px 16px;margin:0 8px;background:white;}
.kpi-value {font-size:28px;font-weight:700;margin:0;color:#2c3e50;} 
.kpi-label {font-size:12px;color:#666;margin:0;}
</style>
""", unsafe_allow_html=True)

st.title("ECG Explorer — 12 derivaciones")

# ------------------ Sidebar ------------------
records = list_local_records("data")
if not records:
    st.error("No se encontraron registros WFDB en ./data (parejas .hea/.mat).")
    st.stop()

with st.sidebar:
    st.header("Opciones")
    rec_base = st.selectbox("Registro", records, index=0)
    lead_names_default = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    # Carga registro
    record, signal, fs, sig_names = load_wfdb_record(rec_base)
    if len(sig_names) != signal.shape[1]:
        sig_names = lead_names_default[:signal.shape[1]]

    # Selección de derivación (por defecto II si está)
    default_idx = sig_names.index('II') if 'II' in sig_names else 0
    lead_name = st.selectbox("Derivación", sig_names, index=default_idx)
    lead_idx = sig_names.index(lead_name)

    seconds = st.slider("Segundos a mostrar", min_value=2, max_value=10, value=5)
    show_grid = st.checkbox("Mostrar papel EKG", value=True)
    clean_toggle = st.checkbox("Limpiar señal (neurokit2.ecg_clean)", value=True)
    range_ok = st.slider("Rango normal FC (lpm)", 40, 140, (60,100))
    # show_montage = st.checkbox("Vista rápida 12 derivaciones (montaje 3×4)", value=False)

    # Controles de performance para el montaje
    # montage_seconds = st.slider("Segs en montaje (3–4 sugerido)", 2, 6, 3)
    # montage_clean = st.checkbox("Limpieza en montaje (lenta)", value=False)

# ------------------ Clasificación (inferencia) ------------------
LABEL_NAMES = ["Sinus Bradycardia", "Sinus Rhythm", "Atrial Fibrillation", "Sinus Tachycardia"]
MODEL_PATH = "models/best_ecgnet.pt"

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
    x = x.astype(np.float32)
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-7
    x = (x - m) / s
    x = x.T  # (12,5000)
    x = np.expand_dims(x, 0)  # (1,12,5000)
    return torch.from_numpy(x).float()

# ------------------ Procesamiento señal ------------------
ecg = pick_lead(signal, lead_idx)
ecg_proc = clean_ecg(ecg, fs) if clean_toggle else ecg

r_idx, hr_inst, hr_mean, status = detect_r_and_hr(ecg_proc, fs)
rr, rr_s = compute_rr_intervals(r_idx, fs)

# ------------------ Alerta y KPIs (storytelling arriba) ------------------
if np.isnan(hr_mean):
    alerta_txt = "No se pudo calcular FC."
    alerta_class = "badge-alert"
else:
    if range_ok[0] <= hr_mean <= range_ok[1]:
        alerta_txt = f"FC media: {hr_mean:.1f} lpm — dentro de rango"
        alerta_class = "badge-ok"
    else:
        alerta_txt = f"FC media: {hr_mean:.1f} lpm — fuera de rango {range_ok[0]}–{range_ok[1]} lpm"
        alerta_class = "badge-alert"

colA, colB, colC, colD = st.columns([1.2,1,1,1.2])
with colA:
    st.markdown(f'<span class="badge {alerta_class}">🩺 {alerta_txt}</span> {build_speed_gain_badge(25, 10)}', unsafe_allow_html=True)
with colB:
    st.markdown('<div class="kpi-card"><p class="kpi-value">{:.1f}</p><p class="kpi-label">FC media (lpm)</p></div>'.format(0 if np.isnan(hr_mean) else hr_mean), unsafe_allow_html=True)
with colC:
    beats = int(len(r_idx))
    st.markdown(f'<div class="kpi-card"><p class="kpi-value">{beats}</p><p class="kpi-label">Latidos detectados</p></div>', unsafe_allow_html=True)
with colD:
    rr_mean = float(np.nanmean(rr_s)) if rr_s.size else np.nan
    st.markdown('<div class="kpi-card"><p class="kpi-value">{}</p><p class="kpi-label">RR medio (s)</p></div>'.format("—" if np.isnan(rr_mean) else f"{rr_mean:.2f}"), unsafe_allow_html=True)

# ------------------ Cache para montaje (rápido) ------------------
@st.cache_data(show_spinner=False)
def get_montage_series(signal, fs, seconds, clean, max_points=2000):
    """
    Devuelve dict {lead_index: (t_ds, y_ds, y0, y1)} para hasta 12 derivaciones.
    - Recorta a 'seconds'
    - (Opcional) limpia (método 'biosppy' si está disponible)
    - Downsamplea a ~max_points
    """
    import neurokit2 as nk
    Nm = int(seconds * fs)
    n_leads = min(12, signal.shape[1])
    out = {}
    t = np.arange(Nm) / fs

    def downsample_xy(x, y, max_points=2000):
        n = len(x)
        if n <= max_points:
            return x, y
        idx = np.linspace(0, n - 1, max_points).astype(int)
        return x[idx], y[idx]

    for i in range(n_leads):
        y = signal[:Nm, i].astype(float)
        if clean:
            try:
                y = nk.ecg_clean(y, sampling_rate=fs, method="biosppy")
            except Exception:
                y = nk.ecg_clean(y, sampling_rate=fs)
        t_ds, y_ds = downsample_xy(t, y, max_points=max_points)
        y0, y1 = nice_ylim(y_ds)
        out[i] = (t_ds, y_ds, y0, y1)
    return out

# ------------------ Tabs ------------------
tab_vis, tab_hr, tab_cls = st.tabs(["👀 Visor", "📈 Frecuencia cardíaca", "🧠 Clasificación"])

# ========= Tab Visor =========
with tab_vis:
    N = int(seconds * fs)
    t = np.arange(len(ecg_proc)) / fs
    t_win, y_win = t[:N], ecg_proc[:N]
    r_win = r_idx[(r_idx >= 0) & (r_idx < N)]

    # Figura principal (interactiva con Plotly, WebGL para fluidez)
    y0, y1 = nice_ylim(y_win)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t_win, y=y_win, mode="lines", name=f"{lead_name}",
        line=dict(width=1.2),
        hovertemplate="t=%{x:.3f} s<br>V=%{y:.3f} mV<extra></extra>"
    ))
    if r_win.size > 0:
        fig.add_trace(go.Scattergl(
            x=t_win[r_win], y=y_win[r_win], mode="markers", name="Picos R",
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="R @ %{x:.3f} s<extra></extra>"
        ))

    # Cuadrícula tipo papel EKG (shapes)
    if show_grid:
        fig.update_layout(shapes=apply_ekg_grid_shapes(
            x0=0, x1=seconds, y0=y0, y1=y1,
            x_minor=0.04, x_major=0.20, y_minor=0.1, y_major=0.5,
            minor_color="#ffcccc", major_color="#ff9999", minor_w=0.5, major_w=1.0
        ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud (mV)",
        xaxis=dict(range=[0, seconds], zeroline=False),
        yaxis=dict(range=[y0, y1], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Montaje 12 derivaciones (rápido) ----
    # if show_montage and signal.shape[1] >= 2:
    #     rows, cols = 3, 4
    #     rtot = min(rows * cols, signal.shape[1])
    #     seconds_montage = montage_seconds  # del sidebar

    #     with st.spinner("Preparando montaje 12 derivaciones…"):
    #         montage_data = get_montage_series(signal, fs, seconds_montage, montage_clean, max_points=2000)

    #     figm = make_subplots(
    #         rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=False,
    #         vertical_spacing=0.06, horizontal_spacing=0.04,
    #         subplot_titles=sig_names[:rtot]
    #     )

    #     # Shapes por subgráfico (reutilizables por rango)
    #     grid_shapes_cache = {}

    #     for i in range(rtot):
    #         r = i // cols + 1
    #         c = i % cols + 1

    #         t_ds, y_ds, y0i, y1i = montage_data[i]
    #         figm.add_trace(
    #             go.Scattergl(
    #                 x=t_ds, y=y_ds, mode="lines", line=dict(width=1),
    #                 hovertemplate="t=%{x:.3f} s<br>V=%{y:.3f} mV<extra></extra>",
    #                 name=sig_names[i]
    #             ),
    #             row=r, col=c
    #         )
    #         figm.update_yaxes(range=[y0i, y1i], row=r, col=c)

    #         if show_grid:
    #             key = (0, seconds_montage, round(y0i, 2), round(y1i, 2))
    #             if key not in grid_shapes_cache:
    #                 # Colores un toque más suaves en el montaje
    #                 grid_shapes_cache[key] = apply_ekg_grid_shapes(
    #                     0, seconds_montage, y0i, y1i,
    #                     x_minor=0.04, x_major=0.20, y_minor=0.1, y_major=0.5,
    #                     minor_color="#ffebee", major_color="#ffcdd2",
    #                     minor_w=0.4, major_w=0.8
    #                 )
    #             for sh in grid_shapes_cache[key]:
    #                 figm.add_shape(sh, row=r, col=c)

    #     figm.update_layout(
    #         height=700,
    #         margin=dict(l=20, r=20, t=40, b=20),
    #         showlegend=False
    #     )
    #     # Etiquetas compactas
    #     for ax in figm.layout:
    #         if ax.startswith("yaxis"):
    #             getattr(figm.layout, ax).zeroline = False
    #         if ax.startswith("xaxis"):
    #             getattr(figm.layout, ax).title = "s"

    #     st.plotly_chart(figm, use_container_width=True)

    with st.expander("Metadatos del registro"):
        st.write({
            "Frecuencia de muestreo (Hz)": fs,
            "Nº muestras (lead seleccionado)": int(len(ecg)),
            "Nº derivaciones": int(signal.shape[1]),
            "Derivaciones": sig_names
        })
    st.caption("Reglas del papel: menor=0.04 s / 0.1 mV, mayor=0.20 s / 0.5 mV. Velocidad=25 mm/s, Ganancia=10 mm/mV.")

# ========= Tab Frecuencia Cardíaca =========
with tab_hr:
    st.markdown("**Método:** detección de picos R (neurokit2) → FC instantánea por inverso del RR.")

    # Serie de FC
    N = int(seconds * fs)
    t = np.arange(len(ecg_proc)) / fs
    hr_win = hr_inst[:N] if hr_inst is not None else None

    if hr_win is not None and np.isfinite(hr_win).any():
        fig2 = go.Figure()
        fig2.add_trace(go.Scattergl(x=t[:N], y=hr_win, mode="lines", name="FC (lpm)",
                                    hovertemplate="t=%{x:.2f} s<br>FC=%{y:.1f} lpm<extra></extra>"))
        # Rango color de referencia
        fig2.add_hrect(y0=range_ok[0], y1=range_ok[1], fillcolor="#e8f5e9", opacity=0.45, line_width=0)
        fig2.update_layout(
            margin=dict(l=20,r=20,t=40,b=10),
            xaxis_title="Tiempo (s)",
            yaxis_title="FC (lpm)",
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Descargables (HR y RR)
        col1, col2 = st.columns(2)
        with col1:
            df_hr = pd.DataFrame({"t_s": t[:N], "HR_lpm": hr_win})
            csv_hr = df_hr.to_csv(index=False).encode()
            st.download_button("⬇️ Descargar FC (CSV)", data=csv_hr, file_name=f"{os.path.basename(rec_base)}_HR.csv", mime="text/csv")
        with col2:
            df_rr = pd.DataFrame({"RR_s": rr_s}) if rr_s.size else pd.DataFrame({"RR_s":[]})
            csv_rr = df_rr.to_csv(index=False).encode()
            st.download_button("⬇️ Descargar RR (CSV)", data=csv_rr, file_name=f"{os.path.basename(rec_base)}_RR.csv", mime="text/csv")
    else:
        st.warning("No se pudo calcular la serie de FC. Prueba otra derivación o habilita limpieza de señal.")

# ========= Tab Clasificación =========
with tab_cls:
    st.markdown("Clasificación automática (4 clases) — *demo educativa*")
    if st.button("🔎 Clasificar este registro"):
        try:
            model, device = load_classifier()
            x = preprocess_12lead_for_model_from_base(rec_base)
            with torch.no_grad():
                logits = model(x.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
            pred_idx = int(np.argmax(probs))
            pred_label = LABEL_NAMES[pred_idx]
            if pred_label == "Sinus Bradycardia":
                color = "badge-warn"
                note = "Frecuencia baja — verificar contexto clínico."
            elif pred_label == "Sinus Tachycardia":
                color = "badge-warn"
                note = "Frecuencia alta — verificar desencadenantes."
            elif pred_label == "Atrial Fibrillation":
                color = "badge-alert"
                note = "Ritmo irregular — evaluar trazado completo."
            else:
                color = "badge-ok"
                note = "Ritmo sinusal."

            st.markdown(f'<span class="badge {color}">Predicción: <b>{pred_label}</b></span> <span class="badge">{note}</span>', unsafe_allow_html=True)

            dfp = pd.DataFrame({"Clase": LABEL_NAMES, "Probabilidad": probs})
            figp = go.Figure(go.Bar(x=dfp["Clase"], y=dfp["Probabilidad"], text=[f"{p*100:.1f}%" for p in probs],
                                    textposition="outside"))
            figp.update_yaxes(range=[0, 1.0])
            figp.update_layout(margin=dict(l=20,r=20,t=40,b=40), yaxis_title="Probabilidad", xaxis_title="")
            st.plotly_chart(figp, use_container_width=True)
        except Exception as e:
            st.error(f"Ocurrió un error durante la clasificación: {e}")
