# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import torch
import torch.nn as nn
import wfdb

from utils.ekg import (
    load_wfdb_record, pick_lead, clean_ecg,
    detect_r_and_hr, apply_ekg_grid_shapes, list_local_records,
    compute_rr_intervals, nice_ylim, build_speed_gain_badge
)

# ------------------ Config global ------------------
st.set_page_config(page_title="ECG Explorer — 12 derivaciones", layout="wide")
st.markdown("""
<style>
/* Badge estilo “pill” */
.badge {display:inline-block;padding:4px 10px;border-radius:999px;font-weight:600;font-size:12px;margin-right:6px;}
.badge-ok {background:#e8f5e9;color:#256029;border:1px solid #c8e6c9;}
.badge-warn {background:#fff3e0;color:#8a4f00;border:1px solid #ffe0b2;}
.badge-alert {background:#ffebee;color:#b71c1c;border:1px solid #ffcdd2;}
.kpi-card {border:1px solid #eaeaea;border-radius:14px;padding:12px 16px;margin:0 8px;background:white;}
.kpi-value {font-size:28px;font-weight:700;margin:0;color:#2c3e50;}  /* números en color oscuro */
.kpi-label {font-size:12px;color:#666;margin:0;}
</style>
""", unsafe_allow_html=True)

# ------------------ Datos base disponibles ------------------
records = list_local_records("data")
if not records:
    st.error("No se encontraron registros WFDB en ./data (parejas .hea/.mat).")
    st.stop()

# Cargamos el primero para mostrar info en Inicio (no hace procesamiento pesado)
try:
    record0, signal0, fs0, sig_names0 = load_wfdb_record(records[0])
except Exception:
    record0, signal0, fs0, sig_names0 = None, None, None, []

# ------------------ Sidebar: menú principal ------------------
with st.sidebar:
    selected = option_menu(
        menu_title="🩺 ECG Explorer",
        options=["📘 Inicio", "📊 Explorador"],
        icons=["book", "activity"],
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "#1E1E1E"},
            "icon": {"color": "#ffffff", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "color": "#ffffff", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#339af0", "color": "#ffffff", "font-weight": "bold", "border-radius":"8px"},
            "menu-title": {"font-size": "22px", "color": "#ffffff", "font-weight": "bold"}
        }
    )

# =====================================================================
#                               INICIO
# =====================================================================
if selected == "📘 Inicio":
    st.title("🩺 ECG Interactivo — Aprendiendo y Explorando")
    st.markdown("""
    _Un visor interactivo para comprender y analizar electrocardiogramas (ECG) siguiendo las **reglas del papel electrocardiográfico**._
    
    Esta aplicación está pensada para **público no técnico**, combinando visualización clara, métricas clave y una sección de **clasificación automática (demo educativa)**.
    """)

    st.header("🎯 Objetivos del proyecto")
    st.markdown("""
    - **O1 — Visualización con papel EKG**: Mostrar las señales de ECG con cuadrícula realista  
      (menor **0.04 s / 0.1 mV**, mayor **0.20 s / 0.5 mV**; velocidad **25 mm/s**, ganancia **10 mm/mV**).
    - **O2 — Frecuencia cardíaca**: Detectar **picos R** (neurokit2), calcular **FC instantánea** y **alertar** si está fuera de rango.
    - **O3 — Clasificación en 4 ritmos** (**Sinus Bradycardia, Sinus Rhythm, Atrial Fibrillation, Sinus Tachycardia**).
    """)

    st.subheader("🧾 ¿Qué datos se utilizan?")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Registros disponibles", len(records))
    with colB:
        st.metric("Derivaciones (máx)", 12)
    with colC:
        st.metric("Velocidad del papel", "25 mm/s")

    if record0 is not None:
        st.markdown("**Ejemplo de registro (primero encontrado):**")
        st.write({
            "Frecuencia de muestreo": f"{fs0} Hz" if fs0 else "—",
            "Nº derivaciones": int(signal0.shape[1]) if signal0 is not None else "—",
            "Derivaciones": sig_names0[:12] if sig_names0 else "—",
            "Directorio de datos": "./data (parejas .hea/.mat)"
        })
    else:
        st.info("No se pudieron leer metadatos del primer registro. Verifica que .hea y .mat estén completos.")

    st.subheader("🧭 Cómo usar la aplicación")
    st.markdown("""
    1. Entra a la pestaña **📊 Explorador** (menú lateral).
    2. Elige el **registro** y la **derivación** (por defecto se sugiere **II**).
    3. Activa **Papel EKG** para ver la cuadrícula estándar.
    4. Revisa las **KPIs**: FC media (lpm), latidos detectados y RR medio (s).
    5. En la pestaña **Frecuencia cardíaca**, explora la serie de FC y descarga **CSV**.
    6. En **Clasificación**, ejecuta la **demo** del modelo (requiere `models/best_ecgnet.pt`).
    """)

    st.subheader("📚 Fuente de los datos")
    st.markdown("""
    Los registros utilizados en esta plataforma provienen de **[PhysioNet](https://physionet.org/lightwave/?db=ecg-arrhythmia/1.0.0)**,  
    un repositorio internacional de acceso abierto que recopila bases de datos biomédicas para investigación y educación.
    
    En particular, se emplea el conjunto **ECG Arrhythmia Database (v1.0.0)**, que contiene electrocardiogramas con distintas condiciones cardíacas.  
    Esto permite **explorar ritmos normales y anómalos** en un entorno interactivo y educativo.
    """)
    
    st.caption("⚠️ Importante: este recurso es de carácter educativo y no reemplaza criterio ni diagnóstico clínico profesional.")

    st.subheader("👨‍💻 Equipo / Autoría")
    st.markdown("""
    Esta herramienta se desarrolló con foco en pedagogía, visualización y análisis reproducible con los siguientes miembros:

    - **Mercedes Díaz Pichiule**  
    Bachiller en Ingeniería Informática – Pontificia Universidad Católica del Perú

    - **Ángel Mayta Coaguila**  
    Ingeniero Civil - Universidad Alas Peruanas

    - **Miguel Lescano Avalos**  
    Bachiller en Ingeniería de Sistemas - Universidad Nacional de Ingeniería

    - **Sun Ji Sánchez**  
    Bachiller en Ingeniería Informática – Pontificia Universidad Católica del Perú

    **Fecha de publicación:** Abril 2025  
    **Ubicación:** Lima, Perú
    """)

# =====================================================================
#                             EXPLORADOR
# =====================================================================
elif selected == "📊 Explorador":
    st.title("📊 Explorador de ECG")

    # ---------- Sidebar de opciones específicas del explorador ----------
    with st.sidebar:
        st.header("Opciones del explorador")
        rec_base = st.selectbox("Registro", records, index=0, key="rec_explorer")
        lead_names_default = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

        # Carga registro
        record, signal, fs, sig_names = load_wfdb_record(rec_base)
        if len(sig_names) != signal.shape[1]:
            sig_names = lead_names_default[:signal.shape[1]]

        default_idx = sig_names.index('II') if 'II' in sig_names else 0
        lead_name = st.selectbox("Derivación", sig_names, index=default_idx, key="lead_explorer")
        lead_idx = sig_names.index(lead_name)

        seconds = st.slider("Segundos a mostrar", min_value=2, max_value=10, value=5, key="secs_explorer")
        show_grid = st.checkbox("Mostrar papel EKG", value=True, key="grid_explorer")
        clean_toggle = st.checkbox("Limpiar señal (neurokit2.ecg_clean)", value=True, key="clean_explorer")
        range_ok = st.slider("Rango normal FC (lpm)", 40, 140, (60,100), key="range_explorer")

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

    # ------------------ Alerta y KPIs ------------------
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

    # ------------------ Tabs del explorador ------------------
    tab_vis, tab_hr, tab_cls = st.tabs(["👀 Visor", "📈 Frecuencia cardíaca", "🧠 Clasificación"])

    # ========= Tab Visor =========
    with tab_vis:
        N = int(seconds * fs)
        t = np.arange(len(ecg_proc)) / fs
        t_win, y_win = t[:N], ecg_proc[:N]
        r_win = r_idx[(r_idx >= 0) & (r_idx < N)]

        # Figura principal (WebGL para fluidez)
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

        # with st.expander("Metadatos del registro"):
        #     st.write({
        #         "Frecuencia de muestreo (Hz)": fs,
        #         "Nº muestras (lead seleccionado)": int(len(ecg)),
        #         "Nº derivaciones": int(signal.shape[1]),
        #         "Derivaciones": sig_names
        #     })

        # ---- Resumen bajo el gráfico: pico máximo y amplitud pico-a-pico en la ventana ----
        if np.isfinite(y_win).any():
            # Pico máximo (positivo) en la ventana
            i_max = int(np.nanargmax(y_win))
            t_max = float(t_win[i_max])
            y_max = float(y_win[i_max])
            # Amplitud pico-a-pico (útil para calibración/ganancia y ruido)
            y_min = float(np.nanmin(y_win))
            p2p = y_max - y_min
            st.success(f"**Pico máximo** en ventana: {y_max:.3f} mV a **t = {t_max:.3f} s** · "
                       f"**Amplitud pico-a-pico**: {p2p:.3f} mV")
        else:
            st.warning("🔎 No hay datos válidos en la ventana seleccionada.")
        
        st.caption("Reglas del papel: menor=0.04 s / 0.1 mV, mayor=0.20 s / 0.5 mV. Velocidad=25 mm/s, Ganancia=10 mm/mV.")

    # ========= Tab Frecuencia Cardíaca =========
    with tab_hr:
        st.markdown("**Método:** detección de picos R (neurokit2) → FC instantánea por inverso del RR.")
        N = int(seconds * fs)
        t = np.arange(len(ecg_proc)) / fs
        hr_win = hr_inst[:N] if hr_inst is not None else None

        if hr_win is not None and np.isfinite(hr_win).any():
            fig2 = go.Figure()
            fig2.add_trace(go.Scattergl(x=t[:N], y=hr_win, mode="lines", name="FC (lpm)",
                                        hovertemplate="t=%{x:.2f} s<br>FC=%{y:.1f} lpm<extra></extra>"))
            fig2.add_hrect(y0=range_ok[0], y1=range_ok[1], fillcolor="#e8f5e9", opacity=0.45, line_width=0)
            fig2.update_layout(margin=dict(l=20,r=20,t=40,b=10), xaxis_title="Tiempo (s)", yaxis_title="FC (lpm)", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                df_hr = pd.DataFrame({"t_s": t[:N], "HR_lpm": hr_win})
                st.download_button("⬇️ Descargar FC (CSV)", data=df_hr.to_csv(index=False).encode(),
                                   file_name=f"{os.path.basename(rec_base)}_HR.csv", mime="text/csv")
            with col2:
                df_rr = pd.DataFrame({"RR_s": rr_s}) if rr_s.size else pd.DataFrame({"RR_s":[]})
                st.download_button("⬇️ Descargar RR (CSV)", data=df_rr.to_csv(index=False).encode(),
                                   file_name=f"{os.path.basename(rec_base)}_RR.csv", mime="text/csv")
        else:
            st.warning("No se pudo calcular la serie de FC. Prueba otra derivación o habilita limpieza de señal.")

        # ---- Conclusión dinámica sobre la FC media vs rango ----
        if np.isnan(hr_mean):
            st.info("ℹ️ No se pudo calcular la **frecuencia cardíaca promedio** en esta ventana.")
        else:
            lo, hi = range_ok
            if hr_mean < lo:
                st.warning(f"🧭 Conclusión: la **FC promedio ({hr_mean:.1f} lpm)** está **por debajo** del rango [{lo}, {hi}] lpm.")
            elif hr_mean > hi:
                st.warning(f"🧭 Conclusión: la **FC promedio ({hr_mean:.1f} lpm)** está **por encima** del rango [{lo}, {hi}] lpm.")
            else:
                st.success(f"🧭 Conclusión: la **FC promedio ({hr_mean:.1f} lpm)** está **dentro** del rango [{lo}, {hi}] lpm.")

    # ========= Tab Clasificación =========
with tab_cls:
    st.markdown("Clasificación automática (4 clases) — *demo educativa*")

    # Mapeo de etiquetas (internas -> visual en español)
    LABEL_NAMES = ["Sinus Bradycardia", "Sinus Rhythm", "Atrial Fibrillation", "Sinus Tachycardia"]
    LABELS_ES = {
        "Sinus Bradycardia": "Bradicardia sinusal",
        "Sinus Rhythm": "Ritmo sinusal",
        "Atrial Fibrillation": "Fibrilación auricular",
        "Sinus Tachycardia": "Taquicardia sinusal",
    }
    # Orden en español alineado a LABEL_NAMES
    LABEL_NAMES_ES_ORDERED = [LABELS_ES[k] for k in LABEL_NAMES]

    if st.button("🔎 Clasificar este registro"):
        try:
            model, device = load_classifier()
            x = preprocess_12lead_for_model_from_base(rec_base)
            with torch.no_grad():
                logits = model(x.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()

            pred_idx = int(np.argmax(probs))
            pred_label_en = LABEL_NAMES[pred_idx]
            pred_label_es = LABELS_ES[pred_label_en]

            # Badge y nota contextual
            if pred_label_en == "Sinus Bradycardia":
                color = "badge-warn"; note = "Frecuencia baja — verificar contexto clínico."
            elif pred_label_en == "Sinus Tachycardia":
                color = "badge-warn"; note = "Frecuencia alta — verificar desencadenantes."
            elif pred_label_en == "Atrial Fibrillation":
                color = "badge-alert"; note = "Ritmo irregular — evaluar trazado completo."
            else:
                color = "badge-ok"; note = "Ritmo sinusal."

            st.markdown(
                f'<span class="badge {color}">Predicción: <b>{pred_label_es}</b></span> '
                f'<span class="badge">{note}</span>',
                unsafe_allow_html=True
            )

            # Explicación breve en español por clase predicha
            EXPLAIN_ES = {
                "Sinus Bradycardia": (
                    "Bradicardia sinusal: ritmo sinusal con **frecuencia baja**. "
                    "Puede ser fisiológica (deportistas, descanso) o por fármacos/hipotiroidismo; "
                    "valorar **síntomas** (mareos, síncope) y contexto clínico."
                ),
                "Sinus Rhythm": (
                    "Ritmo sinusal: actividad auricular normal con onda P positiva y relación P–QRS 1:1. "
                    "Frecuencia acorde al contexto clínico."
                ),
                "Atrial Fibrillation": (
                    "Fibrilación auricular: **ritmo irregular** sin ondas P identificables; "
                    "riesgo de **tromboembolismo**. Requiere valorar anticoagulación y control de frecuencia/ritmo."
                ),
                "Sinus Tachycardia": (
                    "Taquicardia sinusal: ritmo sinusal con **frecuencia alta**. "
                    "Suele ser respuesta a fiebre, dolor, hipovolemia, ansiedad o fármacos; "
                    "buscar y tratar la **causa subyacente**."
                ),
            }
            st.warning(f"**Interpretación breve:** {EXPLAIN_ES.get(pred_label_en, 'Interpretación no disponible.')}")

            # Barras con etiquetas en español
            dfp = pd.DataFrame({
                "Clase (ES)": LABEL_NAMES_ES_ORDERED,
                "Probabilidad": probs
            })
            figp = go.Figure(go.Bar(
                x=dfp["Clase (ES)"],
                y=dfp["Probabilidad"],
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="outside"
            ))
            figp.update_yaxes(range=[0, 1.0])
            figp.update_layout(
                margin=dict(l=20, r=20, t=40, b=40),
                yaxis_title="Probabilidad",
                xaxis_title=""
            )
            st.plotly_chart(figp, use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error durante la clasificación: {e}")
