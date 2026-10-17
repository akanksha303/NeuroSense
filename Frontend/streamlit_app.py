"""
NeuroSense Streamlit dashboard. Talks to the FastAPI backend (Backend/main.py).

Run:  streamlit run streamlit_app.py
Env:  NEUROSENSE_API=http://localhost:8000
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

EEG_LEN = 178
DEFAULT_API = os.getenv("NEUROSENSE_API", "http://localhost:8000")

st.set_page_config(page_title="NeuroSense", page_icon="🧠", layout="wide")

# ---------- helpers ----------
def parse_eeg(raw: bytes):
    try:
        text = raw.decode("utf-8", errors="ignore")
        return [float(v) for v in text.replace(",", " ").replace(";", " ").split()]
    except ValueError:
        return None

def risk_level(p: float):
    if p >= 0.70: return "High", "#C0402B"
    if p >= 0.40: return "Moderate", "#C98A12"
    return "Low", "#2F8F5B"

def gauge(p: float):
    level, color = risk_level(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p * 100, 1),
        number={"suffix": "%"}, title={"text": f"{level} risk"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 40], "color": "#E3F1E9"},
                         {"range": [40, 70], "color": "#F6EBCF"},
                         {"range": [70, 100], "color": "#F3D9D3"}]},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("Backend URL", DEFAULT_API)
    alpha = st.slider("EEG weight (α) in fusion", 0.0, 1.0, 0.5, 0.05,
                      help="Final risk = α × EEG + (1 − α) × MRI. Used only when both files are uploaded.")
    st.caption(f"MRI weight: {1 - alpha:.2f}")
    if st.button("Check backend"):
        try:
            st.json(requests.get(f"{api_url}/health", timeout=5).json())
        except requests.RequestException as e:
            st.error(f"Backend unreachable: {e}")

# ---------- header ----------
st.title("NeuroSense")
st.write("Screen EEG recordings and MRI scans for neurological risk, separately or together.")
st.warning("Research prototype. Not a medical device and not a substitute for clinical diagnosis.")

# ---------- inputs ----------
col_eeg, col_mri = st.columns(2)
eeg_bytes = mri_bytes = None
eeg_ok = True

with col_eeg:
    st.subheader("EEG recording")
    eeg_file = st.file_uploader("CSV or TXT with 178 voltage values", type=["csv", "txt"], key="eeg")
    if eeg_file:
        eeg_bytes = eeg_file.getvalue()
        vals = parse_eeg(eeg_bytes)
        if vals is None:
            st.error("File must contain only numbers.")
            eeg_ok = False
        else:
            if len(vals) != EEG_LEN:
                st.error(f"Found {len(vals)} values; the model expects {EEG_LEN}.")
                eeg_ok = False
            st.line_chart(pd.DataFrame({"voltage": vals}), height=200)

with col_mri:
    st.subheader("MRI scan")
    mri_file = st.file_uploader("JPG or PNG, single brain slice", type=["jpg", "jpeg", "png"], key="mri")
    if mri_file:
        mri_bytes = mri_file.getvalue()
        st.image(mri_bytes, use_container_width=True)

# ---------- mode ----------
if eeg_bytes and mri_bytes:
    mode = "EEG + MRI fusion"
elif eeg_bytes:
    mode = "EEG only"
elif mri_bytes:
    mode = "MRI only"
else:
    mode = None

st.markdown(f"**Mode:** {mode or 'Upload an EEG file, an MRI scan, or both'}")

# ---------- run ----------
run = st.button("Run analysis", type="primary", disabled=(mode is None or not eeg_ok))

if run:
    files = {}
    if eeg_bytes: files["eeg"] = (eeg_file.name, eeg_bytes, "text/csv")
    if mri_bytes: files["mri"] = (mri_file.name, mri_bytes, mri_file.type)
    try:
        with st.spinner("Analysing…"):
            r = requests.post(f"{api_url}/predict", files=files, data={"alpha": alpha}, timeout=60)
        if r.status_code != 200:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
            st.error(f"Analysis failed: {detail}")
        else:
            st.session_state["result"] = r.json()
    except requests.RequestException as e:
        st.error(f"Could not reach the backend at {api_url}. Is it running? ({e})")

# ---------- results ----------
res = st.session_state.get("result")
if res:
    st.divider()
    left, right = st.columns([1, 1.4])
    with left:
        st.plotly_chart(gauge(res["final_risk"]), use_container_width=True)
        if res["mode"] == "fused":
            st.caption(f"Fused with α = {res['alpha']:.2f}")
    with right:
        if "eeg" in res:
            p = res["eeg"]["probability"]
            st.metric("EEG anomaly probability", f"{p * 100:.1f}%")
            st.progress(min(max(p, 0.0), 1.0))
        if "mri" in res:
            st.metric("MRI classification", res["mri"]["label"])
            probs = res["mri"]["probs"]
            df = pd.DataFrame({"probability": probs}).rename(index={"notumor": "no tumor"})
            st.bar_chart(df, horizontal=True, height=220)
    with st.expander("Raw response"):
        st.json(res)