
"""
NeuroSense Streamlit dashboard.
Talks to the FastAPI backend (Backend/main.py).

Run: streamlit run Frontend/streamlit_app.py
Env: NEUROSENSE_API=http://localhost:8000
"""

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

EEG_LEN = 178
DEFAULT_API = os.getenv("NEUROSENSE_API", "http://localhost:8000")

st.set_page_config(
    page_title="NeuroSense",
    page_icon="🧠",
    layout="wide",
)


# ---------- helpers ----------
def parse_eeg(raw: bytes):
    try:
        text = raw.decode("utf-8", errors="ignore")
        return [
            float(v)
            for v in text.replace(",", " ").replace(";", " ").split()
        ]
    except ValueError:
        return None


def risk_level(p: float):
    if p >= 0.70:
        return "High", "#C0402B"
    if p >= 0.40:
        return "Moderate", "#C98A12"
    return "Low", "#2F8F5B"


def show_risk(p: float):
    level, color = risk_level(p)

    st.markdown(f"### {level} risk")
    st.markdown(f"## {p * 100:.1f}%")
    st.progress(min(max(float(p), 0.0), 1.0))

    st.caption(
        "Prototype risk score, not a clinical diagnosis."
    )


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")

    api_url = st.text_input("Backend URL", DEFAULT_API).rstrip("/")

    alpha = st.slider(
        "EEG weight (α) in fusion",
        0.0,
        1.0,
        0.5,
        0.05,
        help=(
            "Final risk = α × EEG + (1 − α) × MRI. "
            "Used only when both files are uploaded."
        ),
    )

    st.caption(f"MRI weight: {1 - alpha:.2f}")

    if st.button("Check backend"):
        try:
            response = requests.get(
                f"{api_url}/health", timeout=5
            )
            response.raise_for_status()
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Backend unreachable: {exc}")
        except ValueError:
            st.error("Backend returned an invalid JSON response.")


# ---------- header ----------
st.title("🧠 NeuroSense")
st.write(
    "Screen EEG recordings and MRI scans separately or together."
)
st.warning(
    "Research prototype only. Not a medical device or a substitute "
    "for professional clinical diagnosis."
)


# ---------- inputs ----------
col_eeg, col_mri = st.columns(2)

eeg_bytes = None
mri_bytes = None
eeg_ok = True

with col_eeg:
    st.subheader("EEG recording")

    eeg_file = st.file_uploader(
        "CSV or TXT with 178 voltage values",
        type=["csv", "txt"],
        key="eeg",
    )

    if eeg_file is not None:
        eeg_bytes = eeg_file.getvalue()
        vals = parse_eeg(eeg_bytes)

        if vals is None or len(vals) != EEG_LEN:
            count = 0 if vals is None else len(vals)
            st.error(
                f"Expected exactly {EEG_LEN} numeric values; "
                f"found {count}."
            )
            eeg_ok = False
        else:
            st.success(f"Loaded {len(vals)} EEG values.")
            st.line_chart(
                pd.DataFrame({"voltage": vals}),
                height=200,
            )

with col_mri:
    st.subheader("MRI scan")

    mri_file = st.file_uploader(
        "JPG or PNG brain slice",
        type=["jpg", "jpeg", "png"],
        key="mri",
    )

    if mri_file is not None:
        mri_bytes = mri_file.getvalue()
        st.image(mri_bytes, use_container_width=True)


# ---------- mode ----------
if eeg_bytes is not None and mri_bytes is not None:
    mode = "EEG + MRI fusion"
elif eeg_bytes is not None:
    mode = "EEG only"
elif mri_bytes is not None:
    mode = "MRI only"
else:
    mode = None

st.markdown(
    f"**Mode:** {mode or 'Upload an EEG file, an MRI scan, or both'}"
)


# ---------- run analysis ----------
run = st.button(
    "Run analysis",
    type="primary",
    disabled=(mode is None or not eeg_ok),
)

if run:
    files = {}

    if eeg_bytes is not None:
        files["eeg"] = (
            eeg_file.name,
            eeg_bytes,
            "text/csv",
        )

    if mri_bytes is not None:
        files["mri"] = (
            mri_file.name,
            mri_bytes,
            mri_file.type,
        )

    try:
        with st.spinner("Analysing..."):
            response = requests.post(
                f"{api_url}/predict",
                files=files,
                data={"alpha": alpha},
                timeout=60,
            )

        if response.status_code != 200:
            try:
                detail = response.json().get(
                    "detail", response.text
                )
            except ValueError:
                detail = response.text

            st.error(f"Analysis failed: {detail}")
        else:
            try:
                st.session_state["result"] = response.json()
            except ValueError:
                st.error("Backend returned invalid JSON.")

    except requests.RequestException as exc:
        st.error(
            f"Could not reach backend at {api_url}. "
            f"Check whether it is running. Details: {exc}"
        )


# ---------- results ----------
res = st.session_state.get("result")

if res:
    st.divider()
    st.subheader("Analysis results")

    left, right = st.columns([1, 1.4])

    with left:
        show_risk(float(res["final_risk"]))

        if res.get("mode") == "fused":
            st.caption(
                f"Fused with α = {res['alpha']:.2f}"
            )

    with right:
        if "eeg" in res:
            probability = float(res["eeg"]["probability"])

            st.metric(
                "EEG anomaly probability",
                f"{probability * 100:.1f}%",
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

        if "mri" in res:
            st.metric(
                "MRI classification",
                res["mri"]["label"],
            )

            probabilities = res["mri"]["probs"]

            if isinstance(probabilities, dict):
                df = pd.DataFrame(
                    {
                        "Class": list(probabilities.keys()),
                        "Probability": list(
                            probabilities.values()
                        ),
                    }
                ).set_index("Class")
            else:
                df = pd.DataFrame(
                    {"Probability": probabilities}
                )

            st.bar_chart(
                df,
                horizontal=True,
                height=220,
            )

    with st.expander("Raw response"):
        st.json(res)
