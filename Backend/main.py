
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

EEG_LEN = 178


st.set_page_config(
    page_title="NeuroSense | EEG Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 1.6rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102A43, #176B70);
        color: white;
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #D9EAF0;
        margin-bottom: 0;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 650;
        margin: 0.5rem 0 0.8rem 0;
    }

    .notice {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background: #FFF4D6;
        color: #674D00;
        margin: 1rem 0;
        font-size: 0.9rem;
    }

    .result-card {
        padding: 1.2rem;
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 14px;
        min-height: 130px;
    }

    .result-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.6rem;
    }

    .result-value {
        font-size: 1.6rem;
        font-weight: 700;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def parse_eeg(raw: bytes):
    """Parse a headerless EEG CSV/TXT containing 178 values."""
    try:
        text = raw.decode("utf-8-sig").strip()

        if not text:
            return None, "The uploaded file is empty."

        normalized = text.replace(",", " ").replace(";", " ")
        tokens = normalized.split()

        if len(tokens) != EEG_LEN:
            return (
                None,
                f"Expected exactly {EEG_LEN} numeric values, "
                f"but found {len(tokens)}.",
            )

        values = np.asarray(
            [float(token) for token in tokens],
            dtype=np.float32,
        )

        if not np.isfinite(values).all():
            return None, "The file contains NaN or infinite values."

        return values, None

    except (ValueError, UnicodeDecodeError):
        return (
            None,
            "The file must contain numeric EEG values only. "
            "Remove headers and non-numeric text.",
        )


def build_signal_chart(values):
    """Create a visualization of the uploaded EEG values."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(values) + 1)),
            y=values,
            mode="lines",
            name="EEG signal",
            line=dict(color="#168A8A", width=1.8),
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis_title="Sample index",
        yaxis_title="Signal value",
        template="plotly_white",
        showlegend=False,
    )

    return fig


def call_health_endpoint(api_url):
    """Check whether the deployed FastAPI backend is reachable."""
    response = requests.get(
        f"{api_url}/health",
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError("Unexpected health response.")

    return data


def call_prediction_endpoint(api_url, file_name, eeg_bytes):
    """Send EEG data using the field expected by FastAPI."""
    response = requests.post(
        f"{api_url}/predict",
        files={
            "eeg": (
                file_name,
                eeg_bytes,
                "text/plain",
            )
        },
        timeout=120,
    )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"Backend returned HTTP {response.status_code}: {detail}"
        )

    result = response.json()

    if not isinstance(result, dict):
        raise ValueError("The backend returned an invalid response.")

    if "prediction" not in result or "anomaly_probability" not in result:
        raise ValueError(
            "The backend response is missing prediction fields."
        )

    probability = float(result["anomaly_probability"])

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "The backend returned an invalid anomaly probability."
        )

    return result


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🧠 NeuroSense")
    st.caption("EEG Research Dashboard")

    st.divider()

    st.markdown("### Backend settings")

    api_url = st.text_input(
        "FastAPI backend URL",
        value=DEFAULT_API,
        help=(
            "Enter your public Render URL in production. "
            "Use localhost only when running the backend locally."
        ),
    ).strip().rstrip("/")

    check_backend = st.button(
        "Check backend connection",
        use_container_width=True,
    )

    if check_backend:
        if not api_url.startswith(("http://", "https://")):
            st.error("Enter a valid URL beginning with http:// or https://.")
        else:
            try:
                with st.spinner("Checking backend..."):
                    health = call_health_endpoint(api_url)

                st.success("Backend is reachable.")

                if health.get("model_loaded") is True:
                    st.success("Model loaded successfully.")
                elif health.get("model_loaded") is False:
                    st.error("Backend is running, but the model is not loaded.")
                else:
                    st.info(f"Health response: {health}")

            except requests.RequestException as exc:
                st.error(f"Backend connection failed: {exc}")
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

    st.divider()

    st.markdown("### Input requirements")
    st.write("- Headerless CSV or TXT")
    st.write("- Exactly 178 numeric values")
    st.write("- One EEG record per analysis")

    st.divider()

    st.caption("NeuroSense • Research prototype")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🧠 NeuroSense</h1>
        <p>EEG Signal Analysis & Machine Learning Research</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        <strong>Research prototype:</strong>
        This application demonstrates machine-learning classification
        on EEG input. Its predictions are not a medical diagnosis,
        a validated clinical risk assessment, or a substitute for
        evaluation by a qualified healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "result" not in st.session_state:
    st.session_state["result"] = None

if "input_signature" not in st.session_state:
    st.session_state["input_signature"] = None


# =========================================================
# EEG INPUT
# =========================================================

st.markdown(
    '<div class="section-title">01 · Upload EEG recording</div>',
    unsafe_allow_html=True,
)

st.write(
    "Upload a headerless CSV or TXT file containing exactly "
    "178 numeric values for one EEG record."
)

uploaded_file = st.file_uploader(
    "Choose an EEG file",
    type=["csv", "txt"],
    key="eeg_upload",
)

eeg_values = None
eeg_error = None
eeg_bytes = None
file_signature = None

if uploaded_file is not None:
    eeg_bytes = uploaded_file.getvalue()

    file_signature = (
        uploaded_file.name,
        len(eeg_bytes),
        hash(eeg_bytes),
    )

    # Clear a result if the uploaded file has changed.
    if file_signature != st.session_state["input_signature"]:
        st.session_state["result"] = None
        st.session_state["input_signature"] = file_signature

    eeg_values, eeg_error = parse_eeg(eeg_bytes)

    if eeg_error:
        st.error(eeg_error)
    else:
        st.success(
            f"Input validated: {len(eeg_values)} numeric values."
        )

        with st.expander("Preview EEG signal", expanded=True):
            st.plotly_chart(
                build_signal_chart(eeg_values),
                use_container_width=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Number of values",
                    len(eeg_values),
                )

            with col2:
                st.metric(
                    "Minimum value",
                    f"{float(np.min(eeg_values)):.3f}",
                )

            with col3:
                st.metric(
                    "Maximum value",
                    f"{float(np.max(eeg_values)):.3f}",
                )

else:
    st.session_state["result"] = None
    st.session_state["input_signature"] = None

    st.info(
        "Upload an EEG file to preview the signal and run the model."
    )


# =========================================================
# ANALYSIS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">02 · Run analysis</div>',
    unsafe_allow_html=True,
)

st.write(
    "The uploaded record will be sent to your FastAPI backend "
    "for prediction using the already-trained model."
)

run_analysis = st.button(
    "🔍 Run EEG analysis",
    type="primary",
    use_container_width=True,
    disabled=(
        uploaded_file is None
        or eeg_error is not None
        or eeg_values is None
        or not api_url.startswith(("http://", "https://"))
    ),
)

if run_analysis and uploaded_file is not None and eeg_bytes is not None:
    try:
        with st.spinner(
            "Analysing EEG recording... This may take a moment."
        ):
            result = call_prediction_endpoint(
                api_url,
                uploaded_file.name,
                eeg_bytes,
            )

        st.session_state["result"] = result
        st.success("Analysis completed successfully.")

    except requests.Timeout:
        st.error(
            "The backend request timed out. If Render is waking "
            "from inactivity, wait briefly and try again."
        )

    except requests.RequestException as exc:
        st.error(f"Could not connect to the backend: {exc}")

    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))


# =========================================================
# RESULTS
# =========================================================

result = st.session_state.get("result")

if result:
    st.divider()

    st.markdown(
        '<div class="section-title">03 · Prediction results</div>',
        unsafe_allow_html=True,
    )

    prediction = str(result["prediction"])
    probability = float(result["anomaly_probability"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Model prediction</div>
                <div class="result-value">{prediction}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Model anomaly score</div>
                <div class="result-value">{probability * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("Anomaly probability reported by the model")
    st.progress(probability)

    st.caption(
        "This score represents the model output as returned by the "
        "backend. It is not necessarily a calibrated probability of "
        "disease or a measure of clinical risk."
    )

    with st.expander("View technical response"):
        st.json(result)

    if st.button("Clear results"):
        st.session_state["result"] = None
        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "NeuroSense | EEG machine-learning research prototype | "
    "Predictions require appropriate validation and clinical context."
)

