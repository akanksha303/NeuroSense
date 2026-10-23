
"""
NeuroSense — AI Neurological Screening Dashboard
Run: streamlit run Frontend/streamlit_app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

EEG_LEN = 178
DEFAULT_API = os.getenv("NEUROSENSE_API", "http://localhost:8000")

st.set_page_config(
    page_title="NeuroSense | AI Brain Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- colorful UI theme ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 0% 0%, rgba(91, 72, 220, .14), transparent 36%),
            radial-gradient(ellipse at 100% 10%, rgba(0, 190, 210, .12), transparent 32%),
            #F5F7FF;
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #18234A;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151A3D 0%, #25265A 55%, #343477 100%);
    }

    [data-testid="stSidebar"] * {
        color: #F2F4FF !important;
    }

    [data-testid="stSidebar"] input {
        color: #18234A !important;
        background: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-baseweb="slider"] div {
        border-color: #9A8CFF;
    }

    .hero {
        background: linear-gradient(120deg, #171B45 0%, #34317B 52%, #087EAD 100%);
        border-radius: 25px;
        padding: 32px 34px;
        color: white;
        margin: 4px 0 24px 0;
        box-shadow: 0 14px 35px rgba(45, 47, 113, .19);
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "🧠";
        position: absolute;
        right: 25px;
        top: 10px;
        font-size: 92px;
        opacity: .95;
        filter: drop-shadow(0 0 18px rgba(104, 224, 255, .3));
    }

    .hero-tag {
        display: inline-block;
        background: rgba(117, 255, 220, .15);
        color: #8DFFE4;
        padding: 6px 12px;
        border: 1px solid rgba(141, 255, 228, .35);
        border-radius: 30px;
        font-size: 12px;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 39px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 14px 0 6px 0;
        line-height: 1.15;
    }

    .hero-subtitle {
        color: #E0E7FF;
        font-size: 15px;
        max-width: 600px;
        line-height: 1.7;
    }

    .section-card {
        background: rgba(255,255,255,.92);
        border: 1px solid #E5E9FA;
        border-radius: 19px;
        padding: 21px;
        box-shadow: 0 7px 23px rgba(41, 54, 105, .055);
        margin-bottom: 12px;
    }

    .section-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: #737B9D;
        text-transform: uppercase;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #222950;
        font-size: 21px;
        font-weight: 700;
        margin: 5px 0 8px 0;
    }

    .pill {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 11px;
        font-size: 12px;
        font-weight: 700;
        margin: 5px 0 10px 0;
    }

    .purple-pill { background: #EEE9FF; color: #6941C6; }
    .cyan-pill { background: #DDF8FC; color: #087D96; }
    .green-pill { background: #DDF8EA; color: #147D52; }

    .metric-card {
        border-radius: 18px;
        padding: 20px;
        background: white;
        border: 1px solid #E8EAF7;
        box-shadow: 0 8px 22px rgba(41, 54, 105, .06);
        min-height: 120px;
    }

    .metric-label {
        color: #707895;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #24265B;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 700;
        margin-top: 7px;
    }

    .risk-panel {
        background: linear-gradient(145deg, #FFFFFF, #F0EDFF);
        border: 1px solid #DED8FF;
        border-radius: 22px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 26px rgba(77, 55, 160, .08);
    }

    .risk-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 46px;
        font-weight: 700;
        margin: 8px 0;
    }

    .notice {
        background: #FFF6E7;
        border: 1px solid #F6D8A6;
        border-radius: 12px;
        padding: 12px 16px;
        color: #80551B;
        font-size: 13px;
        margin: 12px 0 22px 0;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #6654DB, #178FBB);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 700;
        padding: 12px 22px;
        min-height: 48px;
        box-shadow: 0 7px 17px rgba(81, 79, 198, .2);
    }

    .stButton > button {
        border-radius: 10px;
    }

    div[data-testid="stFileUploader"] {
        background: #F8F9FF;
        border: 1px dashed #B7B8E9;
        border-radius: 13px;
        padding: 7px;
    }

    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #7964E8, #17B7C7);
    }

    .footer {
        color: #858BA6;
        text-align: center;
        font-size: 12px;
        padding: 22px 0 8px 0;
    }

    @media (max-width: 700px) {
        .hero { padding: 24px 20px; }
        .hero-title { font-size: 29px; }
        .hero:after { font-size: 55px; right: 12px; top: 8px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- helpers ----------
def parse_eeg(raw: bytes):
    try:
        text = raw.decode("utf-8", errors="ignore")
        return [
            float(value)
            for value in text.replace(",", " ").replace(";", " ").split()
        ]
    except ValueError:
        return None


def risk_level(probability: float):
    if probability >= 0.70:
        return "High", "#D04456", "#FCE8EC"
    if probability >= 0.40:
        return "Moderate", "#C18416", "#FFF2D8"
    return "Low", "#14865C", "#DDF8EA"


def metric_card(label, value, emoji):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{emoji} &nbsp; {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_risk(probability: float):
    level, color, background = risk_level(probability)
    percent = min(max(float(probability) * 100, 0), 100)

    st.markdown(
        f"""
        <div class="risk-panel">
            <div class="section-label">Combined screening score</div>
            <div class="risk-number" style="color:{color}">
                {percent:.1f}%
            </div>
            <div style="display:inline-block; background:{background};
                        color:{color}; padding:7px 16px;
                        border-radius:20px; font-weight:700;">
                ● {level} risk category
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(percent / 100)
    st.caption(
        "This score reflects the configured model output and is not a "
        "validated estimate of clinical risk."
    )


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 🧠 NeuroSense")
    st.caption("AI Brain Analytics")
    st.divider()

    st.markdown("### ⚙️ Configuration")
    api_url = st.text_input("FastAPI backend URL", DEFAULT_API).rstrip("/")

    alpha = st.slider(
        "EEG fusion weight (α)",
        0.0,
        1.0,
        0.5,
        0.05,
        help="Used only when both EEG and MRI files are supplied.",
    )

    st.caption(f"EEG weight: {alpha:.0%}")
    st.caption(f"MRI weight: {1 - alpha:.0%}")

    st.divider()

    if st.button("🔌 Check backend", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            st.success("Backend is responding")
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")
        except ValueError:
            st.error("Backend returned invalid JSON.")


# ---------- hero ----------
st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">✦ INTELLIGENT NEUROLOGY</div>
        <div class="hero-title">NeuroSense</div>
        <div class="hero-subtitle">
            Explore neurological screening signals with an AI-powered
            dashboard for EEG recordings and MRI images.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        ⚕️ <b>Research prototype:</b> Outputs are experimental and must not
        be used to diagnose, rule out, or treat a medical condition.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- quick overview ----------
c1, c2, c3 = st.columns(3)

with c1:
    metric_card("EEG analysis", "178 values", "〰️")

with c2:
    metric_card("MRI input", "Brain image", "🧠")

with c3:
    metric_card("Inference", "FastAPI", "⚡")

st.write("")
st.markdown("## 🔬 Start an analysis")
st.caption("Upload one modality or provide both for a combined analysis.")

# ---------- upload cards ----------
col_eeg, col_mri = st.columns(2)

eeg_bytes = None
mri_bytes = None
eeg_ok = True

with col_eeg:
    st.markdown(
        """
        <div class="section-card">
            <div class="pill purple-pill">01 · SIGNAL PROCESSING</div>
            <div class="section-title">〰️ EEG Recording</div>
            <div style="color:#707895;font-size:14px;">
                Upload a CSV or TXT file containing exactly 178 numeric values.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eeg_file = st.file_uploader(
        "Choose EEG file",
        type=["csv", "txt"],
        key="eeg",
    )

    if eeg_file is not None:
        eeg_bytes = eeg_file.getvalue()
        values = parse_eeg(eeg_bytes)

        if values is None or len(values) != EEG_LEN:
            count = 0 if values is None else len(values)
            st.error(f"Expected 178 numeric values; found {count}.")
            eeg_ok = False
        else:
            st.success(f"✓ {len(values)} numeric values loaded")
            st.markdown("**EEG signal preview**")
            st.line_chart(
                pd.DataFrame({"Signal amplitude": values}),
                height=200,
            )

with col_mri:
    st.markdown(
        """
        <div class="section-card">
            <div class="pill cyan-pill">02 · IMAGE ANALYSIS</div>
            <div class="section-title">🧠 MRI Scan</div>
            <div style="color:#707895;font-size:14px;">
                Upload a supported JPG or PNG brain image for model inference.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mri_file = st.file_uploader(
        "Choose MRI image",
        type=["jpg", "jpeg", "png"],
        key="mri",
    )

    if mri_file is not None:
        mri_bytes = mri_file.getvalue()
        st.image(
            mri_bytes,
            caption="Uploaded image",
            use_container_width=True,
        )

# ---------- selected mode ----------
if eeg_bytes is not None and mri_bytes is not None:
    mode = "EEG + MRI fusion"
    mode_color = "#6A4CDA"
elif eeg_bytes is not None:
    mode = "EEG only"
    mode_color = "#8B55CC"
elif mri_bytes is not None:
    mode = "MRI only"
    mode_color = "#078BA8"
else:
    mode = None
    mode_color = "#747B98"

st.markdown(
    f"""
    <div class="section-card">
        <div class="section-label">Selected analysis mode</div>
        <div style="font-size:19px;font-weight:700;color:{mode_color};
                    margin-top:7px;">
            {mode or "Waiting for an upload"}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- analysis ----------
run = st.button(
    "✨  Run AI Analysis",
    type="primary",
    use_container_width=True,
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
        with st.spinner("🧠 Processing your uploaded data..."):
            response = requests.post(
                f"{api_url}/predict",
                files=files,
                data={"alpha": alpha},
                timeout=60,
            )

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
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
            f"Could not reach the backend at {api_url}. "
            f"Details: {exc}"
        )


# ---------- results ----------
result = st.session_state.get("result")

if result:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    st.caption("Model output from the connected FastAPI backend.")

    left, right = st.columns([0.9, 1.1])

    with left:
        show_risk(float(result["final_risk"]))

        if result.get("mode") == "fused":
            st.info(f"Fusion weight α = {result['alpha']:.2f}")

    with right:
        st.markdown(
            """
            <div class="section-label">INDIVIDUAL MODEL OUTPUTS</div>
            """,
            unsafe_allow_html=True,
        )

        if "eeg" in result:
            probability = float(result["eeg"]["probability"])

            st.markdown(
                '<div class="section-card">'
                '<div class="pill purple-pill">EEG MODEL</div>'
                '<div class="section-title">〰️ Signal analysis</div>',
                unsafe_allow_html=True,
            )

            st.metric(
                "EEG anomaly probability",
                f"{probability * 100:.1f}%",
            )
            st.progress(min(max(probability, 0.0), 1.0))
            st.markdown("</div>", unsafe_allow_html=True)

        if "mri" in result:
            mri_result = result["mri"]

            st.markdown(
                '<div class="section-card">'
                '<div class="pill cyan-pill">MRI MODEL</div>'
                '<div class="section-title">🧠 Image classification</div>',
                unsafe_allow_html=True,
            )

            st.metric("Model label", str(mri_result["label"]))

            probabilities = mri_result.get("probs", {})

            if isinstance(probabilities, dict):
                chart_df = pd.DataFrame(
                    {
                        "Class": list(probabilities.keys()),
                        "Probability": list(probabilities.values()),
                    }
                ).set_index("Class")
            else:
                chart_df = pd.DataFrame(
                    {"Probability": probabilities}
                )

            st.bar_chart(
                chart_df,
                horizontal=True,
                height=220,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🧾 View raw API response"):
        st.json(result)

    st.caption(
        "Clinical interpretation requires qualified healthcare professionals "
        "and appropriate validated diagnostic procedures."
    )

# ---------- footer ----------
st.markdown(
    """
    <div class="footer">
        NeuroSense · AI-assisted neurological research prototype<br>
        Built with Streamlit, FastAPI and machine learning
    </div>
    """,
    unsafe_allow_html=True,
)

