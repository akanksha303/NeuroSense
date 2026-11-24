import json
import streamlit as st

st.set_page_config(
    page_title="NeuroSense — Multimodal Neuro & Tumor Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Design System & Custom Styling ---
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        h1, h2, h3, .brand-font {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            letter-spacing: -0.02em;
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(20, 184, 166, 0.08), transparent 36%),
                radial-gradient(circle at 90% 14%, rgba(249, 115, 22, 0.07), transparent 34%),
                linear-gradient(180deg, #f6fbfa 0%, #ffffff 42%, #fffaf6 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }
        .topbar-wrap {
            background: rgba(255, 255, 255, 0.86);
            backdrop-filter: blur(10px);
            border: 1px solid #e2eeec;
            border-radius: 16px;
            padding: 12px 18px;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 18px rgba(15, 65, 59, 0.04);
        }
        .pill-teal {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #e6f6f3;
            color: #0f665c;
            border: 1px solid #bfe7e0;
            font-weight: 600;
            font-size: 0.76rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 5px 12px;
            border-radius: 999px;
            margin-bottom: 12px;
        }
        .pill-coral {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #fff1e8;
            color: #c2521e;
            border: 1px solid #ffd5bc;
            font-weight: 600;
            font-size: 0.76rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 4px 11px;
            border-radius: 999px;
        }
        .status-chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .chip-high {
            background: #fee9e2;
            color: #b83b14;
            border: 1px solid #f9c4b3;
        }
        .chip-mod {
            background: #fef5e2;
            color: #9e650a;
            border: 1px solid #f7df9c;
        }
        .chip-low {
            background: #e4f5ef;
            color: #136c53;
            border: 1px solid #b5e3d3;
        }
        .telemetry-card {
            background: #ffffff;
            border: 1px solid #dcebe8;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 12px 30px rgba(19, 78, 71, 0.06);
        }
        .pillar-card {
            background: #ffffff;
            border: 1px solid #e4efed;
            border-top: 3px solid #0f766e;
            border-radius: 14px;
            padding: 18px;
            height: 100%;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
        }
        .pillar-card.coral-top {
            border-top-color: #ea580c;
        }
        .pillar-title {
            font-weight: 700;
            font-size: 1rem;
            color: #0f172a;
            margin-bottom: 6px;
        }
        .pillar-desc {
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 0;
        }
        .case-banner {
            background: #f8fcfb;
            border: 1px solid #d5ebe6;
            border-left: 4px solid #0f766e;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        div[data-testid="stMetric"] {
            background-color: #f9fdfc;
            border: 1px solid #e1efec;
            padding: 12px 14px;
            border-radius: 12px;
        }
        div[data-testid="stMetric"] label {
            color: #475569;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sample Cases Data ---
SAMPLE_CASES = {
    "NS-1042 (Temporal Asymmetry + Focal T1w)": {
        "patient_tag": "Case #NS-1042 · 54y",
        "eeg_prob": 0.89,
        "eeg_note": "Elevated theta-to-alpha ratio in temporal channels",
        "mri_prob": 0.93,
        "mri_note": "Well-circumscribed hyperintense region; high MobileNetV2 confidence",
        "trajectory": [0.42, 0.48, 0.55, 0.63, 0.74, 0.85, 0.91],
        "summary": "Both EEG 1D-CNN and MRI MobileNetV2 branches indicate elevated risk. Prioritize neuroradiology & EEG review.",
    },
    "NS-1089 (Baseline Rhythm + Clear Structural Scan)": {
        "patient_tag": "Case #NS-1089 · 39y",
        "eeg_prob": 0.24,
        "eeg_note": "Posterior dominant alpha rhythm preserved; minimal artifact",
        "mri_prob": 0.19,
        "mri_note": "Symmetric ventricles; no focal mass-like signal detected",
        "trajectory": [0.26, 0.23, 0.25, 0.21, 0.22, 0.20, 0.21],
        "summary": "Low probability across both modalities with strong inter-branch agreement for routine follow-up.",
    },
    "NS-1104 (Modality Divergence — Watchlist)": {
        "patient_tag": "Case #NS-1104 · 62y",
        "eeg_prob": 0.78,
        "eeg_note": "Intermittent slow-wave bursts during hyperventilation window",
        "mri_prob": 0.34,
        "mri_note": "Mild age-related white matter changes; low tumor score",
        "trajectory": [0.38, 0.44, 0.49, 0.52, 0.58, 0.56, 0.57],
        "summary": "Moderate branch divergence: EEG suggests functional disturbance while MRI structural score remains lower.",
    },
}

COHORT_PRESETS = {
    "Current Batch": {
        "eeg": 0.89,
        "eeg_d": "+0.11 vs baseline",
        "mri": 0.93,
        "mri_d": "MobileNetV2 score",
        "series": [0.42, 0.48, 0.55, 0.63, 0.74, 0.85, 0.91],
    },
    "Prior Run": {
        "eeg": 0.31,
        "eeg_d": "Baseline rhythm",
        "mri": 0.24,
        "mri_d": "Low anomaly prob",
        "series": [0.29, 0.27, 0.30, 0.25, 0.26, 0.24, 0.27],
    },
    "7d Benchmark": {
        "eeg": 0.76,
        "eeg_d": "Rolling 7d mean",
        "mri": 0.81,
        "mri_d": "Rolling 7d mean",
        "series": [0.61, 0.65, 0.70, 0.72, 0.75, 0.78, 0.79],
    },
}

# --- Session State Initialization ---
defaults = {
    "reviewed_cases": 12,
    "target_cases": 20,
    "active_clinician": None,
    "active_backend": "FastAPI + Late Fusion",
    "selected_case_key": None,
    "reviewer_note": "",
    "checklist_state": {
        "EEG Signal Triage": [True, True, False],
        "MRI Scan Review": [True, False, False],
        "Late-Fusion Summary": [False, False, False],
    },
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def risk_badge(score: float) -> tuple[str, str]:
    if score >= 0.70:
        return "Elevated Risk", "chip-high"
    if score >= 0.45:
        return "Moderate / Watch", "chip-mod"
    return "Low Risk", "chip-low"


# --- Top Navigation Bar ---
st.markdown('<div class="topbar-wrap">', unsafe_allow_html=True)
nav_col1, nav_col2, nav_col3 = st.columns([4.2, 3.2, 2.6], vertical_alignment="center")

with nav_col1:
    st.markdown("### 🧠 **NeuroSense**")
    st.caption("Multimodal EEG 1D-CNN & MRI MobileNetV2 Screening Studio")

with nav_col2:
    cohort_view = st.segmented_control(
        "Cohort Window",
        options=list(COHORT_PRESETS.keys()),
        default="Current Batch",
        label_visibility="collapsed",
    )

with nav_col3:
    label_text = (
        f"👤 {st.session_state.active_clinician.split('@')[0]}"
        if st.session_state.active_clinician
        else "🔐 Reviewer Workspace"
    )
    with st.popover(label_text, use_container_width=True):
        if st.session_state.active_clinician:
            st.success(f"Signed in as **{st.session_state.active_clinician}**")
            st.caption(f"Backend: `{st.session_state.active_backend}`")
            if st.button("Sign Out", key="neuro_sign_out", use_container_width=True):
                st.session_state.active_clinician = None
                st.rerun()
        else:
            st.markdown("**Reviewer Session**")
            reviewer_id = st.text_input("Reviewer ID or Email", placeholder="dr.patel@neurosense.ai")
            track = st.selectbox(
                "Inference Backend",
                ["FastAPI + Late Fusion", "TFLite Edge Mode", "EEG / MRI Ablation"],
            )
            if st.button("Open Workspace", type="primary", use_container_width=True):
                st.session_state.active_clinician = reviewer_id.strip() or "Clinical Reviewer"
                st.session_state.active_backend = track
                st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# --- Hero Section ---
hero_left, hero_right = st.columns([1.15, 1], gap="large")

with hero_left:
    st.markdown('<div class="pill-teal">⚡ Multimodal EEG + MRI Decision Fusion</div>', unsafe_allow_html=True)
    st.markdown("# Clear neurological risk signals across wave and scan.")
    st.write(
        "NeuroSense pairs a **1D-CNN** for EEG temporal & spectral dynamics with a "
        "**MobileNetV2 2D-CNN** for structural brain MRI classification—combining modality "
        "probabilities via decision-level late fusion for transparent screening support."
    )

    cta1, cta2 = st.columns([1, 1])
    with cta1:
        if st.button("Load Sample Case", type="primary", use_container_width=True):
            keys = list(SAMPLE_CASES.keys())
            current_idx = keys.index(st.session_state.selected_case_key) if st.session_state.selected_case_key in keys else -1
            st.session_state.selected_case_key = keys[(current_idx + 1) % len(keys)]
            st.toast(f"Loaded {st.session_state.selected_case_key}", icon="🧠")
            st.rerun()
    with cta2:
        st.link_button("Explore Architecture", "#neuro-pillars", use_container_width=True)

    st.markdown("")
    t1, t2, t3 = st.columns(3)
    t1.metric("Fusion Confidence", "94.2%", "+2.8% vs single-mod")
    t2.metric("Runtime Target", "TFLite / API", "Edge & FastAPI ready")
    t3.metric("Modalities", "2 Streams", "1D-CNN EEG + 2D MRI")

with hero_right:
    st.markdown('<div class="telemetry-card">', unsafe_allow_html=True)
    head_l, head_r = st.columns([3, 1.5], vertical_alignment="center")
    with head_l:
        st.markdown('<div class="pill-coral">📡 Live Fusion Preview</div>', unsafe_allow_html=True)
        st.markdown(f"#### {cohort_view or 'Current Batch'} Telemetry")
    with head_r:
        eeg_weight = st.slider(
            "EEG Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Late-fusion weight assigned to EEG 1D-CNN (MRI receives 1 - weight).",
        )

    snap = COHORT_PRESETS[cohort_view or "Current Batch"]
    fused_score = round(eeg_weight * snap["eeg"] + (1.0 - eeg_weight) * snap["mri"], 2)
    badge_label, badge_cls = risk_badge(fused_score)

    m1, m2, m3 = st.columns(3)
    m1.metric("EEG 1D-CNN", f"{snap['eeg']:.2f}", snap["eeg_d"])
    m2.metric("MRI 2D-CNN", f"{snap['mri']:.2f}", snap["mri_d"])
    with m3:
        st.caption("Late-Fusion Score")
        st.markdown(f"**{fused_score:.2f}**")
        st.markdown(f'<span class="status-chip {badge_cls}">{badge_label}</span>', unsafe_allow_html=True)

    st.caption("Late-Fusion Risk Trajectory (Normalized Window Scores)")
    st.line_chart(snap["series"], height=125, use_container_width=True)

    # Interactive Case Review Progress
    progress_val = min(st.session_state.reviewed_cases / st.session_state.target_cases, 1.0)
    c_col1, c_col2 = st.columns([2.3, 1], vertical_alignment="center")
    with c_col1:
        st.markdown(f"**Cases Verified Today:** `{st.session_state.reviewed_cases}` / `{st.session_state.target_cases}`")
        st.progress(progress_val)
    with c_col2:
        if st.button("+1 Verified", use_container_width=True):
            st.session_state.reviewed_cases += 1
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# --- Loaded Sample Case Inspector (when triggered) ---
if st.session_state.selected_case_key:
    case_key = st.session_state.selected_case_key
    case_data = SAMPLE_CASES[case_key]
    case_fused = round(eeg_weight * case_data["eeg_prob"] + (1.0 - eeg_weight) * case_data["mri_prob"], 2)
    c_badge, c_cls = risk_badge(case_fused)

    st.markdown("")
    st.markdown('<div class="case-banner">', unsafe_allow_html=True)
    top_c1, top_c2, top_c3 = st.columns([3, 1.5, 1], vertical_alignment="center")
    with top_c1:
        st.markdown(f"**Active Sample Case:** {case_key}")
        st.caption(case_data["summary"])
    with top_c2:
        st.markdown(
            f"Fused Score: **`{case_fused:.2f}`** &nbsp; <span class='status-chip {c_cls}'>{c_badge}</span>",
            unsafe_allow_html=True,
        )
    with top_c3:
        if st.button("Clear Case", use_container_width=True):
            st.session_state.selected_case_key = None
            st.rerun()

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**EEG Branch (`{case_data['eeg_prob']:.2f}`):** {case_data['eeg_note']}")
    with d2:
        st.markdown(f"**MRI Branch (`{case_data['mri_prob']:.2f}`):** {case_data['mri_note']}")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# --- Core Architecture Pillars ---
st.markdown('<div id="neuro-pillars"></div>', unsafe_allow_html=True)
st.markdown("### Four Pillars of the NeuroSense Pipeline")
st.caption("Structured for reproducible inference, clear modality weighting, and practical deployment.")

p1, p2, p3, p4 = st.columns(4, gap="medium")
pillars = [
    (p1, "🌊 EEG 1D-CNN Encoder", "Extracts temporal and frequency-band patterns from multi-channel EEG signals.", False),
    (p2, "🧲 MRI MobileNetV2 Net", "Classifies structural brain MRI slices with lightweight convolutional features.", True),
    (p3, "🔀 Decision-Level Fusion", "Combines independent branch probabilities so missing or noisy inputs stay interpretable.", False),
    (p4, "⚡ FastAPI & TFLite Stack", "Serves predictions cleanly via REST endpoints and quantized edge models.", True),
]
for col, title, desc, is_coral in pillars:
    card_cls = "pillar-card coral-top" if is_coral else "pillar-card"
    with col:
        st.markdown(
            f'<div class="{card_cls}"><div class="pillar-title">{title}</div><p class="pillar-desc">{desc}</p></div>',
            unsafe_allow_html=True,
        )

st.divider()

# --- Interactive Stage Explorer ---
st.markdown("### Explore Screening & Triage Stages")
st.caption("Switch across workflow stages to inspect stage guidance, verification items, and branch agreement.")

stage = st.radio(
    "Select workflow stage",
    ["EEG Signal Triage", "MRI Scan Review", "Late-Fusion Summary"],
    horizontal=True,
    label_visibility="collapsed",
)

stage_details = {
    "EEG Signal Triage": {
        "guidance": "Inspect bandpass filtering, artifact rejection, and 1D-CNN class probabilities before fusion.",
        "tag": "Branch A · 1D-CNN",
        "items": [
            "Confirm sampling rate and baseline artifact removal",
            "Check band-power anomalies against reference window",
            "Log 1D-CNN softmax confidence score",
        ],
    },
    "MRI Scan Review": {
        "guidance": "Verify slice normalization, orientation alignment, and MobileNetV2 lesion/tumor class probabilities.",
        "tag": "Branch B · MobileNetV2",
        "items": [
            "Validate image resolution and intensity normalization",
            "Review top predicted tumor/structural class probabilities",
            "Flag low-contrast or motion-affected slices",
        ],
    },
    "Late-Fusion Summary": {
        "guidance": "Compare EEG and MRI branch outputs side by side to detect agreement or modality conflict.",
        "tag": "Combined Decision",
        "items": [
            "Check fusion weight balance between EEG and MRI branches",
            "Highlight cases where branch confidences diverge",
            "Export structured summary for clinical/research review",
        ],
    },
}

current_stage = stage_details[stage]
exp_left, exp_right = st.columns([1.2, 1], gap="large")

with exp_left:
    st.info(f"**{stage} ({current_stage['tag']}):** {current_stage['guidance']}")
    st.markdown("**Stage Verification Checklist**")
    checks = st.session_state.checklist_state[stage]
    for idx, label in enumerate(current_stage["items"]):
        checks[idx] = st.checkbox(label, value=checks[idx], key=f"neuro_{stage}_{idx}")

    completed = sum(checks)
    st.caption(f"Completed **{completed}** of **{len(checks)}** checks for *{stage}*")

    st.session_state.reviewer_note = st.text_area(
        "Screening Notes",
        value=st.session_state.reviewer_note,
        placeholder="Record observations, signal quality flags, or follow-up recommendations...",
        height=85,
    )

    report_payload = {
        "cohort": cohort_view,
        "stage": stage,
        "eeg_weight": eeg_weight,
        "checklist_completed": f"{completed}/{len(checks)}",
        "notes": st.session_state.reviewer_note,
    }
    st.download_button(
        "Export Stage Summary (JSON)",
        data=json.dumps(report_payload, indent=2),
        file_name=f"neurosense_{stage.lower().replace(' ', '_')}.json",
        mime="application/json",
    )

with exp_right:
    st.markdown("**Branch Confidence Across Recent Cases (%)**")
    branch_scores = {
        "EEG Signal Triage": [88, 91, 79, 85, 92, 87, 89],
        "MRI Scan Review": [92, 89, 94, 86, 90, 93, 93],
        "Late-Fusion Summary": [91, 92, 89, 88, 94, 92, 94],
    }
    st.bar_chart(branch_scores[stage], height=210, use_container_width=True)

st.markdown("")
st.caption(
    "⚠️ **Research & Decision-Support Notice:** NeuroSense is built for neurological research and "
    "screening support and does not replace formal clinical diagnosis by a licensed physician."
)