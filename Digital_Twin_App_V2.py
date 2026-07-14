import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# Page Setup
# ==========================================================

st.set_page_config(
    page_title="AI-Enabled Digital Twin Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = os.path.dirname(__file__)

rf_model = joblib.load(os.path.join(APP_DIR, "rf_model.pkl"))
region1_features = joblib.load(os.path.join(APP_DIR, "region1_features.pkl"))
engineering_thresholds = joblib.load(os.path.join(APP_DIR, "engineering_thresholds.pkl"))
df_process = pd.read_csv(os.path.join(APP_DIR, "df_process.csv"))

# ==========================================================
# Styling
# ==========================================================

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #0B3C5D;
}
.sub-title {
    font-size: 18px;
    color: #555;
    margin-bottom: 20px;
}
.kpi-card {
    background-color: #F7F9FC;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #D6DCE5;
    text-align: center;
}
.kpi-label {
    font-size: 15px;
    color: #666;
}
.kpi-value {
    font-size: 32px;
    font-weight: 800;
}
.status-banner {
    padding: 24px;
    border-radius: 16px;
    font-size: 24px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 18px;
}
.gauge {
    width: 260px;
    height: 260px;
    border-radius: 50%;
    margin: auto;
    display: flex;
    align-items: center;
    justify-content: center;
}
.gauge-inner {
    width: 175px;
    height: 175px;
    border-radius: 50%;
    background: white;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}
.gauge-value {
    font-size: 38px;
    font-weight: 900;
}
.gauge-label {
    font-size: 14px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Helper Functions
# ==========================================================

def normalize_risk(value, normal_limit, event_limit):
    score = ((value - normal_limit) / (event_limit - normal_limit)) * 100
    return max(0, min(100, score))

def get_threshold(indicator, column):
    return engineering_thresholds.loc[
        engineering_thresholds["Indicator"] == indicator, column
    ].values[0]

def indicator_status(score):
    if score < 20:
        return "🟢 Normal"
    elif score < 50:
        return "🟡 Monitor"
    elif score < 80:
        return "🟠 High"
    else:
        return "🔴 Critical"

dff_normal = get_threshold("COMP_FILTER_PRESSURE_DFF (psi)", "Normal 95th Percentile")
dff_event = get_threshold("COMP_FILTER_PRESSURE_DFF (psi)", "Event Median")
inlet_normal = get_threshold("COMP_FILTER_PRESSURE_IN (psi)", "Normal 95th Percentile")
inlet_event = get_threshold("COMP_FILTER_PRESSURE_IN (psi)", "Event Median")

def run_digital_twin(snapshot_index):
    live_snapshot = df_process.loc[snapshot_index, region1_features]
    live_input = pd.DataFrame([live_snapshot.values], columns=region1_features)

    ai_probability = rf_model.predict_proba(live_input)[0][1] * 100

    dff_risk = normalize_risk(
        live_snapshot["COMP_FILTER_PRESSURE_DFF (psi)"],
        dff_normal,
        dff_event
    )

    inlet_risk = normalize_risk(
        live_snapshot["COMP_FILTER_PRESSURE_IN (psi)"],
        inlet_normal,
        inlet_event
    )

    engineering_risk = (0.60 * dff_risk) + (0.40 * inlet_risk)
    final_risk = (0.50 * engineering_risk) + (0.50 * ai_probability)

    if final_risk < 20:
        risk_level = "LOW"
        status = "Normal Operation"
        color = "#16A34A"
        action = "Continue normal production and routine monitoring."
        banner = "🟢 NORMAL OPERATION"
        banner_bg = "#DCFCE7"
    elif final_risk < 50:
        risk_level = "MEDIUM"
        status = "Increased Monitoring Required"
        color = "#CA8A04"
        action = "Monitor filter pressure trends and verify circulation stability."
        banner = "🟡 MONITORING REQUIRED"
        banner_bg = "#FEF9C3"
    elif final_risk < 80:
        risk_level = "HIGH"
        status = "Inspection Recommended"
        color = "#EA580C"
        action = "Inspect Region I filters, verify inlet pressure, and prepare cleaning if risk continues."
        banner = "🟠 HIGH CLOGGING RISK"
        banner_bg = "#FFEDD5"
    else:
        risk_level = "CRITICAL"
        status = "Immediate Action Required"
        color = "#DC2626"
        action = "Stop or slow production as appropriate, inspect filters immediately, and initiate maintenance response."
        banner = "🔴 CRITICAL CLOGGING RISK"
        banner_bg = "#FEE2E2"

    return live_snapshot, ai_probability, engineering_risk, final_risk, risk_level, status, color, action, banner, banner_bg, dff_risk, inlet_risk

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("Digital Twin Controls")

mode = st.sidebar.radio(
    "Select Mode",
    ["Historical Snapshot", "Quick Scenario"]
)

if mode == "Historical Snapshot":
    snapshot_index = st.sidebar.slider(
        "Select Process Snapshot",
        min_value=0,
        max_value=len(df_process) - 1,
        value=4500
    )
else:
    scenario = st.sidebar.selectbox(
        "Select Scenario",
        ["Normal Operation", "Critical Risk"]
    )
    snapshot_index = 4500 if scenario == "Normal Operation" else 380

# ==========================================================
# Run App
# ==========================================================

live_snapshot, ai_probability, engineering_risk, final_risk, risk_level, status, color, action, banner, banner_bg, dff_risk, inlet_risk = run_digital_twin(snapshot_index)
timestamp = df_process.loc[snapshot_index, "DateTime(EDT)"]

# ==========================================================
# Header
# ==========================================================

st.markdown("""
<div class="main-title">🏭 AI-Enabled Digital Twin Dashboard</div>
<div class="sub-title">
Region I Compounding Process • Clogging Risk Prediction • Engineering Decision Support Prototype
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Status Banner
# ==========================================================

st.markdown(
    f"""
    <div class="status-banner" style="background-color:{banner_bg}; color:{color}; border:2px solid {color};">
        {banner}<br>
        <span style="font-size:16px; font-weight:500;">{status}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# KPI Cards
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Plant Status</div>
        <div class="kpi-value" style="color:{color};">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">AI Probability</div>
        <div class="kpi-value">{ai_probability:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Engineering Risk</div>
        <div class="kpi-value">{engineering_risk:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Digital Twin Risk</div>
        <div class="kpi-value" style="color:{color};">{final_risk:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================================
# Gauge + Recommendation
# ==========================================================

left, right = st.columns([1, 1])

with left:
    st.subheader("Plant Health Gauge")

    st.markdown(
        f"""
        <div class="gauge" style="background: conic-gradient({color} {final_risk}%, #E5E7EB 0);">
            <div class="gauge-inner">
                <div class="gauge-value" style="color:{color};">{final_risk:.1f}%</div>
                <div class="gauge-label">Digital Twin Risk</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"**Snapshot Index:** {snapshot_index}")
    st.write(f"**Timestamp:** {timestamp}")

with right:
    st.subheader("Operator Recommendation")

    if risk_level == "LOW":
        st.success(action)
    elif risk_level == "CRITICAL":
        st.error(action)
    else:
        st.warning(action)

    st.markdown("### Decision Logic")
    st.write(
        "The Digital Twin combines the Engineering Risk Score and AI Probability "
        "to generate a single operator-facing clogging risk index."
    )

st.divider()

# ==========================================================
# Live Process and Contributors
# ==========================================================

left, right = st.columns(2)

with left:
    st.subheader("Live Process Snapshot")

    snapshot_table = pd.DataFrame({
        "Engineering Indicator": region1_features,
        "Current Value": [round(float(x), 2) for x in live_snapshot.values]
    })

    st.dataframe(snapshot_table, use_container_width=True)

with right:
    st.subheader("Primary Risk Contributors")

    contributor_table = pd.DataFrame({
        "Indicator": [
            "Filter Differential Pressure",
            "Filter Inlet Pressure"
        ],
        "Current Value": [
            round(float(live_snapshot["COMP_FILTER_PRESSURE_DFF (psi)"]), 2),
            round(float(live_snapshot["COMP_FILTER_PRESSURE_IN (psi)"]), 2)
        ],
        "Risk Score": [
            f"{dff_risk:.2f}%",
            f"{inlet_risk:.2f}%"
        ],
        "Status": [
            indicator_status(dff_risk),
            indicator_status(inlet_risk)
        ]
    })

    st.dataframe(contributor_table, use_container_width=True)

st.divider()

# ==========================================================
# Trend Chart
# ==========================================================

st.subheader("Four-Hour Historical Trend")

window_start = max(snapshot_index - 16, 0)
window_end = min(snapshot_index + 1, len(df_process))

trend_df = df_process.loc[
    window_start:window_end,
    [
        "DateTime(EDT)",
        "COMP_FILTER_PRESSURE_DFF (psi)",
        "COMP_FILTER_PRESSURE_IN (psi)"
    ]
].copy()

trend_df["DateTime(EDT)"] = pd.to_datetime(trend_df["DateTime(EDT)"])
trend_df = trend_df.set_index("DateTime(EDT)")

st.line_chart(trend_df, height=360)

st.caption("Trend chart shows the selected snapshot and the preceding four hours of Region I process behavior.")

st.divider()

# ==========================================================
# Thresholds
# ==========================================================

st.subheader("Engineering Thresholds Used by Digital Twin")

threshold_table = engineering_thresholds.copy()
st.dataframe(threshold_table, use_container_width=True)

st.caption("Thresholds were derived from historical normal-operation and confirmed clogging-event conditions.")