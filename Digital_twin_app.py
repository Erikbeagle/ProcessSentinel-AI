import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI-Enabled Digital Twin",
    layout="wide"
)

APP_DIR = os.path.dirname(__file__)

rf_model = joblib.load(os.path.join(APP_DIR, "rf_model.pkl"))
region1_features = joblib.load(os.path.join(APP_DIR, "region1_features.pkl"))
engineering_thresholds = joblib.load(os.path.join(APP_DIR, "engineering_thresholds.pkl"))
df_process = pd.read_csv(os.path.join(APP_DIR, "df_process.csv"))

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #0B1F3A;
}
.sub-title {
    font-size: 16px;
    color: #4B5563;
}
.status-card {
    padding: 22px;
    border-radius: 14px;
    background-color: #F8FAFC;
    border: 1px solid #D9E2EC;
}
.metric-card {
    padding: 18px;
    border-radius: 14px;
    background-color: #FFFFFF;
    border: 1px solid #D9E2EC;
    text-align: center;
}
.big-risk {
    font-size: 48px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


def normalize_risk(value, normal_limit, event_limit):
    score = ((value - normal_limit) / (event_limit - normal_limit)) * 100
    return max(0, min(100, score))


def get_threshold(indicator, column):
    return engineering_thresholds.loc[
        engineering_thresholds["Indicator"] == indicator,
        column
    ].values[0]


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
        color = "green"
        action = "Continue normal production and routine monitoring."
    elif final_risk < 50:
        risk_level = "MEDIUM"
        status = "Increased Monitoring Required"
        color = "orange"
        action = "Monitor filter pressure trends and verify circulation stability."
    elif final_risk < 80:
        risk_level = "HIGH"
        status = "Inspection Recommended"
        color = "orange"
        action = "Inspect Region I filters, verify inlet pressure, and prepare cleaning if risk continues."
    else:
        risk_level = "CRITICAL"
        status = "Immediate Action Required"
        color = "red"
        action = "Stop or slow production as appropriate, inspect filters immediately, and initiate maintenance response."

    return live_snapshot, ai_probability, engineering_risk, final_risk, risk_level, status, color, action, dff_risk, inlet_risk


# Sidebar
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


live_snapshot, ai_probability, engineering_risk, final_risk, risk_level, status, color, action, dff_risk, inlet_risk = run_digital_twin(snapshot_index)
timestamp = df_process.loc[snapshot_index, "DateTime(EDT)"]


# Header
st.markdown('<div class="main-title">AI-Enabled Digital Twin – Region I Clogging Risk</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Operator decision-support prototype using AI prediction and engineering risk logic</div>', unsafe_allow_html=True)

st.divider()

# Top Dashboard
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

with col1:
    st.markdown("### Plant Status")
    if color == "green":
        st.success(f"{risk_level} – {status}")
    elif color == "red":
        st.error(f"{risk_level} – {status}")
    else:
        st.warning(f"{risk_level} – {status}")

with col2:
    st.metric("AI Probability", f"{ai_probability:.2f}%")

with col3:
    st.metric("Engineering Risk", f"{engineering_risk:.2f}%")

with col4:
    st.metric("Final Risk Index", f"{final_risk:.2f}%")


# Risk Gauge
st.subheader("Digital Twin Risk Gauge")

st.progress(int(final_risk))

st.markdown(
    f"""
    <div class="status-card">
        <div class="big-risk" style="color:{color};">{final_risk:.2f}%</div>
        <b>Risk Level:</b> {risk_level}<br>
        <b>Timestamp:</b> {timestamp}<br>
        <b>Snapshot Index:</b> {snapshot_index}
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Operator Action
st.subheader("Recommended Operator Action")
if color == "green":
    st.success(action)
elif color == "red":
    st.error(action)
else:
    st.warning(action)

st.divider()

# Process Values and Contributors
left, right = st.columns(2)

with left:
    st.subheader("Live Process Snapshot")

    snapshot_table = pd.DataFrame({
        "Engineering Indicator": region1_features,
        "Current Value": live_snapshot.values
    })

    st.dataframe(snapshot_table, use_container_width=True)

with right:
    st.subheader("Primary Risk Contributors")

    contributors = pd.DataFrame({
        "Indicator": [
            "Filter Differential Pressure",
            "Filter Inlet Pressure"
        ],
        "Current Risk Contribution": [
            f"{dff_risk:.2f}%",
            f"{inlet_risk:.2f}%"
        ],
        "Engineering Role": [
            "Primary Indicator",
            "Primary Indicator"
        ]
    })

    st.dataframe(contributors, use_container_width=True)

st.divider()

# Historical Trend
st.subheader("Historical Trend Around Selected Snapshot")

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

st.line_chart(trend_df)

st.caption("The trend chart shows the selected snapshot and the preceding historical process behavior used by the Digital Twin prototype.")