import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go




BASE_DIR = Path(__file__).resolve().parent.parent

DB_FILE = BASE_DIR / "data" / "iot_fault_diagnosis.db"




st.set_page_config(
    page_title="IoT Fault Diagnosis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown(
    """
    <style>

    /* -----------------------------
       GENERAL
    ----------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 15px;
        color: #777;
        margin-bottom: 22px;
    }


    /* -----------------------------
       SIDEBAR
    ----------------------------- */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #dddddd;
    }

    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        border-radius: 10px;
        border: 1px solid transparent;
        padding: 12px 14px;
        font-size: 15px;
        margin-bottom: 5px;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        border: 1px solid #999999;
    }


    /* -----------------------------
       STATUS BOX
    ----------------------------- */

    .status-box {
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-size: 21px;
        font-weight: 700;
        margin: 12px 0 22px 0;
    }

    .status-normal {
        background: #e8f5e9;
        color: #1b5e20;
    }

    .status-fault {
        background: #fff3e0;
        color: #e65100;
    }

    .status-anomaly {
        background: #ffebee;
        color: #b71c1c;
    }


    /* -----------------------------
       INFO CARDS
    ----------------------------- */

    .info-card {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
        background: rgba(255,255,255,0.5);
    }

    .info-card-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-card-text {
        color: #666;
        font-size: 14px;
        line-height: 1.6;
    }


    /* -----------------------------
       LIVE INDICATOR
    ----------------------------- */

    .live-indicator {
        padding: 10px 14px;
        border-radius: 10px;
        background: #e8f5e9;
        color: #1b5e20;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }


    /* -----------------------------
       PIPELINE
    ----------------------------- */

    .pipeline {
        padding: 16px;
        border-radius: 12px;
        background: #eef5ff;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



def get_connection():

    return sqlite3.connect(
        DB_FILE,
        timeout=30
    )


def load_data():

    if not DB_FILE.exists():
        return pd.DataFrame()

    query = """
        SELECT
            r.id,
            r.timestamp,
            r.device_id,

            r.voltage_magnitude,
            r.current_magnitude,

            r.voltage_imbalance,
            r.current_imbalance,

            r.power,
            r.power_variation,

            a.predicted_fault,
            a.confidence,

            a.anomaly,
            a.anomaly_score,

            a.severity,
            a.evidence

        FROM raw_sensor_data r

        LEFT JOIN analysis_results a
        ON r.id = a.raw_data_id

        ORDER BY r.id ASC
    """

    try:

        with get_connection() as conn:

            df = pd.read_sql_query(
                query,
                conn
            )

        if not df.empty:

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

            df["confidence_percent"] = (
                df["confidence"].fillna(0) * 100
            )

        return df

    except Exception as error:

        st.error(
            f"Database error: {error}"
        )

        return pd.DataFrame()




if "page" not in st.session_state:

    st.session_state.page = "Live Monitoring"


def change_page(page_name):

    st.session_state.page = page_name



with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            margin-bottom:5px;
        ">
        ⚡ IoT Fault Diagnosis
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Knowledge-Driven ML Monitoring System"
    )

    st.divider()

    st.markdown("### 🧭 SYSTEM")

    
    if st.button(
        "🟢  Live Monitoring",
        key="live_button",
        use_container_width=True
    ):

        change_page("Live Monitoring")


    
    st.markdown("### 🗄️ DATA SOURCE")

    if st.button(
        "SQLite Database",
        key="sqlite_button",
        use_container_width=True
    ):

        change_page("Data Source")


    

    st.markdown("### 📡 COMMUNICATION")

    if st.button(
        "MQTT / HiveMQ Cloud",
        key="mqtt_button",
        use_container_width=True
    ):

        change_page("Communication")


    
    st.markdown("### 🤖 FAULT DETECTION")

    if st.button(
        "Random Forest",
        key="rf_button",
        use_container_width=True
    ):

        change_page("Fault Model")


    

    st.markdown("### 🔍 ANOMALY DETECTION")

    if st.button(
        "Isolation Forest",
        key="if_button",
        use_container_width=True
    ):

        change_page("Anomaly Model")


    st.divider()


    

    df_sidebar = load_data()

    if not df_sidebar.empty:

        latest_sidebar = df_sidebar.iloc[-1]

        st.success(
            "🟢 System Online"
        )

        st.caption(
            f"Records: {len(df_sidebar)}"
        )

        if pd.notna(
            latest_sidebar["predicted_fault"]
        ):

            st.caption(
                f"Latest: "
                f"{latest_sidebar['predicted_fault']}"
            )

    else:

        st.warning(
            "🟡 Waiting for data"
        )




st.markdown(
    '<div class="main-title">⚡ IoT ML Fault Diagnosis Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Real-time IoT monitoring, machine learning fault classification
    and anomaly detection.
    </div>
    """,
    unsafe_allow_html=True
)



if st.session_state.page == "Live Monitoring":

    st.markdown(
        """
        <div class="pipeline">
        ML Generator
        →
        HiveMQ Cloud
        →
        MQTT Subscriber
        →
        SQLite
        →
        Random Forest + Isolation Forest
        →
        Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )


    @st.fragment(run_every="2s")
    def live_dashboard():

        df = load_data()


        
        if df.empty:

            st.warning(
                "Waiting for sensor data..."
            )

            st.info(
                "Start the MQTT subscriber and ML data generator."
            )

            return


        latest = df.iloc[-1]


        

        fault = (
            latest["predicted_fault"]
            if pd.notna(latest["predicted_fault"])
            else "Waiting"
        )

        anomaly = (
            latest["anomaly"]
            if pd.notna(latest["anomaly"])
            else "WAITING"
        )

        severity = (
            latest["severity"]
            if pd.notna(latest["severity"])
            else "WAITING"
        )


        

        st.subheader("🟢 Current System Status")


        if anomaly == "ANOMALY":

            status_text = "🔴 ANOMALY DETECTED"

            status_class = "status-anomaly"

        elif fault != "Normal":

            status_text = "🟠 FAULT DETECTED"

            status_class = "status-fault"

        else:

            status_text = "🟢 SYSTEM NORMAL"

            status_class = "status-normal"


        st.markdown(
            f"""
            <div class="status-box {status_class}">
                {status_text}
            </div>
            """,
            unsafe_allow_html=True
        )


        

        st.markdown(
            """
            <div class="live-indicator">
            🟢 LIVE MONITORING — Updating automatically
            </div>
            """,
            unsafe_allow_html=True
        )


       
        st.subheader("📡 Latest Sensor Data")


        c1, c2, c3, c4, c5, c6 = st.columns(6)


        with c1:

            st.metric(
                "Voltage",
                f"{latest['voltage_magnitude']:.3f}"
            )


        with c2:

            st.metric(
                "Current",
                f"{latest['current_magnitude']:.3f}"
            )


        with c3:

            st.metric(
                "Power",
                f"{latest['power']:.3f}"
            )


        with c4:

            st.metric(
                "Voltage Imbalance",
                f"{latest['voltage_imbalance']:.2f}"
            )


        with c5:

            st.metric(
                "Current Imbalance",
                f"{latest['current_imbalance']:.2f}"
            )


        with c6:

            st.metric(
                "Power Variation",
                f"{latest['power_variation']:.3f}"
            )


        
        st.divider()

        st.subheader("🤖 Machine Learning Diagnosis")


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Predicted Fault",
                str(fault)
            )


        with c2:

            st.metric(
                "Confidence",
                f"{latest['confidence_percent']:.2f}%"
            )


        with c3:

            st.metric(
                "Anomaly",
                str(anomaly)
            )


        with c4:

            st.metric(
                "Severity",
                str(severity)
            )


        if pd.notna(latest["evidence"]):

            st.info(
                f"🔎 Evidence: {latest['evidence']}"
            )


        

        st.divider()

        st.subheader("📊 Real-Time Sensor Analytics")


        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Voltage & Current",
                "Power",
                "Imbalance",
                "ML Confidence"
            ]
        )


        with tab1:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["voltage_magnitude"],
                    mode="lines",
                    name="Voltage"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["current_magnitude"],
                    mode="lines",
                    name="Current"
                )
            )

            fig.update_layout(
                title="Voltage & Current",
                height=420,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="live_voltage_current"
            )


       

        with tab2:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["power"],
                    mode="lines",
                    name="Power"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["power_variation"],
                    mode="lines",
                    name="Power Variation"
                )
            )

            fig.update_layout(
                title="Power Analysis",
                height=420,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="live_power"
            )


        

        with tab3:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["voltage_imbalance"],
                    mode="lines",
                    name="Voltage Imbalance"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["current_imbalance"],
                    mode="lines",
                    name="Current Imbalance"
                )
            )

            fig.update_layout(
                title="Phase Imbalance",
                height=420,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="live_imbalance"
            )


        

        with tab4:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["confidence_percent"],
                    mode="lines+markers",
                    name="Confidence"
                )
            )

            fig.update_layout(
                title="Random Forest Confidence",
                yaxis=dict(
                    range=[0, 100]
                ),
                height=420,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="live_confidence"
            )


        

        st.divider()

        st.subheader("📈 ML Statistics")


        normal_count = (
            df["predicted_fault"] == "Normal"
        ).sum()


        fault_count = (
            df["predicted_fault"].notna()
            &
            (df["predicted_fault"] != "Normal")
        ).sum()


        anomaly_count = (
            df["anomaly"] == "ANOMALY"
        ).sum()


        average_confidence = (
            df["confidence_percent"].mean()
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Normal",
                int(normal_count)
            )


        with c2:

            st.metric(
                "Faults",
                int(fault_count)
            )


        with c3:

            st.metric(
                "Anomalies",
                int(anomaly_count)
            )


        with c4:

            st.metric(
                "Avg Confidence",
                f"{average_confidence:.2f}%"
            )


        

        st.divider()

        st.subheader("🚨 Recent Diagnosis History")


        history = df[
            [
                "timestamp",
                "device_id",
                "predicted_fault",
                "confidence_percent",
                "anomaly",
                "anomaly_score",
                "severity",
                "evidence"
            ]
        ].copy()


        history = history.sort_values(
            "timestamp",
            ascending=False
        )


        history["confidence_percent"] = (
            history["confidence_percent"].round(2)
        )


        history["anomaly_score"] = (
            history["anomaly_score"].round(4)
        )


        st.dataframe(
            history.head(15),
            use_container_width=True,
            hide_index=True
        )


    live_dashboard()




elif st.session_state.page == "Data Source":

    st.header("🗄️ Data Source")

    st.write(
        "SQLite stores raw sensor data before analysis and "
        "ML diagnosis results after analysis."
    )

    st.divider()


    c1, c2, c3 = st.columns(3)


    df = load_data()


    with c1:

        st.metric(
            "Database",
            "SQLite"
        )


    with c2:

        st.metric(
            "Raw Records",
            len(df)
        )


    with c3:

        if not df.empty:

            analyzed = df["predicted_fault"].notna().sum()

        else:

            analyzed = 0

        st.metric(
            "Analyzed Records",
            int(analyzed)
        )


    st.divider()


    st.subheader("📁 Database Location")

    st.code(
        str(DB_FILE),
        language="text"
    )


    st.subheader("🧩 Database Pipeline")

    st.code(
        """
raw_sensor_data
       ↓
ML Analysis
       ↓
analysis_results
       ↓
Dashboard
        """,
        language="text"
    )


    if not df.empty:

        st.subheader("📊 Stored Data")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No sensor records found."
        )



elif st.session_state.page == "Communication":

    st.header("📡 MQTT / HiveMQ Cloud")

    st.write(
        "MQTT provides communication between the ML data generator "
        "and the diagnosis system."
    )

    st.divider()


    st.subheader("🔄 Communication Architecture")

    st.code(
        """
ML Data Generator
        │
        ▼
MQTT Publisher
        │
        ▼
HiveMQ Cloud
        │
        ▼
MQTT Subscriber
        │
        ▼
SQLite
        │
        ▼
ML Diagnosis
        │
        ▼
Dashboard
        """,
        language="text"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            📤 ML Data Generator
            </div>

            <div class="info-card-text">

            File:

            <b>ml/data_generator.py</b>

            Generates new synthetic sensor measurements
            from learned feature distributions.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            📥 MQTT Subscriber
            </div>

            <div class="info-card-text">

            File:

            <b>backend/mqtt_subscriber.py</b>

            Receives sensor messages and sends them
            to the ML diagnosis pipeline.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.subheader("📨 Message Processing")

    st.write(
        """
        1. Generator creates sensor data.
        2. MQTT publishes the JSON message.
        3. HiveMQ Cloud routes the message.
        4. Subscriber receives the message.
        5. Raw data enters SQLite.
        6. ML models analyze the data.
        7. Results are stored in SQLite.
        """
    )




elif st.session_state.page == "Fault Model":

    st.header("🤖 Random Forest Fault Detection")

    st.write(
        "Random Forest performs supervised classification of sensor patterns."
    )

    st.divider()


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Algorithm",
            "Random Forest"
        )


    with c2:

        st.metric(
            "Estimators",
            "300"
        )


    with c3:

        st.metric(
            "Classes",
            "2"
        )


    st.divider()


    st.subheader("🧠 Input Features")


    features = [
        ("Voltage Magnitude", "Overall voltage level"),
        ("Current Magnitude", "Overall current level"),
        ("Voltage Imbalance", "Phase voltage imbalance"),
        ("Current Imbalance", "Phase current imbalance"),
        ("Power", "Voltage × Current"),
        ("Power Variation", "Change in power")
    ]


    for name, description in features:

        st.markdown(
            f"""
            <div class="info-card">

            <div class="info-card-title">
            {name}
            </div>

            <div class="info-card-text">
            {description}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.subheader("🔄 Prediction Pipeline")

    st.code(
        """
Sensor Features
       ↓
Random Forest
       ↓
Fault Classification
       ↓
Confidence Score
       ↓
SQLite
       ↓
Dashboard
        """,
        language="text"
    )


    df = load_data()


    if not df.empty:

        analyzed = df[
            df["predicted_fault"].notna()
        ]


        if not analyzed.empty:

            st.subheader("📊 Current Model Statistics")


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "Predictions",
                    len(analyzed)
                )


            with c2:

                st.metric(
                    "Normal",
                    int(
                        (
                            analyzed["predicted_fault"]
                            == "Normal"
                        ).sum()
                    )
                )


            with c3:

                st.metric(
                    "Average Confidence",
                    f"{analyzed['confidence_percent'].mean():.2f}%"
                )



elif st.session_state.page == "Anomaly Model":

    st.header("🔍 Isolation Forest Anomaly Detection")

    st.write(
        "Isolation Forest detects sensor behavior that differs "
        "from learned normal patterns."
    )

    st.divider()


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Algorithm",
            "Isolation Forest"
        )


    with c2:

        st.metric(
            "Purpose",
            "Anomaly Detection"
        )


    with c3:

        st.metric(
            "Output",
            "Normal / Anomaly"
        )


    st.divider()


    st.subheader("🔄 Anomaly Detection Pipeline")

    st.code(
        """
Sensor Features
       ↓
Isolation Forest
       ↓
Anomaly Prediction
       ↓
Anomaly Score
       ↓
Severity
       ↓
SQLite
       ↓
Dashboard
        """,
        language="text"
    )


    df = load_data()


    if not df.empty:

        normal_count = (
            df["anomaly"] == "NORMAL"
        ).sum()


        anomaly_count = (
            df["anomaly"] == "ANOMALY"
        ).sum()


        st.subheader("📊 Anomaly Statistics")


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Normal",
                int(normal_count)
            )


        with c2:

            st.metric(
                "Anomalies",
                int(anomaly_count)
            )


        with c3:

            st.metric(
                "Total",
                len(df)
            )


        st.divider()


        st.subheader("🚨 Recent Anomaly Results")


        anomaly_data = df[
            [
                "timestamp",
                "device_id",
                "anomaly",
                "anomaly_score",
                "severity",
                "evidence"
            ]
        ].copy()


        anomaly_data = anomaly_data.sort_values(
            "timestamp",
            ascending=False
        )


        anomaly_data["anomaly_score"] = (
            anomaly_data["anomaly_score"].round(4)
        )


        st.dataframe(
            anomaly_data.head(20),
            use_container_width=True,
            hide_index=True
        )




st.divider()

st.caption(
    "IoT Fault Diagnosis | ML + MQTT + SQLite | "
    "Random Forest + Isolation Forest"
)