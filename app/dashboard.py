import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json


st.set_page_config(
    page_title="RUL Prediction Dashboard",
    page_icon="⚙️",
    layout="wide"
)


st.title("⚙️ RUL Prediction Dashboard")
st.markdown("### Remaining Useful Life Monitoring for Turbofan Engines")


def load_predictions():
    """Simulate loading predictions from API."""
    np.random.seed(42)
    n_engines = 50
    
    data = {
        'engine_id': range(1, n_engines + 1),
        'predicted_rul': np.random.uniform(10, 280, n_engines),
        'current_cycle': np.random.uniform(100, 280, n_engines),
        'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], n_engines),
        'maintenance_action': np.random.choice(
            ['MONITOR', 'PLAN_NEXT', 'SCHEDULE_SOON', 'IMMEDIATE'], 
            n_engines,
            p=[0.3, 0.35, 0.25, 0.1]
        )
    }
    
    df = pd.DataFrame(data)
    df['risk_score'] = df['risk_level'].map({
        'LOW': 20,
        'MEDIUM': 50,
        'HIGH': 75,
        'CRITICAL': 95
    })
    
    return df


col1, col2, col3, col4 = st.columns(4)

df = load_predictions()

with col1:
    st.metric("Total Engines", len(df))

with col2:
    critical = len(df[df['risk_level'] == 'CRITICAL'])
    st.metric("Critical", critical, delta=-critical if critical > 0 else 0)

with col3:
    avg_rul = df['predicted_rul'].mean()
    st.metric("Avg RUL (cycles)", f"{avg_rul:.0f}")

with col4:
    needs_maintenance = len(df[df['risk_level'].isin(['HIGH', 'CRITICAL'])])
    st.metric("Needs Attention", needs_maintenance)


st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Engine Status Distribution")
    
    fig_risk = px.pie(
        df, 
        names='risk_level',
        color='risk_level',
        color_discrete_map={
            'LOW': '#2ecc71',
            'MEDIUM': '#f39c12',
            'HIGH': '#e74c3c',
            'CRITICAL': '#8e44ad'
        },
        hole=0.4
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with col_right:
    st.subheader("Maintenance Actions")
    
    action_counts = df['maintenance_action'].value_counts()
    
    fig_bar = px.bar(
        x=action_counts.index,
        y=action_counts.values,
        color=action_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'x': 'Action', 'y': 'Count'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)


st.divider()

st.subheader("Engine RUL Predictions")

risk_filter = st.multiselect(
    "Filter by Risk Level",
    options=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
    default=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
)

filtered_df = df[df['risk_level'].isin(risk_filter)]

st.dataframe(
    filtered_df.sort_values('risk_score', ascending=False),
    use_container_width=True,
    hide_index=True
)


st.divider()

st.subheader("RUL Distribution by Risk Level")

fig_hist = px.histogram(
    df, 
    x='predicted_rul', 
    color='risk_level',
    color_discrete_map={
        'LOW': '#2ecc71',
        'MEDIUM': '#f39c12',
        'HIGH': '#e74c3c',
        'CRITICAL': '#8e44ad'
    },
    nbins=20,
    barmode='overlay'
)

fig_hist.update_layout(
    xaxis_title="Predicted RUL (cycles)",
    yaxis_title="Count",
    legend_title="Risk Level"
)

st.plotly_chart(fig_hist, use_container_width=True)


st.divider()

st.subheader("Individual Engine Analysis")

engine_id = st.selectbox(
    "Select Engine",
    options=sorted(df['engine_id'].unique())
)

engine_data = df[df['engine_id'] == engine_id].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Cycle", f"{engine_data['current_cycle']:.0f}")

with col2:
    st.metric("Predicted RUL", f"{engine_data['predicted_rul']:.0f} cycles")

with col3:
    st.metric("Risk Level", engine_data['risk_level'])

st.info(f"**Recommended Action:** {engine_data['maintenance_action']}")

if engine_data['risk_level'] in ['HIGH', 'CRITICAL']:
    st.error("⚠️ This engine requires immediate attention!")
elif engine_data['risk_level'] == 'MEDIUM':
    st.warning("This engine should be serviced soon.")
else:
    st.success("This engine is operating normally.")


st.divider()

with st.expander("API Usage"):
    st.markdown("**La API corre en: `http://127.0.0.1:8002`**")
    st.code("""
# Example API call
import requests
import numpy as np

# API en puerto 8002
BASE_URL = "http://127.0.0.1:8002"

# IMPORTANTE: 30 timesteps x 59 features
sensor_data = np.random.randn(30, 59).tolist()

payload = {
    "unit_number": 1,
    "sensor_data": sensor_data,
    "sequence_length": 30
}

response = requests.post(
    f"{BASE_URL}/predict_rul",
    json=payload
)

print(response.json())
    """, language="python")


st.markdown("---")
st.caption("RUL Prediction System | Powered by LSTM + Metaheuristic Optimization + Type-2 Fuzzy")