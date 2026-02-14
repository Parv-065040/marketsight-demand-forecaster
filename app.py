import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go # New Interactive Library

# 1. Page Config
st.set_page_config(page_title="MarketSight | Demand AI", page_icon="📈", layout="wide")

# 2. Load Assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('sales_forecast_lstm.h5')
    scaler = joblib.load('sales_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ Model files missing. Please check your GitHub repository.")
    st.stop()

# 3. UI Layout
st.title("📈 MarketSight: Demand Forecaster")
st.markdown("Predict future sales trends to optimize **Ad Spend** and **Inventory**.")

# Sidebar Configuration
st.sidebar.header("Input Historical Data")
st.sidebar.info("ℹ️ **Note:** Enter values between **0 and 50**.")

# Collect input in Reverse Order (Day -1 first)
user_input_reversed = []
for i in range(1, 11):
    val = st.sidebar.number_input(f"Day -{i} Sales", min_value=0, max_value=100, value=30)
    user_input_reversed.append(val)

# Prepare data for Model (Chronological: Day -10 to Day -1)
user_input_chronological = user_input_reversed[::-1]

if st.button("🔮 Forecast Next Day Demand", type="primary"):
    # --- Prediction Logic ---
    avg_val = np.mean(user_input_chronological)
    full_input = [avg_val] * 50 + user_input_chronological
    
    input_array = np.array(full_input).reshape(-1, 1)
    scaled_input = scaler.transform(input_array)
    X_test = scaled_input.reshape(1, 60, 1)
    
    prediction = model.predict(X_test)
    predicted_sales = scaler.inverse_transform(prediction)
    result = round(predicted_sales[0][0])
    
    # --- Results Section ---
    st.markdown("---")
    st.subheader("🎯 Analysis Results")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric(label="Predicted Sales Volume", value=f"{result} Units", delta=f"{result - user_input_chronological[-1]} vs Yesterday")
    
    with col2:
        if result > avg_val * 1.1:
            st.success("📈 **Trend: Rising Demand**\n\nRecommendation: **Increase Ad Spend** to capitalize on interest.")
        elif result < avg_val * 0.9:
            st.warning("📉 **Trend: Falling Demand**\n\nRecommendation: **Run a Discount** to clear inventory.")
        else:
            st.info("➡️ **Trend: Stable**\n\nRecommendation: **Maintain** current strategy.")

    # --- EXOTIC INTERACTIVE GRAPH ---
    st.markdown("### 📊 Interactive Demand Trend")
    
    # Data for plotting
    x_labels = [f"Day -{i}" for i in range(10, 0, -1)] + ["Tomorrow (Forecast)"]
    y_values = user_input_chronological + [result]
    
    # Create Plotly Figure
    fig = go.Figure()

    # 1. The Historical Line (Neon Blue)
    fig.add_trace(go.Scatter(
        x=x_labels[:-1], 
        y=y_values[:-1],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#00f2ff', width=4), # Cyan Neon
        marker=dict(size=10, color='#00f2ff', line=dict(width=2, color='white')),
        fill='tozeroy', # Gradient fill
        fillcolor='rgba(0, 242, 255, 0.1)' # Transparent Cyan
    ))

    # 2. The Prediction Line (Connecting Yesterday to Tomorrow)
    fig.add_trace(go.Scatter(
        x=[x_labels[-2], x_labels[-1]], 
        y=[y_values[-2], y_values[-1]],
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='#ff0055', width=4, dash='dot'), # Pink/Red Neon Dotted
        marker=dict(size=14, color='#ff0055', symbol='star')
    ))

    # 3. Exotic Layout Styling
    fig.update_layout(
        template="plotly_dark", # Dark mode theme
        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified", # Tooltip shows all data
        xaxis=dict(
            showgrid=False, 
            title="Timeline",
            color='#ffffff'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            title="Sales Units",
            color='#ffffff'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)
