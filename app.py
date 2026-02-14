import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="MarketSight | Demand AI", page_icon="📈")

# 2. Load Assets
@st.cache_resource
def load_assets():
    # Ensure these files are in your GitHub repo
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

st.sidebar.header("Input Historical Data")
st.sidebar.write("Enter sales for the last 10 days (starting from Yesterday):")
st.sidebar.info("ℹ️ **Note:** Please enter values between **0 and 50** for accurate results.")

# --- CHANGED: Loop runs from Day -1 (Yesterday) down to Day -10 ---
user_input_reversed = []
for i in range(1, 11):
    # Label is now "Day -1", then "Day -2", etc.
    val = st.sidebar.number_input(f"Day -{i} Sales", min_value=0, max_value=100, value=30)
    user_input_reversed.append(val)

# The list is now [Day-1, Day-2, ... Day-10]
# The AI needs [Day-10, ... Day-2, Day-1] (Chronological)
user_input = user_input_reversed[::-1] 
# ------------------------------------------------------------------

if st.button("🔮 Forecast Next Day Demand"):
    # Preprocess Input
    avg_val = np.mean(user_input)
    full_input = [avg_val] * 50 + user_input # Pad 50 days + 10 real days
    
    input_array = np.array(full_input).reshape(-1, 1)
    scaled_input = scaler.transform(input_array)
    
    # Reshape for LSTM [1, 60, 1]
    X_test = scaled_input.reshape(1, 60, 1)
    
    # Predict
    prediction = model.predict(X_test)
    predicted_sales = scaler.inverse_transform(prediction)
    result = round(predicted_sales[0][0])
    
    # Display Result
    st.markdown("### 🎯 Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Predicted Sales Volume", value=f"{result} Units")
    
    with col2:
        if result > avg_val * 1.1:
            st.success("📈 **Trend: Rising Demand.**\n\nRecommendation: Increase Ad Spend to capitalize on interest.")
        elif result < avg_val * 0.9:
            st.warning("📉 **Trend: Falling Demand.**\n\nRecommendation: Run a discount promotion to boost volume.")
        else:
            st.info("➡️ **Trend: Stable.**\n\nRecommendation: Maintain current marketing cadence.")
    
    # Visualization
    st.markdown("#### Trend Visualization")
    chart_data = user_input + [result]
    
    fig, ax = plt.subplots()
    ax.plot(range(1, 12), chart_data, marker='o', linestyle='-', color='#4a90e2', label='Sales Trend')
    ax.plot(11, result, marker='o', markersize=10, color='#ff4b4b', label='AI Forecast')
    
    ax.set_xticks(range(1, 12))
    # Update Chart Labels to match the new order
    ax.set_xticklabels([f"Day -{11-i}" for i in range(1, 11)] + ["Tomorrow"], rotation=45)
    ax.set_ylabel("Sales Units")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    st.pyplot(fig)
