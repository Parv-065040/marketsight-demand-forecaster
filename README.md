#  MarketSight: AI Demand Forecasting

##  Executive Summary
Marketing and Operations teams often operate in silos. Marketing runs ads that drive demand, while Operations struggles to maintain inventory. This mismatch leads to either stockouts (lost revenue) or overstocking (wasted capital).

MarketSight is a Time-Series Forecasting tool powered by an LSTM (Long Short-Term Memory) Neural Network. It analyzes historical sales data to predict future demand with high precision, allowing managers to align their ad spend and inventory planning dynamically.

##  The Business Value
- **Inventory Optimization:** Predicts exact stock needs to prevent overstocking and reduce warehousing costs.
- **Dynamic Ad Spending:** Signals marketing managers to increase ad spend when organic demand is predicted to dip, or pause ads when demand is naturally high.
- **Supply Chain Efficiency:** Moves from "Reactive" restocking to "Proactive" planning.

##  Technical Architecture
- **Algorithm:** LSTM (Long Short-Term Memory) Recurrent Neural Network for Regression.
- **Task:** Time-Series Forecasting (Predicting continuous values).
- **Framework:** TensorFlow / Keras.
- **Dataset:** Store Item Demand Forecasting (5 years of daily sales data).
- **Interface:** Streamlit (Interactive Dashboard with Trend Visualization).

##  How to Use the Dashboard
1. Open the live web application.
2. Input the sales figures for the last 10 days (simulated input for the demo).
3. Click "Forecast Next Day Demand."
4. The AI processes the sequence and outputs a predicted sales volume.
5. Review the "Managerial Recommendation" to see if you should increase or decrease marketing pressure.

##  Repository Structure
- `app.py`: The main dashboard script.
- `sales_forecast_lstm.h5`: The trained Time-Series model.
- `sales_scaler.pkl`: The mathematical scaler used to normalize the data.
- `requirements.txt`: Dependencies for cloud deployment.
