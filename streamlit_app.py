import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import joblib
import pickle
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import pymysql  # Ensure this is installed: pip install pymysql

# Define model paths using os.path.join for compatibility

with open("xgboost_final_model_new.pkl", 'rb') as f:
    xgboost_model = pickle.load(f)

with open("rf_final_model_new.pkl", 'rb') as f:
    rf_model = pickle.load(f)

scaler = joblib.load("scaler")

def prepare_future_dates(last_date, periods=30):
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    return pd.DataFrame({'Date': future_dates})

def create_future_features(future_df, last_values):
    future_df['Day'] = future_df['Date'].dt.day
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Day_of_Week'] = future_df['Date'].dt.dayofweek
    future_df['Quarter'] = future_df['Date'].dt.quarter

    for lag in range(1, 8):
        future_df[f'lag_{lag}'] = np.nan

    for i, lag in enumerate(range(1, 8)):
        if i < len(last_values):
            future_df.loc[0, f'lag_{lag}'] = last_values[i]

    return future_df

def scale_future_features(future_df, scaler, scale_features):
    scaled_array = scaler.transform(future_df[scale_features])
    scaled_df = pd.DataFrame(scaled_array, columns=scale_features)
    final_df = pd.concat([scaled_df, future_df[['Date', 'Day', 'Month', 'Year', 'Day_of_Week', 'Quarter']].reset_index(drop=True)], axis=1)
    return final_df.set_index('Date')

def forecast_next_30_days_combine(xgb_model, rf_model, scaler, last_date, last_values, scale_features):
    future_df = prepare_future_dates(last_date)
    future_df = create_future_features(future_df, last_values)

    predictions = []
    for i in range(30):
        current_row = future_df.iloc[i:i+1].copy()
        scaled_features = scale_future_features(current_row, scaler, scale_features)
        model_features = scaled_features[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                                          'Day', 'Month', 'Year', 'Day_of_Week', 'Quarter']]
        xgb_pred = xgb_model.predict(model_features)[0]
        rf_pred = rf_model.predict(model_features)[0]
        pred = (xgb_pred + rf_pred) / 2
        predictions.append(pred)

        if i < 29:
            for lag in range(7, 1, -1):
                future_df.loc[i+1, f'lag_{lag}'] = future_df.loc[i, f'lag_{lag-1}']
            future_df.loc[i+1, 'lag_1'] = pred

    future_df['Weighted_MCP_Prediction'] = predictions
    return future_df[['Date', 'Weighted_MCP_Prediction']]

def forecast_next_30_days_xgb(xgb_model,scaler, last_date, last_values, scale_features):
    future_df = prepare_future_dates(last_date)
    future_df = create_future_features(future_df, last_values)

    predictions = []
    for i in range(30):
        current_row = future_df.iloc[i:i+1].copy()
        scaled_features = scale_future_features(current_row, scaler, scale_features)
        model_features = scaled_features[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                                          'Day', 'Month', 'Year', 'Day_of_Week', 'Quarter']]
        xgb_pred = xgb_model.predict(model_features)[0]
        pred = xgb_pred
        predictions.append(pred)

        if i < 29:
            for lag in range(7, 1, -1):
                future_df.loc[i+1, f'lag_{lag}'] = future_df.loc[i, f'lag_{lag-1}']
            future_df.loc[i+1, 'lag_1'] = pred

    future_df['Weighted_MCP_Prediction'] = predictions
    return future_df[['Date', 'Weighted_MCP_Prediction']]

def forecast_next_30_days_rf(rf_model, scaler, last_date, last_values, scale_features):
    future_df = prepare_future_dates(last_date)
    future_df = create_future_features(future_df, last_values)

    predictions = []
    for i in range(30):
        current_row = future_df.iloc[i:i+1].copy()
        scaled_features = scale_future_features(current_row, scaler, scale_features)
        model_features = scaled_features[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                                          'Day', 'Month', 'Year', 'Day_of_Week', 'Quarter']]
        
        rf_pred = rf_model.predict(model_features)[0]
        pred = rf_pred
        predictions.append(pred)

        if i < 29:
            for lag in range(7, 1, -1):
                future_df.loc[i+1, f'lag_{lag}'] = future_df.loc[i, f'lag_{lag-1}']
            future_df.loc[i+1, 'lag_1'] = pred

    future_df['Weighted_MCP_Prediction'] = predictions
    return future_df[['Date', 'Weighted_MCP_Prediction']]


def show_forecast_results(forecast_df):
    st.subheader("30-Day Electricity Price Forecast")
    std_dev = 1672.62
    forecast_df['Lower_Limit'] = forecast_df['Weighted_MCP_Prediction'] - 1.645 * std_dev
    forecast_df['Upper_Limit'] = forecast_df['Weighted_MCP_Prediction'] + 1.645 * std_dev
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d-%m-%Y') 

    cm = sns.light_palette("#121622", as_cmap = True)    

    styled_table = forecast_df.style.background_gradient(cmap = cm).set_properties(**{'text-align': 'left'})

    st.markdown("""
        <style>
            .reportview-container .main .block-container {
                max-width: 100%;
                padding-left: 1rem;
                padding-right: 1rem;
                text-align: center;
            }
            table {
                width: 100% !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.dataframe(styled_table, use_container_width=True)


     # Interactive Plotly Chart with Tooltips
    fig = go.Figure()

    # Forecast Line
    fig.add_trace(go.Scatter(
        x = pd.to_datetime(forecast_df['Date']).dt.strftime('%Y-%m-%d'),
        y = forecast_df['Weighted_MCP_Prediction'],
        mode = 'lines+markers',
        name = 'Forecast',
        line = dict(color='#F97068'),
        hovertemplate = (
        '📅 Date: %{x|%d-%m-%Y}<br>' +
        '⚡ Prediction: %{y:.2f} RS/MWh<br>' +
        '🔼 Upper Limit: %{customdata[0]:.2f} RS/MWh<br>' +
        '🔽 Lower Limit: %{customdata[1]:.2f} RS/MWh<extra></extra>'
),
    customdata = forecast_df[['Upper_Limit', 'Lower_Limit']].values
    ))

    # Confidence Interval Fill
    fig.add_trace(go.Scatter(
        x = pd.concat([pd.to_datetime(forecast_df['Date']).dt.strftime('%Y-%m-%d'), pd.to_datetime(forecast_df['Date']).dt.strftime('%Y-%m-%d')[::-1]]),
        y = pd.concat([forecast_df['Upper_Limit'], forecast_df['Lower_Limit'][::-1]]),
        fill = 'toself',
        fillcolor = 'rgba(255, 100, 0, 0.3)',
        line = dict(color = 'rgba(255,255,255,0)'),
        hoverinfo = 'skip',
        name = '90% Confidence Interval'
    ))

    fig.update_layout(
        title = '30-Day Electricity Price Forecast',
        xaxis_title = 'Date',
        yaxis_title = 'Price (Rs/MWh)',
        hovermode = 'x unified',
        template = 'seaborn',
        height = 500,
        legend = dict(
            x = 0.5,
            y = 1.1,
            xanchor = 'center',
            yanchor = 'bottom',  # Anchor relative to y
            orientation = 'h',   # 'v' = vertical, 'h' = horizontal
            bgcolor = 'rgba(255,255,255,0)',  # Transparent background
            bordercolor = 'gray',
            borderwidth = 1)
        

    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    # Set Page Configuration
    st.set_page_config(page_title="Power Trading App", layout = "wide")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <img src="https://360digit.b-cdn.net/assets/admin/ckfinder/userfiles/images/12-01-2024/aispry.png" width="300">
        <h1 style="color: #ff6400; margin-top: 10px;">Electricity Price Forecasting</h1>
        <p style="color: #666; font-size: 16px;">
            Predict electricity market prices for the next 30 days
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("Forecast Configuration")
        st.markdown("""<div style="padding:0px; background-color: #f0f2f6; border-radius: 5px;">
            <p style="color:black; text-align:center;">Enter database credentials</p></div>""",
            unsafe_allow_html=True)
        user = st.sidebar.text_input("user", "Type Here")
        pw = st.sidebar.text_input("password", "Type Here")
        db = st.sidebar.text_input("database", "Type Here")
        model = st.sidebar.selectbox("Select a model", ["XGBoost", "Random Forest", "XGBoost & Random Forest Combine"])
        st.markdown("---")

    if st.button("Generate Forecast", key = "forecast_button"):
        with st.spinner("Generating 30-day forecast..."):
            try:
                last_date = pd.to_datetime("2025-03-21")
                last_7_values = [4491.87, 4181.42, 4325.82, 4714.66, 4584.31, 3186.81, 3514.95]
                scale_features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7']

                if model == "XGBoost & Random Forest Combine":
                    forecast_df = forecast_next_30_days_combine(
                                    xgboost_model, rf_model, scaler,
                                    last_date, last_7_values, scale_features)
                elif model == "XGBoost":
                    forecast_df = forecast_next_30_days_xgb(
                                    xgboost_model, scaler,
                                    last_date, last_7_values, scale_features)
                else:
                    forecast_df = forecast_next_30_days_rf(
                                    rf_model, scaler,
                                    last_date, last_7_values, scale_features)

                show_forecast_results(forecast_df)                

                try:

                    # Ensure Date column is datetime
                    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                    # Only keep necessary columns
                    db_df = forecast_df[['Date', 'Weighted_MCP_Prediction', 'Lower_Limit', 'Upper_Limit']]
                    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
                    db_df.to_sql('price_forecasts', engine, if_exists = 'append', index = False, chunksize = 1000)
                    st.success("Forecast saved to database successfully!")
                
                except Exception as e:
                    st.error(f"Database error: Please give the Correct Database Credentials")

            except Exception as e:
                st.error(f"Forecast generation failed: {str(e)}")

if __name__ == '__main__':
    main()
