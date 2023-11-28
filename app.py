import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Load industry growth data
iip_data = pd.read_excel("IIP_Data.xlsx")

# Load default stock revenue data
stock_revenue_data = pd.read_excel("stock_revenue.xlsx")

# Load stock price data
stock_price_folder = "Stock_Data"
stock_files = [f"{stock_price_folder}/{file}" for file in os.listdir(stock_price_folder) if file.endswith(".xlsx")]

# Helper function to load stock price data
def load_stock_data(file_path):
    return pd.read_excel(file_path)

# Helper function for ARIMA analysis
def perform_arima_analysis(data):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=0)
    return results

# Helper function for ETS analysis
def perform_ets_analysis(data):
    model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=12)
    results = model.fit()
    return results

# Helper function for Regression analysis
def perform_regression_analysis(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Helper function for Random Forest analysis
def perform_random_forest_analysis(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# Helper function for Gradient Boosting analysis
def perform_gradient_boosting_analysis(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

# Helper function for Neural Networks analysis
def perform_neural_networks_analysis(X, y):
    model = MLPRegressor()
    model.fit(X, y)
    return model

# Helper function for Principal Component Analysis
def perform_pca_analysis(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    return pca

# Helper function for Cross-Correlation analysis
def perform_cross_correlation_analysis(series1, series2):
    corr_coefficient, _ = pearsonr(series1, series2)
    return corr_coefficient

# Streamlit app
st.title("Correlation and Trend Analysis")

# ... (rest of your code)

# Correlation Analysis
correlation_result = merged_data.corr(method='pearson')
st.write("## Correlation Analysis Result")
st.write(correlation_result)

# Time Series Analysis: Autoregressive Integrated Moving Average (ARIMA)
arima_results = perform_arima_analysis(merged_data[selected_industry])
st.write("## ARIMA Analysis Result")
st.write(arima_results.summary())

# Exponential Smoothing State Space Models (ETS)
ets_results = perform_ets_analysis(merged_data[selected_industry])
st.write("## ETS Analysis Result")
st.write(ets_results.summary())

# Regression Analysis
X_reg = merged_data[selected_industry].values.reshape(-1, 1)
y_reg = merged_data["Net Income"].values
regression_model = perform_regression_analysis(X_reg, y_reg)
st.write("## Regression Analysis Result")
st.write(f"Regression Coefficient: {regression_model.coef_[0]}")
st.write(f"Intercept: {regression_model.intercept_}")

# ... (rest of your analysis)

# Display graphs
st.write("## Graphs")

# Correlation Plot
plt.figure(figsize=(8, 6))
plt.title('Correlation Analysis')
plt.imshow(correlation_result, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_result.columns)), correlation_result.columns, rotation='vertical')
plt.yticks(range(len(correlation_result.columns)), correlation_result.columns)
st.pyplot(plt)

# ARIMA Analysis Result graph
st.write("### ARIMA Analysis Result Graph")
plt.plot(merged_data['Date'], arima_results.fittedvalues, color='red', label='Fitted values')
plt.plot(merged_data['Date'], merged_data[selected_industry], label='Actual values')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Values')
st.pyplot(plt)

# ETS Analysis Result graph
st.write("### ETS Analysis Result Graph")
plt.plot(merged_data['Date'], ets_results.fittedvalues, color='red', label='Fitted values')
plt.plot(merged_data['Date'], merged_data[selected_industry], label='Actual values')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Values')
st.pyplot(plt)

# ... (continue with other analysis results)

# Cross-Correlation Analysis Result graph
st.write("### Cross-Correlation Analysis Result Graph")
plt.scatter(merged_data[selected_industry], merged_data["Net Income"])
plt.xlabel(selected_industry)
plt.ylabel("Net Income")
plt.title("Cross-Correlation Analysis")
st.pyplot(plt)
