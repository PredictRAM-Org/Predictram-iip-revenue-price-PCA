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
st.title("Percentage Change, Correlation, and Trend Analysis")

# Upload user's stock revenue file
user_stock_revenue_file = st.file_uploader("Upload your stock revenue file (in Excel format)", type=["xlsx"])

if user_stock_revenue_file is not None:
    stock_revenue_data = pd.read_excel(user_stock_revenue_file)

# Select industry and stock
selected_industry = st.selectbox("Select Industry", iip_data.columns[1:])
selected_stock = st.selectbox("Select Stock", [file.split("/")[-1] for file in stock_files])

# Filter data based on user selection
industry_data = iip_data[["Date", selected_industry]]
stock_revenue_data = stock_revenue_data[["Date", "Net Income"]]

# Convert "Date" columns to datetime with adjusted format
try:
    industry_data["Date"] = pd.to_datetime(industry_data["Date"], format="%Y:%m (%b)")
    stock_revenue_data["Date"] = pd.to_datetime(stock_revenue_data["Date"])
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Merge industry growth and stock revenue data on date
merged_data = pd.merge(industry_data, stock_revenue_data, on="Date", how="inner")

# Calculate percentage change
percentage_change_data = merged_data.copy()
percentage_change_data[selected_industry] = percentage_change_data[selected_industry].pct_change() * 100
percentage_change_data["Net Income"] = percentage_change_data["Net Income"].pct_change() * 100

# Correlation Analysis
correlation_result = percentage_change_data.corr(method='pearson')
st.write("## Correlation Analysis Result (Percentage Change)")
st.write(correlation_result)

# Time Series Analysis: Autoregressive Integrated Moving Average (ARIMA)
arima_results = perform_arima_analysis(percentage_change_data[selected_industry].dropna())
st.write("## ARIMA Analysis Result (Percentage Change)")
st.write(arima_results.summary())

# Exponential Smoothing State Space Models (ETS)
ets_results = perform_ets_analysis(percentage_change_data[selected_industry].dropna())
st.write("## ETS Analysis Result (Percentage Change)")
st.write(ets_results.summary())

# Regression Analysis
X_reg = percentage_change_data[selected_industry].dropna().values.reshape(-1, 1)
y_reg = percentage_change_data["Net Income"].dropna().values
regression_model = perform_regression_analysis(X_reg, y_reg)
st.write("## Regression Analysis Result (Percentage Change)")
st.write(f"Regression Coefficient: {regression_model.coef_[0]}")
st.write(f"Intercept: {regression_model.intercept_}")

# Random Forests
X_rf = percentage_change_data[selected_industry].dropna().values.reshape(-1, 1)
y_rf = percentage_change_data["Net Income"].dropna().values
random_forest_model = perform_random_forest_analysis(X_rf, y_rf)
st.write("## Random Forest Analysis Result (Percentage Change)")
st.write(f"Feature Importances: {random_forest_model.feature_importances_}")

# Gradient Boosting
X_gb = percentage_change_data[selected_industry].dropna().values.reshape(-1, 1)
y_gb = percentage_change_data["Net Income"].dropna().values
gradient_boosting_model = perform_gradient_boosting_analysis(X_gb, y_gb)
st.write("## Gradient Boosting Analysis Result (Percentage Change)")
st.write(f"Feature Importances: {gradient_boosting_model.feature_importances_}")

# Neural Networks
X_nn = percentage_change_data[selected_industry].dropna().values.reshape(-1, 1)
y_nn = percentage_change_data["Net Income"].dropna().values
neural_networks_model = perform_neural_networks_analysis(X_nn, y_nn)
st.write("## Neural Networks Analysis Result (Percentage Change)")
st.write(f"Neural Networks R2 Score: {neural_networks_model.score(X_nn, y_nn)}")

# Principal Component Analysis
X_pca = percentage_change_data[selected_industry].dropna().values.reshape(-1, 1)
pca_model = perform_pca_analysis(X_pca)
st.write("## PCA Analysis Result (Percentage Change)")
st.write(f"Explained Variance Ratios: {pca_model.explained_variance_ratio_}")

# Cross-Correlation Analysis
cross_correlation_result = perform_cross_correlation_analysis(percentage_change_data[selected_industry].dropna(), percentage_change_data["Net Income"].dropna())
st.write("## Cross-Correlation Analysis Result (Percentage Change)")
st.write(f"Pearson Correlation Coefficient: {cross_correlation_result}")

# Display graphs
st.write("## Graphs")

# Plot industry growth and net income percentage change
plt.figure(figsize=(10, 6))
plt.plot(percentage_change_data['Date'], percentage_change_data[selected_industry], label='Industry Growth (%)')
plt.plot(percentage_change_data['Date'], percentage_change_data['Net Income'], label='Net Income (%)')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.title('Industry Growth vs. Net Income (Percentage Change)')
plt.legend()
st.pyplot(plt)

# Plot stock price and net income percentage change
stock_price_data = load_stock_data(selected_stock)
stock_price_data["Date"] = pd.to_datetime(stock_price_data["Date"])
merged_stock_data = pd.merge(stock_price_data, stock_revenue_data, on="Date", how="inner")
percentage_change_stock_data = merged_stock_data.copy()
percentage_change_stock_data['Adj Close'] = percentage_change_stock_data['Adj Close'].pct_change() * 100
percentage_change_stock_data['Net Income'] = percentage_change_stock_data['Net Income'].pct_change() * 100

plt.figure(figsize=(10, 6))
plt.plot(percentage_change_stock_data['Date'], percentage_change_stock_data['Adj Close'], label='Stock Price (%)')
plt.plot(percentage_change_stock_data['Date'], percentage_change_stock_data['Net Income'], label='Net Income (%)')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.title('Stock Price vs. Net Income (Percentage Change)')
plt.legend()
st.pyplot(plt)
