import streamlit as st
import pandas as pd
from prophet import Prophet

from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
# import datetime


# Function to perform Prophet modeling
def prophet_modeling(holidays=None):
    model = Prophet(
        # yearly_seasonality=st.sidebar.checkbox('Yearly Seasonality', value=True),
        # weekly_seasonality=st.sidebar.checkbox('Weekly Seasonality', value=True),
        # daily_seasonality=st.sidebar.checkbox('Daily Seasonality', value=False),
        holidays=holidays
    )

    # if st.sidebar.checkbox('Add Custom Seasonality'):
    #     name = st.sidebar.text_input('Seasonality Name', 'monthly')
    #     period = st.sidebar.number_input('Period (days)', value=30)
    #     fourier_order = st.sidebar.number_input('Fourier Order', value=5)
    #     model.add_seasonality(name=name, period=period, fourier_order=fourier_order)

    return model

# Function to plot performance metrics
def plot_performance_metrics(perf_metrics):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(perf_metrics['horizon'], perf_metrics['mape'], label='MAPE')
    plt.plot(perf_metrics['horizon'], perf_metrics['rmse'], label='RMSE')
    plt.xlabel('Forecast horizon')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Performance Metrics')
    st.pyplot(fig)
