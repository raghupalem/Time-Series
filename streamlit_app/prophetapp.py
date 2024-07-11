import streamlit as st
import pandas as pd
from prophet import Prophet

from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import datetime

from utils.load import load_image,load_toy_dataset
from utils.models import prophet_modeling,plot_performance_metrics

# Info
with st.expander("Prophet model to build a time series forecasting model in a few clicks", expanded=False):
    st.write("Prophet model")
    
st.write("Time series forecasting")

st.sidebar.image(load_image("prophet_logo.PNG"), use_column_width=True)

##
st.sidebar.title("1. Data")

with st.sidebar.expander("Dataset", expanded=True):
    if st.checkbox(
        "Load a dummy dataset", True):
        with st.sidebar.expander("Defalut dataset", expanded=True):
            # file = 'ApplicationMaxDailyTPS.xlsx'
            file = "C:\\Users\\NH2395\\Desktop\\TS\\st_prophet_app\\lib\\inputs\\ApplicationMaxDailyTPS.xlsx"
            df = pd.read_excel(file)
            if df is None:
                st.stop()
    else:
        if st.checkbox(
            "Upload my own config file", False):
            with st.sidebar.expander("Dataset", expanded=True):
                file = st.file_uploader(label="Upload an Excel file",
                type=["xlsx", "xls"],
                help="Upload your Excel dataset",
                accept_multiple_files=False)
                
                # Check if a file has been uploaded
                if file is not None:
                    # Read the Excel file to get the sheet names
                    xls = pd.ExcelFile(file)
                    sheet_names = xls.sheet_names

                    # Create a dropdown menu for selecting a sheet
                    sheet_selected = st.selectbox(
                        label="Select a sheet",
                        options=sheet_names,
                        help="Select a sheet from the uploaded Excel file"
                    )

                    # Read the selected sheet into a DataFrame
                    df = pd.read_excel(file, sheet_name=sheet_selected)
                else:
                    st.write("Please upload an Excel file")
                    st.stop()
        else: st.stop()
# Column names
if df is not None:
    with st.sidebar.expander("Columns", expanded=True):
        date_col = st.selectbox("Date column",sorted(df.columns))
        target_col = st.selectbox( "Target column", sorted(set(df.columns) - {date_col}) )
        df = df.rename(columns={date_col: "ds", target_col: "y"})
        
# st.write(df)
###********************************************************************************
st.sidebar.title("2. Modelling")

# Split data into training and testing sets
st.sidebar.header('Data Splitting')

# Get the first and last date
first_date = df['ds'].min()
last_date = df['ds'].max()

# Display the first and last date in the sidebar
st.sidebar.write(f"First Date: {first_date.strftime('%Y-%m-%d')}")
st.sidebar.write(f"Last Date: {last_date.strftime('%Y-%m-%d')}")

# split_date = st.sidebar.date_input('Select the date for splitting data', value=df.ds[0])

# train = df[df['ds'] < pd.to_datetime(split_date)]
# test = df[df['ds'] >= pd.to_datetime(split_date)]

# Slider for selecting the percentage of data for training
split_percentage = st.sidebar.slider('Select the percentage for training data', 0, 100, 90, step=10)

# Calculate the split index based on the selected percentage
split_index = int(len(df) * split_percentage / 100)

# Split the data
train = df[:split_index]
test = df[split_index:]

# Display the split data
st.sidebar.write(f"Training data length: {train.shape[0]}")
st.subheader('Training Data')
st.write(train.head())

st.sidebar.write(f"Testing data length: {test.shape[0]}")
st.subheader('Testing Data')
st.write(test.head())

 
# Custom Holidays data
with st.sidebar.expander("Custom Holidays"):
    holidays_data = st.file_uploader(label="Upload a excel file with holiday and dates", type=["xlsx", "xls"], help="holiday_data_upload", accept_multiple_files=False)
    if holidays_data is not None:
        holidays_df = pd.read_excel(holidays_data)
    else:
        holidays_df = None
        # params['holidays'] = hd_df

 
##********************** Evaluation *******************************    
st.sidebar.title("3. Evaluation")

# Choose whether or not to do evaluation
evaluate = st.sidebar.checkbox("Evaluate my model", value=True)

if evaluate:
    
    # Perform modeling
    model = prophet_modeling(holidays_df)

    # Fit the model
    model.fit(train)
    
    # Make future dataframe for the period of the test set
    future = model.make_future_dataframe(periods=len(test), freq='D')
    forecast = model.predict(future)

##********************** Forecast *******************************    
    st.sidebar.title("4. Forecast")
    forecast_data = st.sidebar.checkbox("Launch forecast", value=False)

    if forecast_data:
        # Plot forecast
        st.subheader('Forecast')
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Plot components
        st.subheader('Components')
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
        
    cross_validate = st.sidebar.checkbox("Launch cross validation", value=False)
    if cross_validate:
        
        # Perform cross-validation
        initial = st.sidebar.text_input('Initial Training Period', '365 days')
        period = st.sidebar.text_input('Period', '180 days')
        horizon = st.sidebar.text_input('Horizon', '90 days')
        cv_results = cross_validation(model, initial=initial, period=period, horizon=horizon)
        perf_metrics = performance_metrics(cv_results)

        # Display performance metrics
        st.subheader('Performance Metrics')
        st.write(perf_metrics)

        # # Plot cross-validation performance metrics
        # st.subheader('Cross-Validation Performance Metrics')
        # plot_performance_metrics(perf_metrics)

        # Plot additional metrics
        st.subheader('Cross-Validation Metrics Plot')
        fig3 = plot_cross_validation_metric(cv_results, metric='mape')
        st.pyplot(fig3)
    
        