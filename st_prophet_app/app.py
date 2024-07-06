from typing import Any, Dict, List
import pandas as pd
import numpy as np
import streamlit as st

from prophet import Prophet
from prophet.diagnostics import cross_validation

from lib.utils.load import load_image, load_config
from lib.inputs.params import (
    input_holidays_params,
    input_prior_scale_params, 
    input_seasonality_params
    )

from lib.inputs.dates import (
    input_cv,
    input_forecast_dates,
    input_train_dates,
    input_val_dates,
)


# Page config
st.set_page_config(page_title="Prophet", layout="wide")

# Load config
config, instructions, readme = load_config(
    "config_streamlit.toml", "config_instructions.toml", "config_readme.toml"
)

# Initialization
dates: Dict[Any, Any] = dict()
report: List[Dict[str, Any]] = []

# Info
with st.expander("Prophet model to build a time series forecasting model in a few clicks", expanded=False):
    st.write("Prophet model")
    
st.write("Forecast")

st.sidebar.image(load_image("prophet_logo.PNG"), use_column_width=True)

##
st.sidebar.title("1. Data")

# Load data
with st.sidebar.expander("Dataset", expanded=True):
    file = st.file_uploader(label="Upload a csv file", type="csv", help="dataset_upload", accept_multiple_files=False)
    if file is not None:
        df = pd.read_csv(file)
    else:
        st.stop()
        
# Column names
if df is not None:
    with st.sidebar.expander("Columns", expanded=True):
        date_col = st.selectbox("Date column",sorted(df.columns))
        target_col = st.selectbox( "Target column", sorted(set(df.columns) - {date_col}) )

st.sidebar.title("2. Modelling")


# Prior scale
with st.sidebar.expander("Prior scale", expanded=False):
    params = input_prior_scale_params(config, readme)

# Seasonalities
with st.sidebar.expander("Seasonalities", expanded=False):
    params = input_seasonality_params(config, params, readme)

# Holidays
with st.sidebar.expander("Holidays"):
    params = input_holidays_params(params, readme, config)
    
# Custom Holidays data
with st.sidebar.expander("Custom Holidays"):
    holidays_data = st.file_uploader(label="Upload a csv file with holiday and dates", type="csv", help="holiday_dates_upload", accept_multiple_files=False)
    if holidays_data is not None:
        hd_df = pd.read_csv(holidays_data)
        params['holidays'] = hd_df
        
st.sidebar.title("3. Evaluation")

# Choose whether or not to do evaluation
evaluate = st.sidebar.checkbox(
    "Evaluate my model", value=True, help=readme["tooltips"]["choice_eval"]
)


if evaluate:

    # Split
    with st.sidebar.expander("Split", expanded=True):
        use_cv = st.checkbox(
            "Perform cross-validation", value=False, help=readme["tooltips"]["choice_cv"]
        )
        
    # # Performance metrics
    # with st.sidebar.expander("Metrics", expanded=False):
    #     eval = input_metrics(readme, config)

    # # Scope of evaluation
    # with st.sidebar.expander("Scope", expanded=False):
    #     eval = input_scope_eval(eval, use_cv, readme)





# Launch training & forecast
if st.checkbox(
    "Launch forecast",
    value=False,
    help=readme["tooltips"]["launch_forecast"],
):
    track_experiments = st.checkbox(
        "Track experiments", value=False, help=readme["tooltips"]["track_experiments"]
    )
    
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    df = df[["ds", "y"]]
    # st.write(df.head())
    model = Prophet()
    model.fit(df)
    

    st.sidebar.title("4. Forecast")

    # Choose whether or not to do future forecasts
    make_future_forecast = st.sidebar.checkbox(
        "Make forecast on future dates", value=False, help=readme["tooltips"]["choice_forecast"]
    )

    if make_future_forecast:
        future = model.make_future_dataframe(periods=30, freq='D')
        forecast = model.predict(future)
    else:
        forecast = None
        
    st.write(forecast)