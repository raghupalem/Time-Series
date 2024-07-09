from typing import Any, Dict, List
import pandas as pd
import numpy as np
import streamlit as st

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

from prophet.diagnostics import cross_validation

from lib.dataprep.split import get_train_set, get_train_val_sets

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from lib.inputs.eval import input_metrics, input_scope_eval
from lib.utils.load import load_image, load_config
from lib.inputs.params import (
    input_holidays_params,
    input_prior_scale_params, 
    input_seasonality_params
    )

from lib.models.prophet import forecast_workflow
from lib.dataprep.clean import clean_df

from lib.inputs.dates import (
    input_cv,
    input_forecast_dates,
    input_train_dates,
    input_val_dates,
)

from lib.dataprep.format import (
    add_cap_and_floor_cols,
    check_dataset_size,
    filter_and_aggregate_df,
    format_date_and_target,
    format_datetime,
    print_empty_cols,
    print_removed_cols,
    remove_empty_cols,
    resample_df,
)

from lib.inputs.dataprep import input_cleaning, input_dimensions, input_resampling
from lib.inputs.dataset import (
    input_columns,
    input_dataset,
    input_future_regressors,
)
from lib.inputs.dates import (
    input_cv,
    input_forecast_dates,
    input_train_dates,
    input_val_dates,
)

from lib.inputs.params import (
    input_holidays_params,
    input_other_params,
    input_prior_scale_params,
    input_regressors,
    input_seasonality_params,
)

from lib.exposition.visualize import (
    plot_components,
    plot_future,
    plot_overview,
    plot_performance,
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
df = pd.DataFrame()
# Load data

# Load data
with st.sidebar.expander("Dataset", expanded=True):
    # df, config, datasets = input_dataset(config, readme, instructions)
    df, config, datasets = input_dataset(config, readme, instructions)
    df, empty_cols = remove_empty_cols(df)
    print_empty_cols(empty_cols)
    
    
# with st.sidebar.expander("Dataset", expanded=True):
#     if st.checkbox(
#         "Load a dummy dataset", True, help=readme["tooltips"]["upload_choice"]
#     ):
#         with st.sidebar.expander("Defalut dataset", expanded=True):
#             # file = 'ApplicationMaxDailyTPS.xlsx'
#             file = "C:\\Users\\NH2395\\Desktop\\TS\\st_prophet_app\\lib\\inputs\\ApplicationMaxDailyTPS.xlsx"
#             df = pd.read_excel(file)
#             if df is None:
#                 st.stop()
#     else:
#         if st.checkbox(
#             "Upload my own config file", False, help=readme["tooltips"]["custom_config_choice"]
#         ):
#             with st.sidebar.expander("Dataset", expanded=True):
#                 file = st.file_uploader(label="Upload an Excel file",
#                 type=["xlsx", "xls"],
#                 help="Upload your Excel dataset",
#                 accept_multiple_files=False)
                
#                 # Check if a file has been uploaded
#                 if file is not None:
#                     # Read the Excel file to get the sheet names
#                     xls = pd.ExcelFile(file)
#                     sheet_names = xls.sheet_names

#                     # Create a dropdown menu for selecting a sheet
#                     sheet_selected = st.selectbox(
#                         label="Select a sheet",
#                         options=sheet_names,
#                         help="Select a sheet from the uploaded Excel file"
#                     )

#                     # Read the selected sheet into a DataFrame
#                     df = pd.read_excel(file, sheet_name=sheet_selected)
#                 else:
#                     st.write("Please upload an Excel file")
#                     st.stop()
                
# Column names
if df is not None:
    with st.sidebar.expander("Columns", expanded=True):
        date_col = st.selectbox("Date column",sorted(df.columns))
        target_col = st.selectbox( "Target column", sorted(set(df.columns) - {date_col}) )
        # df = df.rename(columns={date_col: "ds", target_col: "y"})
        df = format_date_and_target(df, date_col, target_col, config)


# Filtering
with st.sidebar.expander("Filtering", expanded=False):
    dimensions = input_dimensions(df, readme, config)
    df, cols_to_drop = filter_and_aggregate_df(df, dimensions, config, date_col, target_col)
    print_removed_cols(cols_to_drop)

# Resampling
with st.sidebar.expander("Resampling", expanded=False):
    resampling = input_resampling(df, readme)
    df = format_datetime(df, resampling)
    df = resample_df(df, resampling)
    check_dataset_size(df, config)

# Cleaning
with st.sidebar.expander("Cleaning", expanded=False):
    cleaning = input_cleaning(resampling, readme, config)
    df = clean_df(df, cleaning)
    check_dataset_size(df, config)
    
###********************************************************************************
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

# External regressors
if len(df.columns) > 2:
    with st.sidebar.expander("Regressors"):
        params = input_regressors(df, config, params, readme)
else:
    params = params

# Other parameters
with st.sidebar.expander("Other parameters", expanded=False):
    params = input_other_params(config, params, readme)
    df = add_cap_and_floor_cols(df, params)
   
# Custom Holidays data
with st.sidebar.expander("Custom Holidays"):
    holidays_data = st.file_uploader(label="Upload a csv file with holiday and dates", type=["xlsx", "xls"], help="holiday_dates_upload", accept_multiple_files=False)
    if holidays_data is not None:
        hd_df = pd.read_csv(holidays_data)
        params['holidays'] = hd_df
    
##********************** Evaluation *******************************    
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
        dates = input_train_dates(df, use_cv, config, resampling, dates)
        if use_cv:
            dates = input_cv(dates, resampling, config, readme)
            datasets = get_train_set(df, dates, datasets)
        else:
            dates = input_val_dates(df, dates, config)
            datasets = get_train_val_sets(df, dates, config, datasets)

    # Performance metrics
    with st.sidebar.expander("Metrics", expanded=False):
        eval = input_metrics(readme, config)

    # Scope of evaluation
    with st.sidebar.expander("Scope", expanded=False):
        eval = input_scope_eval(eval, use_cv, readme)

else:
    use_cv = False


############################################
st.sidebar.title("4. Forecast")

# Choose whether or not to do future forecasts
make_future_forecast = st.sidebar.checkbox(
    "Make forecast on future dates", value=False, help=readme["tooltips"]["choice_forecast"]
)
if make_future_forecast:
    with st.sidebar.expander("Horizon", expanded=False):
        dates = input_forecast_dates(df, dates, resampling, config, readme)
    # with st.sidebar.expander("Regressors", expanded=False):
    #     datasets = input_future_regressors(datasets, dates, params, dimensions,  date_col)  #load_options

# Launch training & forecast
if st.checkbox(
    "Launch forecast",
    value=False,
    help=readme["tooltips"]["launch_forecast"],
):

    if not (evaluate | make_future_forecast):
        st.error("Please check at least 'Evaluation' or 'Forecast' in the sidebar.")

    track_experiments = st.checkbox(
        "Track experiments", value=False, help=readme["tooltips"]["track_experiments"]
    )

    datasets, models, forecasts = forecast_workflow(
        config,
        use_cv,
        make_future_forecast,
        evaluate,
        cleaning,
        resampling,
        params,
        dates,
        datasets,
        df,
        date_col,
        target_col,
        dimensions,
        # load_options,
    )


# Visualizations

if evaluate | make_future_forecast:
    st.write("# 1. Overview")
    report = plot_overview(
        make_future_forecast, use_cv, models, forecasts, target_col, cleaning, readme, report
    )

if evaluate:
    st.write(
        f'# 2. Evaluation on {"CV" if use_cv else ""} {eval["set"].lower()} set{"s" if use_cv else ""}'
    )
    report = plot_performance(
        use_cv, target_col, datasets, forecasts, dates, eval, resampling, config, readme, report
    )

if evaluate | make_future_forecast:
    st.write(
        "# 3. Impact of components and regressors"
        if evaluate
        else "# 2. Impact of components and regressors"
    )
    report = plot_components(
        use_cv,
        make_future_forecast,
        target_col,
        models,
        forecasts,
        cleaning,
        resampling,
        config,
        readme,
        df,
        report,
    )

if make_future_forecast:
    st.write("# 4. Future forecast" if evaluate else "# 3. Future forecast")
    report = plot_future(models, forecasts, dates, target_col, cleaning, readme, report)

# Save experiment
if track_experiments:
    display_save_experiment_button(
        report,
        config,
        use_cv,
        make_future_forecast,
        evaluate,
        cleaning,
        resampling,
        params,
        dates,
        date_col,
        target_col,
        dimensions,
    )

    # Visualizations


# st.sidebar.title("4. Forecast")

# # Choose whether or not to do future forecasts
# make_future_forecast = st.sidebar.checkbox(
#     "Make forecast on future dates", value=False, help=readme["tooltips"]["choice_forecast"]
# )

# # Launch training & forecast
# if st.checkbox(
#     "Launch forecast",
#     value=False,
#     help=readme["tooltips"]["launch_forecast"],
# ):
#     track_experiments = st.checkbox(
#         "Track experiments", value=False, help=readme["tooltips"]["track_experiments"]
#     )
    
#     # df = df.rename(columns={date_col: "ds", target_col: "y"})
#     # df = df[["ds", "y"]]
#     # st.write(df.head())
#     model = Prophet()
#     model.fit(df)
    
#     forecast_days = 30
#     future = model.make_future_dataframe(periods=forecast_days, freq='D')
#     forecast = model.predict(future)
#     # Visualizations
#     if evaluate | make_future_forecast:
#         st.write("# 1. Overview")
#         fig1 = plot_plotly( 
#         model,
#         forecast,
#         ylabel=target_col,
#         # changepoints=True,
#         # trend=True,
#         # uncertainty=bool_param,
#     )
#     st.plotly_chart(fig1)
    
#     if evaluate:
#         st.write(
#             f'# 2. Evaluation of data'
#         )
#         fig2 = plot_components_plotly(model, forecast)
#         st.plotly_chart(fig2)


#         st.write("## Performance metrics")
#         y_true = df.y
#         y_pred = forecast.yhat[:df.shape[0]]
#         mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
#         mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)

#         st.write("MAE: ", round(mae,2))
#         st.write("MAPE: ", round(mape,2))

 


# forecast_days = 30
# if make_future_forecast:
#     future = model.make_future_dataframe(periods=forecast_days, freq='D')
#     forecast = model.predict(future)
# else:
#     forecast = None
    
# st.write(forecast)
