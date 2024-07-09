from typing import Any, Dict, List

import datetime

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score

def display_metrics(df, pred):
    metrics = pd.DataFrame()
    y_true = df.y
    y_pred = pred.yhat[:df.shape[0]]
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    metrics['MAE'] = round(mae,2)
    metrics['MAPE'] = round(mape,2)
    return metrics

def plot_performance():
    style = config["style"]
    
    st.write("## Performance metrics")
    report = display_metrics(df, pred)
    # display_expanders_performance(use_cv, dates, resampling, style, readme)
    # display_expander(readme, "helper_metrics", "How to evaluate my model?", True)
    
    # st.write("### Global performance")
    # report = display_global_metrics(evaluation_df, eval, dates, resampling, use_cv, config, report)
    
    # st.write("### Deep dive")
    # report = plot_detailed_metrics(metrics_df, metrics_dict, eval, use_cv, style, report)
    
    # st.write("## Error analysis")
    # display_expander(readme, "helper_errors", "How to troubleshoot forecasting errors?", True)
    # fig1 = plot_forecasts_vs_truth(evaluation_df, target_col, use_cv, style)
    # fig2 = plot_truth_vs_actual_scatter(evaluation_df, use_cv, style)
    # fig3 = plot_residuals_distrib(evaluation_df, use_cv, style)
    # st.plotly_chart(fig1)
    # st.plotly_chart(fig2)
    # st.plotly_chart(fig3)
    # report.append({"object": fig1, "name": "eval_forecast_vs_truth_line", "type": "plot"})
    # report.append({"object": fig2, "name": "eval_forecast_vs_truth_scatter", "type": "plot"})
    # report.append({"object": fig3, "name": "eval_residuals_distribution", "type": "plot"})
    # report.append({"object": evaluation_df, "name": "eval_data", "type": "dataset"})
    # report.append(
    #     {"object": metrics_df.reset_index(), "name": "eval_detailed_performance", "type": "dataset"}
    # )
    return report
