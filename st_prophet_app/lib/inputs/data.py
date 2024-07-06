from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st


import pandas as pd
import streamlit as st


def input_columns(df):
    date_col = st.selectbox("Date column",sorted(df.columns))
    target_col = st.selectbox( "Target column", sorted(set(df.columns) - {date_col}) )
    return date_col, target_col


