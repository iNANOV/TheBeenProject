import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import morethemes as mt
import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import check_password_hash
from functions import *


# Streamlit Login System
usr = st.secrets["mongodb"]["usr"]
pwd = st.secrets["mongodb"]["pwd"]
url = st.secrets["mongodb"]["url"]
database = st.secrets["mongodb"]["database"]

# MongoDB connection
@st.cache_resource
def get_db_client():
    db_uri = f"mongodb://{usr}:{pwd}@{url}/{database}?authSource=admin&retryWrites=true&w=majority"
    client = MongoClient(db_uri)
    return client[database]

db = get_db_client()
users_collection = db["users"]  # Users collection

def show_overview_page():
    st.title("Overview Page")
    st.write("Welcome to the Overview page! Here, you can see the main insights and visualizations.")

    # Fetch all models containing "_master" in "Sim" column
    master_models = list(db.Models.find({"Sim": {"$regex": "_master"}}, {"Sim": 1}))

    if not master_models:
        st.error("No master models found in the database.")
        st.stop()

    # Convert model names to a list for selection
    model_options = {model["Sim"]: model["_id"] for model in master_models}
    selected_model = st.selectbox("Select a master model:", list(model_options.keys()))

    # Fetch the corresponding ObjectId from selection
    selected_model_id = model_options[selected_model]

    # Retrieve the selected model data
    try:
        data = db.Models.find_one({"_id": ObjectId(selected_model_id)}, {"Now": 1, "_id": 0})
        if not data or "Now" not in data:
            st.error("No data found for the selected model.")
            st.stop()

        df = pd.DataFrame(data["Now"])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    st.write(f"Showing data for: **{selected_model}**")

    # calculate simple Signal
    df["Signal_simple"] = df.filter(regex="^S\d+").apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # calculate weighted average Signal
    df = calculate_weighted_signal(df)

    # calculate weighted balanced Signal
    df = calculate_balanced_signal(df)

    st.write(df)

    plot_line_with_dots(df, selected_model)

    plot_unique_ids(df, selected_model)

    df_f = df.copy()

    # Step 1: Retrieve the '_id' and 'pycaret_function' from the Models collection in MongoDB
    model_mapping = pd.DataFrame(list(db.Models.find({}, {"_id": 1, "pycaret_function": 1})))

    # Step 2: Convert the retrieved data into a dictionary for fast lookups (_id -> pycaret_function)
    function_lookup = dict(zip(model_mapping["_id"], model_mapping["pycaret_function"]))

    # Step 3: Loop through columns M1 to M10 in df, and map the corresponding pycaret_function to new columns F1 to F10
    for i in range(1, 11):  # Loop over M1 to M10
        df_f[f"F{i}"] = df[f"M{i}"].map(function_lookup)  # Map each M{i} column to the corresponding pycaret_function

    # Ensure the 'date' column stays as a date (not datetime)
    #df['date'] = pd.to_datetime(df['date']).dt.date


    plot_function_value_distribution_by_date(df_f, selected_model)

    df_signal, signal_type = select_signal_type(df)

    #df['date'] = pd.to_datetime(df['date']).dt.date

    #df_signal['date'] = pd.to_datetime(df_signal['date']).dt.date

    # Display results in Streamlit
    st.write(f"Selected Signal Type: {signal_type}")
    st.dataframe(df_signal)

    plot_signal_chart(df_signal, selected_model, signal_type)

    # Assume df_signal is your DataFrame with a "Signal" column
    min_signal = df_signal["Signal"].min()#float(df_signal["Signal"].min())
    max_signal =  df_signal["Signal"].max()#float(df_signal["Signal"].max())
    signal_range = max_signal - min_signal

    # Determine a step size based on the range:
    if signal_range == 10:
        step = 1
    else:
        step = 0.001

    # Create a single-value slider with a dynamic step size
    selected_signal = st.slider(
        "Select a Signal Value",
        min_value=min_signal,
        max_value=max_signal,
        value=min_signal,  # default starting value
        step=step,
        format="%.3f"  # display three decimal places
    )

    plot_ohlc_volume(df_signal, selected_model, signal_type, signal_threshold=selected_signal)

    # Add explanation inside an expander
    with st.expander("ℹ️ Explanation: Hypergeometric Test (Click to Expand)"):
        st.markdown("""
        ### 📊 Rolling Window Calculation  

        For each window of size **`window`**:

        - **N** → Total rows in the window  
        - **K** → Count where **R_t > 0**  
        - **n_draws** → Count where **Signal ≥ signal_threshold**  
        - **k** → Count where **both R_t > 0 & Signal ≥ signal_threshold**  

        #### 🧮 Formula:  
        ```
        p = hypergeom.sf(k-1, N, K, n_draws)
        ```

        📌 **Stored in column:** `"p"`
        """)

    st.divider()  # Adds a separation line

    # Create a single-value slider with a dynamic step size
    selected_window = st.slider(
        "Select a moving window Value",
        min_value=12,
        max_value=52,
        value=52,  # default starting value
        step=step
    )

    df_signal = calculate_p_value_rolling(df_signal, signal_threshold=selected_signal, window = selected_window)

    # filter out the forecasts
    df_signal_filtered = df_signal.query("not (Target == 0 and R_t == 0)")

    st.dataframe(df_signal_filtered)

    plot_p_value(df_signal_filtered, selected_model, signal_threshold=selected_signal, window = selected_window)

      # Add explanation inside an expander
    with st.expander("ℹ️ Explanation: Win rate (Click to Expand)"):
        st.markdown("""
        ### 📊 Rolling Window Calculation  

        For each window of size **`window`**:

        - **N** → Total rows in the window  
        - **n_draws** → Count where **Signal ≥ signal_threshold**  
        - **k** → Count where **both R_t > 0 & Signal ≥ signal_threshold**  

        #### 🧮 Formula:  
        ```
        win_rate = k / n_draws
        ```
        """)

    st.divider()  # Adds a separation line


    plot_win_rate(df_signal_filtered, selected_model, signal_threshold=selected_signal, window = selected_window)

    st.markdown("<hr><p style='text-align:center;'>Data source: Yahoo Finance</p>", unsafe_allow_html=True)

