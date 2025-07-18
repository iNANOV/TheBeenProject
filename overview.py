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

master = st.secrets["users"]["master"]

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

        id_y = pd.DataFrame(list(db.Models.find({"_id": ObjectId(selected_model_id)}, {"ID_Y"})))["ID_Y"][0]

        id_y_omega =  pd.DataFrame(list(db.Y_opt.find({"_id": ObjectId(id_y)}, {"ID_Y_Omega"})))["ID_Y_Omega"][0]

        id_y_k = pd.DataFrame(list(db.Y_opt.find({"_id": ObjectId(id_y_omega)}, {"ID_Y_k"})))["ID_Y_k"][0]

        # retrieve last sli and slo from data

        # retrieve k* using id_y_k
        k_opt = 4 

        if not data or "Now" not in data:
            st.error("No data found for the selected model.")
            st.stop()

        df = pd.DataFrame(data["Now"])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    st.write(f"Showing data for: **{selected_model}**")

    # calculate simple Signal
    df["Signal_simple"] = df.filter(regex=r"^S\d+").apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # calculate weighted average Signal
    df = calculate_weighted_signal(df)

    # calculate weighted balanced Signal
    df = calculate_balanced_signal(df)

    # calculate weighted balanced Signal excluding R
    df = calculate_balanced_signal_excl_R(df)

    # calculate weighted balanced Signal not weighted
    df = calculate_balanced_not_weighted_signal(df)

    # calculate weighted balanced Signal not weighted excluding R
    df = calculate_balanced_not_weighted_signal_excl_R(df)

    # calculate mean winrate
    #df = calculate_mean_winrate(df)

    # in case  R_i < 0
   # if max(df['Signal_balanced']) == 0:
    #    df['Signal_balanced'] = df["Signal_balanced"] * -1

    #if max(df['Signal_balanced_nw']) == 0:
     #   df['Signal_balanced_nw'] = df["Signal_balanced"] * -1

    # add k_low column
    #df = add_k_low(df, k = k_opt)

     # add k_high column
    #df = add_k_high(df, k = k_opt)

    # Assume selected_model is something like 'call_gspc_compare' or 'put_something_else'
    if selected_model.startswith("call"):
        df = add_k_low(df, k=k_opt)
    elif selected_model.startswith("put"):
        df = add_k_high(df, k=k_opt)
    else:
        raise ValueError(f"Invalid model type in selected_model: {selected_model}")


    #st.write(df)

    if "username" in st.session_state and st.session_state["username"] == master:

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

        plot_function_value_distribution_by_date(df_f, selected_model)

    # Define the k_opt variable
    # k_opt = 4  # You can change this value as needed

    with st.expander("ℹ️ Explanation: Signal (Click to Expand)"):
        st.markdown(f"""
        ### 📈 What is a Signal?  

        **Signal** is the forecasting trigger used for trading decisions. Here's how it works:

        - When **Signal > 0**, the recommendation is to **buy** on the next working day (usually Monday) at the **open price**.
        - The trade is **held** for a maximum of **{k_opt} weeks** (until one of the following scenarios occurs):
            - **Stop limit** at **0.041**  
            - **Stop loss** at **0.21**  
            - The **close price** on the last working week day (usually Friday).

        #### 📝 Example:
        - **Signal > 0** on **5th January 2024** → **Buy on 8th January** (Monday) at the open price  
        - **Sell latest on 2nd February 2024** at close price (if stop limit or stop loss were not reached during the {k_opt} weeks)

        📌 **Note:** This methodology is used for trading based on predicted market movements, with clear exit strategies to limit losses or lock in profits.
        """)

    df_signal, signal_type = select_signal_type(df)

    # Display results in Streamlit
    st.write(f"Selected Signal Type: {signal_type}")
    #st.dataframe(df_signal)

    plot_signal_chart(df_signal, selected_model, signal_type)

    # Assume df_signal is your DataFrame with a "Signal" column
    min_signal = df_signal["Signal"].min()#float(df_signal["Signal"].min())
    max_signal =  df_signal["Signal"].max()#float(df_signal["Signal"].max())
    signal_range = max_signal - min_signal

    # First Signal where all R_t are getting positive 
    min_pos_signal = find_min_signal_threshold(df_signal)

    if min_pos_signal == None:
        min_pos_signal = min_signal
    else:
        min_pos_signal = min_pos_signal
    
    # Determine a step size based on the range:
    #if signal_range == 10:
    if isinstance(signal_range, np.integer):
        print("here")
        step = 1
        min_signal = int(min_signal)
        max_signal = int(max_signal)
        min_pos_signal = int(min_pos_signal)

    else:
        step = 0.001
        min_signal = float(min_signal)
        max_signal = float(max_signal)
        min_pos_signal = float(min_pos_signal)

    print("min signal: " + str(min_signal))
    print("max signal: " + str(max_signal))

    # Extendible Info Box for Slider
    with st.expander("ℹ️ Explanation: Slider 'Select a Signal Value'"):
        st.markdown(
            """
            🔧 **What is the Slider?**
            
            The slider allows you to dynamically select a **Signal value** within the range of your data (i.e., from the minimum to maximum Signal value). The selected value can be used for filtering or setting the Signal threshold.
            
            ### **How does it work?**
            1. **Signal Range Calculation**:
            - The range of values for the Signal is determined by finding the **minimum** and **maximum** values of the `Signal`.
            
            2. **Step Size**:
            - The slider's step size is determined dynamically based on the Signal range:
                - **If the range is 10**, the step size is **1**.
                - **Otherwise**, the step size is **0.001** for more granular selection.
            
            3. **Slider Configuration**:
            - The slider allows the user to select a value within the calculated Signal range.
            - The **default value** is set to the **first Signal value** where **all `R_t` values are greater than or equal to 0**.
            - The slider is displayed with a step size and formatted to show **3 decimal places**.
            """
        )

    # Create a single-value slider with a dynamic step size
    selected_signal = st.slider(
        "Select a Signal Value",
        min_value=min_signal,
        max_value=max_signal,
        value = min_pos_signal,  # default starting value
        step=step,
        format="%.3f"  # display three decimal places
    )

    #a1 = df_signal.loc[(df_signal["R_t"] > 0) & (df_signal["Signal"] > 0), ["date", "R_t", "Signal"]]

    #st.dataframe(a1)

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
        p = hypergeom.sf(k-1, N, K, n_draws)      ```

        """)

    st.divider()  # Adds a separation line

    # Create a single-value slider with a dynamic step size
    selected_window = st.slider(
        "Select a moving window Value",
        min_value=12,
        max_value=52,
        value=52,  # default starting value
        step=1
    )

    df_signal = calculate_p_value_rolling(df_signal, signal_threshold=selected_signal, window = selected_window)

    # filter out the forecasts
    df_signal_filtered = df_signal.query("not (Target == 0 and R_t == 0)")

   # st.dataframe(df_signal_filtered)

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

    if "username" in st.session_state and st.session_state["username"] == master:
        for col in [f"M{i}" for i in range(1, 11)]:
            df_signal[col] = df_signal[col].astype(str)

        st.dataframe(df_signal)

        st.dataframe(df_signal[["date", "R_t", "Signal"]].assign(R_t=df_signal["R_t"].round(3)))

    st.markdown("<hr><p style='text-align:center;'>Data source: Yahoo Finance</p>", unsafe_allow_html=True)

