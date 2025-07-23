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

        # retrieve k* using id_y_k
        k_opt = int(pd.DataFrame(list(db.Y_opt.find({"_id": ObjectId(id_y_k)}, {"Optim": 1})))["Optim"][0][0]["k"])

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

    # Assume selected_model is something like 'call_gspc_compare' or 'put_something_else'
    if selected_model.startswith("call"):
        df = add_k_low(df, k=k_opt)
    elif selected_model.startswith("put"):
        df = add_k_high(df, k=k_opt)
    else:
        raise ValueError(f"Invalid model type in selected_model: {selected_model}")

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

    with st.expander("ℹ️ Explanation – Signal Types (Click to Expand)"):
        st.markdown(f"""
        ### ⚙️ Overview of Available Signal Calculations

        We generate several flavours of **Signal**, each adding a different layer of information or weighting.  
        Use the simplest version when you just need a quick market pulse, and the more elaborate versions when you want deeper context.

        #### 1️⃣ `Signal_simple`
        * **What it is:** A straight count of how many base models (`S1…S10`) fire **positive** at the same time.  
        * **Interpretation:**  
        - `0` → no models agree  
        - `10` → all models agree  
        * No weighting or scaling—pure consensus.

        ---

        #### 2️⃣ `Signal_weighted`
        * **Method:** Exponentially‑weighted average of `S1…S10`.  
        - `S1` gets the highest weight, `S10` the lowest (`0.9⁰, 0.9¹, …`).  
        * **Use‑case:** Quick composite signal that still emphasises the more reliable early slots.

        ---

        #### 3️⃣ `Signal_balanced`
        * **Adds three real‑world adjustments** to each `Si` before weighting:  
        1. **`Nᵢ`** – number of observations that produced `Si` (scaled to yearly frequency).  
        2. **`Rᵢ`** – _out‑of‑sample return_ historically realised when taking `Si`.  
        3. **`Wᵢ`** – domain‑specific weight (e.g., asset importance).  
        * Then applies the same exponential weights as above.  
        * **Goal:** Reward signals that have fired often **and** paid off in the past.

        ---

        #### 4️⃣ `Signal_balanced_excl_R`
        * Same as *balanced*, but **omits the `R` component**.  
        * Choose this when historical return data are missing or deemed noisy.

        ---

        #### 5️⃣ `Signal_balanced_nw`
        * A “flat” version of *balanced*: **no exponential weights**—all `S1…S10` treated equally.  
        * Still includes `N`, `R`, and `W`.

        ---

        #### 6️⃣ `Signal_balanced_nw_excl_R`
        * Like the flat version above, but **drops `R`** as well.  
        * Useful when you want frequency (`N`) and custom weights (`W`) only.

        ---
        📌 **Tip:**  
        *For most production runs, `Signal_balanced` offers the best trade‑off between simplicity and historical effectiveness.  
        Use `Signal_simple` for a quick sanity check, or when model metadata (`N`, `R`, `W`) are unavailable.*
        """)


    df_signal, signal_type = select_signal_type(df)

    # Display results in Streamlit
    st.write(f"Selected Signal Type: {signal_type}")

    plot_signal_chart(df_signal, selected_model, signal_type)

    # Assume df_signal is your DataFrame with a "Signal" column
    min_signal = df_signal["Signal"].min()
    max_signal =  df_signal["Signal"].max()
    signal_range = max_signal - min_signal

    # First Signal threshold where p (HGT) is getting max smaller
    min_pos_signal, stats = optimize_signal_threshold_by_hypergeometric(df_signal)

    print("Best threshold:", min_pos_signal)
    print("Stats for best threshold:", stats[min_pos_signal])

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
            - The **default value** is set to the **Signal threshold** that **maximizes the frequency of statistically significant associations** between `Signal` and positive `R_t`, based on a rolling *hypergeometric test*.
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

    plot_ohlc_volume(df_signal, selected_model, signal_type, signal_threshold=round(selected_signal,3))

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

    plot_p_value(df_signal_filtered, selected_model, signal_threshold=round(selected_signal,3), window = selected_window)

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

    plot_win_rate(df_signal_filtered, df_signal, selected_model, selected_signal, selected_window)

    st.divider()  # Adds a separation line

    # Create a single-value slider with a dynamic step size
    selected_slo = st.slider(
        "Stop Loss",
        min_value=0.001,
        max_value=max(df_signal["slo"]),
        value=max(df_signal["slo"]),  # default starting value
        step=0.01,
        format="%.3f"  # display three decimal places
    )

    st.divider()  # Adds a separation line

    # Create a single-value slider with a dynamic step size
    # --- Extract 'sli' value from df_signal row 0 ---
    sli_val = None
    if "sli" in df_signal.columns:
        sli_val = df_signal["sli"].max()
    elif "sli1" in df_signal.columns and "sli2" in df_signal.columns:
        sli_val = max(df_signal["sli1"].max(), df_signal["sli2"].max())
    else:
        st.warning("No 'sli', 'sli1', or 'sli2' found in df_signal.")
        sli_val = 0.1  # fallback default value

    # --- Streamlit slider for 'sli' ---
    selected_sli = st.slider(
        "Stop Limit",
        min_value=0.001,
        max_value=float(sli_val),
        value=float(sli_val),  # default to max
        step=0.001,
        format="%.3f"
    )


    st.divider()  # Adds a separation line

    # Create a single-value slider with a dynamic step size
    selected_leverage = st.slider(
        "Leverage",
        min_value=1,
        max_value=25,
        value=4,  # default starting value
        step=1
    )

    st.divider()  # Adds a separation line

    # Create a single-value slider with a dynamic step size
    selected_cost = st.slider(
        "Cost for a trade as decimal",
        min_value=0.0,
        max_value=0.05,
        value=0.00624,  # default starting value
        step=0.00001,
        format="%.5f"
    )

    df_bench_result = compute_benchmark_returns(df_signal,
                                                     selected_leverage, 
                                                     selected_cost,
                                                     k_opt,
                                                     selected_model
                                                     )

    df_strategy_result = compute_strategy_returns(
                df_signal,
                selected_leverage,
                selected_cost,
                k_opt,
                selected_signal,
                selected_slo,
                selected_sli,
                selected_model
                )

    df_result = pd.merge(
            df_bench_result,
            df_strategy_result[["date", "R_t_temp", "R_t_comp"]],
            on="date",
            how="left"
        )

    # Move result values to the time they are realized (t + k_opt)
    df_result["R_t_bench_temp"] = df_result["R_t_bench_temp"].shift(k_opt)
    df_result["R_t_bench_comp"] = df_result["R_t_bench_comp"].shift(k_opt)
    df_result["R_t_temp"] = df_result["R_t_temp"].shift(k_opt)
    df_result["R_t_comp"] = df_result["R_t_comp"].shift(k_opt)
    df_result["R_t_comp"] = df_result["R_t_comp"].ffill()

    plot_strategy_vs_benchmark(
    df_result,
    selected_model,
    k_opt,
    selected_slo,
    selected_sli,
    selected_leverage,
    selected_cost
    )

    df_strategy_stats = analyze_weekly_returns_df(df_result, selected_signal, return_column="R_t_comp")
    df_bench_stats = analyze_weekly_returns_df(df_result, return_column="R_t_bench_comp")

    # Combine both for comparison
    df_comparison = pd.concat([df_strategy_stats, df_bench_stats], axis=1)
    #st.dataframe(df_comparison)

    plot_metrics_comparison(df_comparison)

    if "username" in st.session_state and st.session_state["username"] == master:
        for col in [f"M{i}" for i in range(1, 11)]:
            df_signal[col] = df_signal[col].astype(str)

        df_comparison_reset = df_comparison.reset_index().rename(columns={'index': 'metric'})
        st.dataframe(df_comparison_reset.round(4))

        print(df_comparison_reset.round(4))

        st.dataframe(df_result[["date","Target","R_t", "Signal","open","low","high","close","R_t_bench_temp","R_t_bench_comp","R_t_temp","R_t_comp", "p", "N", "K", "n_draws", "k"]])
     
    st.markdown("<hr><p style='text-align:center;'>Data source: Yahoo Finance</p>", unsafe_allow_html=True)

