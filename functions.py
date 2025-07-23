import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
import morethemes as mt
import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import check_password_hash
from matplotlib.cm import ScalarMappable 
from scipy.stats import hypergeom


def add_k_low(df, k):
    """
    For each row t in df, compute:
       k_low_t = (min(low_{t+1}, ..., low_{t+k}) - open_{t+1}) / open_{t+1}
    and add the result as a new column 'k_low'.
    If there are fewer than k rows after t, the result is NaN.
    
    Parameters:
      df : pandas DataFrame with columns 'date', 'open', and 'low'
      k  : int, the number of future rows to include in the calculation
    
    Returns:
      The DataFrame with an added 'k_low' column.
    """
    results = []
    n = len(df)
    for t in range(n):
        # Check if there are at least k rows after row t
        if t + k < n:
            window = df['low'].iloc[t+1:t+k+1]
            open_next = df['open'].iloc[t+1]
            k_low = (window.min() - open_next) / open_next
            results.append(k_low)
        else:
            results.append(np.nan)
    df['k_low'] = results
    return df

def add_k_high(df, k):
    """
    For each row t in df, compute:
       k_high_t = (max(high_{t+1}, ..., high_{t+k}) - open_{t+1}) / open_{t+1}
    and add the result as a new column 'k_high'.
    If there are fewer than k rows after t, the result is NaN.
    
    Parameters:
      df : pandas DataFrame with columns 'date', 'open', and 'high'
      k  : int, the number of future rows to include in the calculation
    
    Returns:
      The DataFrame with an added 'k_high' column.
    """
    results = []
    n = len(df)
    for t in range(n):
        if t + k < n:
            window = df['high'].iloc[t+1:t+k+1]
            open_next = df['open'].iloc[t+1]
            k_high = (window.max() - open_next) / open_next
            results.append(k_high)
        else:
            results.append(np.nan)
    df['k_high'] = results
    return df

def calculate_weighted_signal(df):
    # Select S1 to S10 columns
    S_columns = df.filter(regex=r"^S\d+")
    
    # Number of S columns
    num_S = S_columns.shape[1]
    
    # Create exponential weights (S1 has highest weight, S10 lowest)
    weights = np.array([0.9 ** (i-1) for i in range(1, num_S+1)])
    
    # Normalize weights so they sum to 1 (optional)
    weights /= weights.sum()
    
    # Convert S columns to numeric
    S_numeric = S_columns.apply(pd.to_numeric, errors="coerce")
    
    # Compute weighted sum
    df["Signal_weighted"] = S_numeric.dot(weights)
    
    return df

def calculate_balanced_signal(df):
    # Ensure required columns exist
    required_S_cols = {f"S{i}" for i in range(1, 11)}
    required_N_cols = {f"N{i}" for i in range(1, 11)}
    required_R_cols = {f"R{i}" for i in range(1, 11)}
    required_W_cols = {f"W{i}" for i in range(1, 11)}
    
    required_cols = required_S_cols | required_N_cols | required_R_cols | required_W_cols
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select S1 to S10 columns
    S_columns = df.filter(regex=r"^S\d+")
    
    # Number of S columns (should be 10 for S1 to S10)
    num_S = S_columns.shape[1]
    
    # Create exponential weights (S1 has highest weight, S10 lowest)
    weights = np.array([0.9 ** (i-1) for i in range(1, num_S+1)])
    
    # Normalize weights so they sum to 1 (optional, to keep scale consistent)
    weights /= weights.sum()
    
    # Compute weighted sum for all S1 to S10
    signal_balanced = sum(
        df[f"S{i}"] * (df[f"N{i}"] / 52) * df[f"R{i}"] * weights[i-1] * df[f"W{i}"]
        for i in range(1, num_S + 1)
    )
    
    # Store result in the dataframe
    df["Signal_balanced"] = signal_balanced
    
    return df

def calculate_balanced_signal_excl_R(df):
    # Ensure required columns exist
    required_S_cols = {f"S{i}" for i in range(1, 11)}
    required_N_cols = {f"N{i}" for i in range(1, 11)}
    required_W_cols = {f"W{i}" for i in range(1, 11)}
    
    required_cols = required_S_cols | required_N_cols | required_W_cols
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select S1 to S10 columns
    S_columns = df.filter(regex=r"^S\d+")
    
    # Number of S columns (should be 10 for S1 to S10)
    num_S = S_columns.shape[1]
    
    # Create exponential weights (S1 has highest weight, S10 lowest)
    weights = np.array([0.9 ** (i-1) for i in range(1, num_S+1)])
    
    # Normalize weights so they sum to 1 (optional, to keep scale consistent)
    weights /= weights.sum()
    
    # Compute weighted sum for all S1 to S10
    signal_balanced_excl_R = sum(
        df[f"S{i}"] * (df[f"N{i}"] / 52) * weights[i-1] * df[f"W{i}"]
        for i in range(1, num_S + 1)
    )
    
    # Store result in the dataframe
    df["Signal_balanced_excl_R"] = signal_balanced_excl_R
    
    return df

def calculate_balanced_not_weighted_signal(df):
    # Ensure required columns exist
    required_S_cols = {f"S{i}" for i in range(1, 11)}
    required_N_cols = {f"N{i}" for i in range(1, 11)}
    required_R_cols = {f"R{i}" for i in range(1, 11)}
    required_W_cols = {f"W{i}" for i in range(1, 11)}
    
    required_cols = required_S_cols | required_N_cols | required_R_cols | required_W_cols
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select S1 to S10 columns
    S_columns = df.filter(regex=r"^S\d+")
    
    # Number of S columns (should be 10 for S1 to S10)
    num_S = S_columns.shape[1]
    
    # Create exponential weights (S1 has highest weight, S10 lowest)
    #weights = np.array([0.9 ** (i-1) for i in range(1, num_S+1)])
    
    # Normalize weights so they sum to 1 (optional, to keep scale consistent)
    #weights /= weights.sum()
    
    # Compute weighted sum for all S1 to S10
    signal_balanced_nw = sum(
        df[f"S{i}"] * (df[f"N{i}"] / 52) * df[f"R{i}"] * df[f"W{i}"] 
        for i in range(1, num_S + 1)
    )
    
    # Store result in the dataframe
    df["Signal_balanced_nw"] = signal_balanced_nw
    
    return df

def calculate_balanced_not_weighted_signal_excl_R(df):
    # Ensure required columns exist
    required_S_cols = {f"S{i}" for i in range(1, 11)}
    required_N_cols = {f"N{i}" for i in range(1, 11)}
    required_W_cols = {f"W{i}" for i in range(1, 11)}
    
    required_cols = required_S_cols | required_N_cols | required_W_cols 
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select S1 to S10 columns
    S_columns = df.filter(regex=r"^S\d+")
    
    # Number of S columns (should be 10 for S1 to S10)
    num_S = S_columns.shape[1]
    
    # Create exponential weights (S1 has highest weight, S10 lowest)
    #weights = np.array([0.9 ** (i-1) for i in range(1, num_S+1)])
    
    # Normalize weights so they sum to 1 (optional, to keep scale consistent)
    #weights /= weights.sum()
    
    # Compute weighted sum for all S1 to S10
    signal_balanced_nw_excl_R = sum(
        df[f"S{i}"] * (df[f"N{i}"] / 52) * df[f"W{i}"] 
        for i in range(1, num_S + 1)
    )
    
    # Store result in the dataframe
    df["Signal_balanced_nw_excl_R"] = signal_balanced_nw_excl_R
    
    return df

def calculate_mean_winrate(df):
    # Ensure required columns exist
    required_W_cols = {f"W{i}" for i in range(1, 11)}
    
    if not required_W_cols.issubset(df.columns):
        missing_cols = required_W_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Number of S columns (assumed 10)
    num_S = 10

    # Compute mean of row-wise sum across W1 to W10
    signal_mean_winrate = df[[f"W{i}" for i in range(1, num_S + 1)]].sum(axis=1).mean()
    
    # Store result in the DataFrame
    df["Signal_mean_winrate"] = signal_mean_winrate

    return df

def select_signal_type(df):
    # Dynamically find all columns that start with "Signal_"
    signal_columns = [col for col in df.columns if col.startswith("Signal_")]

    if not signal_columns:
        st.error("No Signal_ columns found in the DataFrame!")
        return df, None

    # Extract signal names (removing "Signal_" prefix)
    signal_options = {col.replace("Signal_", ""): col for col in signal_columns}

    # Streamlit dropdown (selectbox)
    selected_option = st.selectbox("Select Signal Type:", list(signal_options.keys()))

    # Get the corresponding column name
    signal_type = signal_options[selected_option]

    # Keep all other columns but rename the selected signal column to "Signal"
    df_signal = df.copy()
    df_signal = df_signal.rename(columns={signal_type: "Signal"})

    return df_signal, signal_type

def plot_line_with_dots(df, selection):
    # Ensure 'Sim_datetime' is in proper datetime format
    df['Sim_datetime'] = pd.to_datetime(df['Sim_datetime'])

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Sim_datetime'], df['Sim_trials'], marker='o', linestyle='-', color='b', label='Sim_trials')

    # Set title and labels
    ax.set_title(f"Simulation {selection} progressing till adding new best models", fontsize=9, fontweight='normal', family='sans-serif')
    #ax.set_xlabel('Datetime', fontsize=9)
    ax.set_ylabel('Simulation Trials', fontsize=9)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def calculate_unique_ids(df):
    # Initialize the 'Munique' column
    df['Munique'] = 0
    seen_ids = set()  # Set to track unique IDs encountered so far

    # List of column names to check (M1 to M10)
    m_columns = [f'M{i}' for i in range(1, 11)]
    
    # Loop over the rows to calculate unique ids
    for i in range(len(df)):
        # Extract the relevant columns (M1 to M10)
        current_ids = set(df.loc[i, m_columns])
        
        # Identify new unique IDs for this row (that haven't been seen before)
        new_unique_ids = current_ids - seen_ids
        
        # Update the set of seen IDs with the new unique ones
        seen_ids.update(new_unique_ids)
        
        # Store the count of new unique IDs in the 'Munique' column
        df.at[i, 'Munique'] = len(new_unique_ids)
    
    # Calculate the cumulative sum of the 'Munique' column
    df['Munique'] #= df['Munique']
    
    return df

def plot_unique_ids(df, selection):

    df = calculate_unique_ids(df)

    df['Munique_Cumsum'] = df['Munique'].cumsum()

    # Ensure 'Sim_datetime' is in proper datetime format
    df['Sim_datetime'] = pd.to_datetime(df['Sim_datetime'])

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Sim_datetime'], df['Munique_Cumsum'], marker='o', linestyle='-', color='b', label='Sim_trials')

    # Set title and labels
    ax.set_title(f"Adding new best models from {selection} simulation", fontsize=9, fontweight='normal', family='sans-serif')
    #ax.set_xlabel('Datetime', fontsize=9)
    ax.set_ylabel('Cumsum', fontsize=9)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def find_min_signal_threshold(df: pd.DataFrame) -> float:
    """
    Finds the minimal threshold for the Signal column such that all R_t values corresponding to Signal >= threshold are >= 0.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing Signal and R_t columns.

    Returns:
    - float: The minimal threshold for Signal such that all R_t values corresponding to Signal >= threshold are >= 0.
    """
    
    # Get unique sorted Signal values in ascending order
    unique_signals = sorted(df['Signal'].unique())

    # Iterate through the unique signal values
    for threshold in unique_signals:
        # Filter the DataFrame where Signal >= threshold
        filtered_df = df[df['Signal'] >= threshold]
        
        # Check if all R_t values for this threshold are >= 0
        if (filtered_df['R_t'] >= 0).all():
            return threshold
    
    # If no threshold is found, return None or handle accordingly
    return None

def find_optimal_signal_threshold_by_rt(df: pd.DataFrame, method: str = "mean") -> float:
    """
    Finds the Signal threshold that maximizes R_t (mean or sum) for Signal >= threshold.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Signal' and 'R_t' columns.
    - method (str): "mean" or "sum" — defines the optimization criterion.

    Returns:
    - float: Signal threshold that maximizes the chosen R_t aggregate.
    """

    assert method in {"mean", "sum"}, "method must be either 'mean' or 'sum'"

    # Get unique sorted signal values (ascending)
    unique_signals = sorted(df["Signal"].unique())

    best_threshold = None
    best_value = float("-inf")

    for threshold in unique_signals:
        filtered = df[df["Signal"] >= threshold]

        if not filtered.empty:
            rt_value = filtered["R_t"].mean() if method == "mean" else filtered["R_t"].sum()
            if rt_value > best_value:
                best_value = rt_value
                best_threshold = threshold

    return best_threshold

def rolling_hypergeometric_pvals(df, window=52, signal_threshold=0.5):
    """
    Performs a rolling hypergeometric test.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'Signal' and 'R_t' columns.
    - window (int): Rolling window size.
    - signal_threshold (float): Threshold for Signal to count as a draw.

    Returns:
    - List of p-values for each window (NaN for early rows).
    """
    p_values = [np.nan] * len(df)

    for i in range(window - 1, len(df)):
        window_df = df.iloc[i - window + 1:i + 1]

        N = len(window_df)
        K = (window_df["R_t"] > 0).sum()
        n_draws = (window_df["Signal"] >= signal_threshold).sum()
        k = ((window_df["Signal"] >= signal_threshold) & (window_df["R_t"] > 0)).sum()

        # Hypergeometric test: probability of ≥ k successes in n_draws given population of N with K successes
        if N > 0 and K > 0 and n_draws > 0:
            p_val = hypergeom.sf(k - 1, N, K, n_draws)
        else:
            p_val = np.nan

        p_values[i] = p_val

    return p_values

def optimize_signal_threshold_by_hypergeometric(df, window=52, signal_range=None, alpha_threshold=0.05):
    """
    Optimizes Signal threshold to minimize p-values over rolling hypergeometric tests.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'Signal' and 'R_t' columns.
    - window (int): Rolling window size.
    - signal_range (list or None): List of candidate signal thresholds (default: all unique Signal values).
    - alpha_threshold (float): p-value threshold to count as "significant".

    Returns:
    - float: Optimal signal threshold.
    - dict: Summary stats per threshold.
    """
    if signal_range is None:
        signal_range = sorted(df['Signal'].unique())

    best_threshold = None
    best_score = -1
    results = {}

    for threshold in signal_range:
        p_vals = rolling_hypergeometric_pvals(df, window=window, signal_threshold=threshold)
        p_vals = np.array(p_vals)

        # Count how often p-value is below the significance level
        significant_count = np.sum((p_vals < alpha_threshold) & ~np.isnan(p_vals))
        total_tested = np.sum(~np.isnan(p_vals))
        ratio_significant = significant_count / total_tested if total_tested > 0 else 0

        results[threshold] = {
            "significant_count": significant_count,
            "total_windows": total_tested,
            "ratio_significant": ratio_significant
        }

        if ratio_significant > best_score:
            best_score = ratio_significant
            best_threshold = threshold

    return best_threshold, results

def plot_function_value_distribution_by_date_old(df, selection):
    """Plots how often the function names appear over time for F1 to F10 columns, with consistent colors in the legend."""
    
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Melt the dataframe to combine all F1 to F10 into one column
    melted_df = df.melt(id_vars=['date'], value_vars=[f'F{i}' for i in range(1, 11)], var_name='F_column', value_name='Function')

    # Count occurrences of each function value by date
    function_value_counts = melted_df.groupby(['date', 'Function']).size().unstack(fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the stacked bar chart
    function_value_counts.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', width=1.0)

    # Adjust the legend to show only the function names, and ensure consistent coloring
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Method", loc='upper left')

    # Set title and labels
    ax.set_title(f"Method Value Distribution over Time for {selection}", fontsize=9, fontweight='normal', family='sans-serif')
    
    # Remove x-axis label
    ax.set_xlabel('')  # This removes the label

    ax.set_ylabel('Counts of Method Occurrences', fontsize=9)

    # Rotate the x-axis labels for better readability
    ax.set_xticklabels(function_value_counts.index.strftime('%Y-%m-%d'), rotation=90, fontsize=9)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def plot_function_value_distribution_by_date(df, selection):
    """Plots how often the function names appear over time for F1 to F10 columns, with consistent colors in the legend."""

    import matplotlib.dates as mdates
    from matplotlib.cm import get_cmap

    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Melt the dataframe to combine all F1 to F10 into one column
    melted_df = df.melt(
        id_vars=['date'],
        value_vars=[f'F{i}' for i in range(1, 11)],
        var_name='F_column',
        value_name='Function'
    )

    # Count occurrences of each function value by date
    function_value_counts = melted_df.groupby(['date', 'Function']).size().unstack(fill_value=0)

    # Set up colors for each function
    method_list = function_value_counts.columns.tolist()
    cmap = get_cmap('tab20', len(method_list))
    colors = {method: cmap(i) for i, method in enumerate(method_list)}

    # Prepare plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(function_value_counts))
    dates = function_value_counts.index

    # Manually stack bars for each method
    for method in method_list:
        ax.bar(dates,
               function_value_counts[method],
               bottom=bottom,
               label=method,
               color=colors[method],
               width=pd.Timedelta(days=3))  # Narrower bars for better spacing
        bottom += function_value_counts[method].values

    # Format x-axis as real dates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=90, labelsize=9)

    ax.set_title(f"Method Value Distribution over Time for {selection}", fontsize=9, fontweight='normal', family='sans-serif')
    ax.set_ylabel('Counts of Method Occurrences', fontsize=9)
    ax.set_xlabel('')
    ax.legend(title="Method", loc='upper left', fontsize=9)
    ax.grid(True, axis='y')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_signal_chart_old(df, selection, signal):
    
    # Blue scale: Light blue for low values, dark blue for high values
    bar_cmap = LinearSegmentedColormap.from_list("lightblue_darkblue", ["lightblue", "darkblue"])
    
    # Define color mappings for OHLC and Volume plots
    ohlc_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    volume_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    
    # Create figure and axis for the bar chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Normalize Signal for color mapping
    norm = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())
    
    # Plot bars for R_t with color based on Signal
    bars = ax1.bar(df["date"], df["R_t"], color=bar_cmap(norm(df["Signal"])))
    
    # Set tick labels alignment and rotation
    #ax1.set_xticklabels(df["date"], rotation=90, ha='center')

    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df["date"], rotation=90, ha='center')

    ax1.tick_params(labelsize=9)
    ax1.grid(True)
    ax1.set_xlabel("")
    ax1.set_ylabel("R_t", fontsize=9, fontweight='normal', family='sans-serif')
    ax1.set_title(f"Master model based on {selection} simulation / {signal}", fontsize=9, fontweight='normal', family='sans-serif')
    

    # Add a color bar (legend) for Signal
    sm = plt.cm.ScalarMappable(cmap=bar_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Signal')
    
    # Adding values on top of bars
    for bar, sig in zip(bars, df["Signal"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{sig:.2f}',
                 ha='center', va='center', fontsize=9, rotation=90)
    
    # Add k_low values as red dots. They will be plotted on the same x-axis as the bars.
    # Adjust marker size (s) as needed.
    #ax1.scatter(df["date"], df["k_low"], color='red', marker='o', s=20, zorder=3, label='k_low')

    # Dynamically determine which column to use for scatter plot
    if "k_low" in df.columns:
        ax1.scatter(df["date"], df["k_low"], color='red', marker='o', s=20, zorder=3, label='k_low')
    elif "k_high" in df.columns:
        ax1.scatter(df["date"], df["k_high"]*-1, color='red', marker='o', s=20, zorder=3, label='k_high')

        
    # Adding the legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.05, 0.1), ncol=1, fontsize=9, frameon=False)
   
    
    # Adjust layout to ensure the legend doesn't overlap with the plot
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig1)
    plt.close(fig1)

def plot_signal_chart(df, selection, signal):
    import matplotlib.dates as mdates

    # Ensure date column is in datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Blue scale colormap
    bar_cmap = LinearSegmentedColormap.from_list("lightblue_darkblue", ["lightblue", "darkblue"])

    # Define color mappings
    ohlc_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    volume_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]

    # Create figure and axis
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Normalize Signal values
    norm = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())

    # Plot bar chart
    bars = ax1.bar(df["date"], df["R_t"], color=bar_cmap(norm(df["Signal"])))

    # Format x-axis with dates
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=90, labelsize=9)

    ax1.grid(True)
    ax1.set_xlabel("")
    ax1.set_ylabel("R_t (Trade return after k periods)", fontsize=9, fontweight='normal', family='sans-serif')
    ax1.set_title(f"Master model based on {selection} simulation / {signal}", fontsize=9, fontweight='normal', family='sans-serif')

    # Color bar for Signal
    sm = plt.cm.ScalarMappable(cmap=bar_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Signal')

    # Annotate bars with Signal values
    for bar, sig in zip(bars, df["Signal"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{sig:.2f}',
                 ha='center', va='center', fontsize=9, rotation=90)

    # Plot scatter dots for k_low or k_high
    if "k_low" in df.columns:
        ax1.scatter(df["date"], df["k_low"], color='red', marker='o', s=20, zorder=3, label='k_low')
    elif "k_high" in df.columns:
        ax1.scatter(df["date"], df["k_high"] * -1, color='red', marker='o', s=20, zorder=3, label='k_high')

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.05, 0.1), ncol=1, fontsize=9, frameon=False)

    # Layout and display
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

def extract_second_part(selection: str) -> str:
    """Extracts the second part of a string separated by underscores."""
    parts = selection.split("_")
    return parts[1] if len(parts) > 1 else selection  # Default to full string if no underscores

def plot_ohlc_volume_last_used(df, selection, signal, signal_threshold=0):
    """Plots OHLC and volume charts with round dots below lows, colored based on Signal values.
    
    Only rows with Signal > signal_threshold are used to display the dots.
    The first set of dots (1% below low) is colored based on Signal,
    while the second set (1% further below) is colored based on R_t using a red-to-green scale.
    Everything else is displayed as before.
    """

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Define OHLC colors based on price movement
    ohlc_colors = ['green' if close >= open_ else 'red' 
                   for close, open_ in zip(df['close'], df['open'])]

    # Volume colors based on price movement
    volume_colors = ['green' if close >= open_ else 'red' 
                     for close, open_ in zip(df['close'], df['open'])]

    # Normalize Signal and R_t values for color mapping
    norm_signal = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())
    norm_r_t = Normalize(vmin=df["R_t"].min(), vmax=df["R_t"].max())

    # Create custom blue-scale colormap for Signal
    bar_cmap_signal = LinearSegmentedColormap.from_list("lightblue_darkblue", ["lightblue", "darkblue"])

    # Create figure and subplots (adjusting figure size for better visibility)
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

    # Candlestick OHLC Plot (top)
    body_width = pd.Timedelta(hours=72)  # Adjust body width
    for i in range(len(df)):
        color = ohlc_colors[i]
        ax2.plot([df['date'][i], df['date'][i]], [df['low'][i], df['high'][i]], color=color, lw=2)
        ax2.add_patch(plt.Rectangle(
            (df['date'][i] - body_width / 2, min(df['open'][i], df['close'][i])),
            body_width, abs(df['close'][i] - df['open'][i]),
            color=color, lw=2))

    # Filter rows where Signal > threshold
    signal_filtered = df["Signal"] >= signal_threshold

    # Add first set of dots (Signal-based) 1% below the low
    ax2.scatter(df["date"][signal_filtered],
                df["low"][signal_filtered] * 0.99,
                c=bar_cmap_signal(norm_signal(df["Signal"][signal_filtered])),
                s=50, edgecolors='black', alpha=0.8)

    # Add second set of dots (R_t-based) 1% below the first dots,
    # using the RdYlGn colormap (red-to-green)
    ax2.scatter(df["date"][signal_filtered],
                df["low"][signal_filtered] * 0.99 * 0.99,
                c=plt.cm.RdYlGn(norm_r_t(df["R_t"][signal_filtered])),
                s=30, edgecolors='black', alpha=0.8)

    # Formatting for OHLC plot
    ax2.set_ylabel("Price", fontsize=9, fontweight='normal', family='sans-serif')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=90, labelsize=9)
    ax2.grid(True)

    # Volume Plot (bottom)
    ax3.bar(df['date'], df['volume'], color=volume_colors, alpha=0.7)
    for bar in ax3.patches:
        bar.set_width(0.4)

    ax3.set_xlabel("", fontsize=9, fontweight='normal', family='sans-serif')  # Removed 'Date' label
    ax3.set_ylabel("Volume", fontsize=9, fontweight='normal', family='sans-serif')
    ax3.tick_params(axis='both', labelsize=9)
    ax3.tick_params(axis='x', rotation=90)
    ax3.grid(True)

    # Uncomment the following lines to add extra padding on the x-axis if needed:
    # x_min = df['date'].min() - pd.Timedelta(days=20)  # Add more padding before the first date
    # x_max = df['date'].max() + pd.Timedelta(days=40)  # Add more padding after the last date (increase for more space)
    # ax2.set_xlim(x_min, x_max)
    # ax3.set_xlim(x_min, x_max)

    # Extract second part of the selection string for the title
    title_part = selection.split("_")[-1].upper()
    fig.suptitle(f"{title_part}: OHLC & Volume Chart / {signal} >= {signal_threshold}", fontsize=9, fontweight='normal', family='sans-serif')

    # Add color bar for Signal (on the right of the first plot)
    sm_signal = ScalarMappable(cmap=bar_cmap_signal, norm=norm_signal)
    sm_signal.set_array([])
    plt.colorbar(sm_signal, ax=ax2, orientation='vertical', label="Signal")

    # Add color bar for R_t (on the right of the second plot) using RdYlGn colormap
    sm_r_t = ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm_r_t)
    sm_r_t.set_array([])
    plt.colorbar(sm_r_t, ax=ax3, orientation='vertical', label="R_t")

    # Display in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def plot_ohlc_volume(df, selection, signal, signal_threshold=0):
    """Plots OHLC and volume charts with round dots below lows, colored based on Signal and R_t values.

    - Signal-based dots are 1% below the low.
    - R_t-based dots are 1% further below, using a custom red-white-green colormap.
    """

    df['date'] = pd.to_datetime(df['date'])

    # Define OHLC colors based on price movement
    ohlc_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    volume_colors = ohlc_colors

    # Normalize Signal values for color mapping
    norm_signal = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())

    # Get min/max for R_t and ensure correct ordering
    r_t_min = df["R_t"].min()
    r_t_max = df["R_t"].max()

    # Debug print
    print(f"R_t min: {r_t_min}, R_t max: {r_t_max}")

    # Custom colormap ensuring 0 is PURE WHITE
    colors = [
        (0.0, "darkred"),   # Lowest values - dark red
        (0.25, "red"),      # Mid-low values - red
        (0.5, "white"),     # Zero exactly - white
        (0.75, "green"),    # Mid-high values - green
        (1.0, "darkgreen")  # Highest values - dark green
    ]
    custom_r_t_cmap = LinearSegmentedColormap.from_list("custom_r_t", [c[1] for c in colors], N=256)

    # Normalize R_t values properly so 0 is mapped to white
    norm_r_t = TwoSlopeNorm(
        vmin=r_t_min,  
        vcenter=0,  
        vmax=r_t_max
    )

    # Create Signal colormap
    bar_cmap_signal = LinearSegmentedColormap.from_list("lightblue_darkblue", ["lightblue", "darkblue"])

    # Create figure and subplots
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

    # Candlestick OHLC Plot
    body_width = pd.Timedelta(hours=72)
    for i in range(len(df)):
        color = ohlc_colors[i]
        ax2.plot([df['date'][i], df['date'][i]], [df['low'][i], df['high'][i]], color=color, lw=2)
        ax2.add_patch(plt.Rectangle(
            (df['date'][i] - body_width / 2, min(df['open'][i], df['close'][i])),
            body_width, abs(df['close'][i] - df['open'][i]),
            color=color, lw=2))

    # Filter rows where Signal > threshold
    signal_filtered = df["Signal"] >= signal_threshold

    # Add first set of dots (Signal-based)
    ax2.scatter(df["date"][signal_filtered],
                df["low"][signal_filtered] * 0.99,
                c=bar_cmap_signal(norm_signal(df["Signal"][signal_filtered])),
                s=50, edgecolors='black', alpha=0.8)

    # Add second set of dots (R_t-based)
    ax2.scatter(df["date"][signal_filtered],
                df["low"][signal_filtered] * 0.99 * 0.99,
                c=custom_r_t_cmap(norm_r_t(df["R_t"][signal_filtered])),
                s=30, edgecolors='black', alpha=0.8)

    # Formatting for OHLC plot
    ax2.set_ylabel("Price", fontsize=9, fontweight='normal', family='sans-serif')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=90, labelsize=9)
    ax2.grid(True)

    # Volume Plot
    ax3.bar(df['date'], df['volume'], color=volume_colors, alpha=0.7)
    for bar in ax3.patches:
        bar.set_width(0.4)

    ax3.set_ylabel("Volume", fontsize=9, fontweight='normal', family='sans-serif')
    ax3.tick_params(axis='both', labelsize=9)
    ax3.tick_params(axis='x', rotation=90)
    ax3.grid(True)

    # Extract title part
    title_part = selection.split("_")[-1].upper()
    fig.suptitle(f"{title_part}: OHLC & Volume Chart / {signal} >= {signal_threshold}", fontsize=9, fontweight='normal', family='sans-serif')

    # Add color bar for Signal
    sm_signal = ScalarMappable(cmap=bar_cmap_signal, norm=norm_signal)
    sm_signal.set_array([])
    plt.colorbar(sm_signal, ax=ax2, orientation='vertical', label="Signal")

    # Add color bar for R_t (custom red-white-green)
    sm_r_t = ScalarMappable(cmap=custom_r_t_cmap, norm=norm_r_t)
    sm_r_t.set_array([])
    plt.colorbar(sm_r_t, ax=ax3, orientation='vertical', label="R_t")

    # Display in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def calculate_p_value_rolling(df, signal_threshold, window):
    """
    Calculates the hypergeometric p-value on a moving window.

    For each window of size `window` (if fewer than `window` rows exist, p-value is NaN), the following are computed:
      - N: total number of rows in the window.
      - K: count of rows where R_t > 0.
      - n_draws: count of rows where Signal >= signal_threshold.
      - k: count of rows where both (R_t > 0) and (Signal >= signal_threshold).

    The p-value is computed as:
          p = hypergeom.sf(k-1, N, K, n_draws)
    and stored in a new column "p".

    Parameters:
      df: a pandas DataFrame containing columns "R_t" and "Signal".
      signal_threshold: the threshold value for Signal.
      window: the size of the moving window.

    Returns:
      A new DataFrame with an added column "p" containing the p-values.
    """
    p_vals = []
    N_vals = []
    K_vals = []
    n_draws_vals = []
    k_hg_vals = []

    for i in range(len(df)):
        window_df = df.iloc[max(0, i - window + 1): i + 1]

        if len(window_df) < window:
            p_vals.append(np.nan)
            N_vals.append(np.nan)
            K_vals.append(np.nan)
            n_draws_vals.append(np.nan)
            k_hg_vals.append(np.nan)
        else:
            N = len(window_df)
            K = (window_df["R_t"] > 0).sum()
            n_draws = (window_df["Signal"] >= signal_threshold).sum()
            k = ((window_df["R_t"] > 0) & (window_df["Signal"] >= signal_threshold)).sum()

            if n_draws == 0:
                p_val = np.nan
            else:
                p_val = hypergeom.sf(k-1, N, K, n_draws)

            p_vals.append(p_val)
            N_vals.append(N)
            K_vals.append(K)
            n_draws_vals.append(n_draws)
            k_hg_vals.append(k)

    df = df.copy()
    df["p"] = p_vals
    df["N"] = N_vals
    df["K"] = K_vals
    df["n_draws"] = n_draws_vals
    df["k"] = k_hg_vals  # Renamed for consistency

    return df

def plot_p_value(df, selection, signal_threshold, window):
    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'], df['p'], marker='o', linestyle='-', color='b')
    plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label="Threshold (0.05)")

    # Set title and labels
    ax.set_title(f"Randomness over time \nSimulation {selection} signal_threshold : {signal_threshold} / window {window}", fontsize=9, fontweight='normal', family='sans-serif')
    #ax.set_xlabel('Datetime', fontsize=9)
    ax.set_ylabel('p-value', fontsize=9)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def plot_win_rate_old(df, selection, signal_threshold, window):    
    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'],  df['k'] / df['n_draws'], marker='o', linestyle='-', color='b', label='Sim_trials')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label="Threshold (0.5)")

    # Set title and labels
    ax.set_title(f"Winrate over time \nSimulation {selection} signal_threshold : {signal_threshold} / window {window}", fontsize=9, fontweight='normal', family='sans-serif')
    #ax.set_xlabel('Datetime', fontsize=9)
    ax.set_ylabel('win rate', fontsize=9)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def plot_win_rate(df, df_benchmark, selection, signal_threshold, window):    
    # Dynamically trim trailing NaNs based on available column
    key_col = None
    if 'k_low' in df_benchmark.columns:
        key_col = 'k_low'
    elif 'k_high' in df_benchmark.columns:
        key_col = 'k_high'

    if key_col:
        last_valid_idx = df_benchmark[key_col].last_valid_index()
        if last_valid_idx is not None:
            df_benchmark = df_benchmark.loc[:last_valid_idx]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot the main (filtered) win rate
    ax.plot(
        df['date'],
        df['k'] / df['n_draws'],
        marker='o',
        linestyle='-',
        color='blue',
        label='Win Rate'
    )

    # Plot the benchmark win rate
    ax.plot(
        df_benchmark['date'],
        df_benchmark['K'] / df_benchmark['N'],
        marker='x',
        linestyle='--',
        color='gray',
        label='Benchmark Win Rate'
    )

    # Add threshold line
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label="Random Threshold (0.5)")

    # Title and labels
    ax.set_title(
        f"Winrate over time\nSimulation {selection} | signal_threshold: {round(signal_threshold,3)} | window: {window}",
        fontsize=9, fontweight='normal', family='sans-serif'
    )
    ax.set_ylabel('Win Rate', fontsize=9)

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Add legend
    ax.legend(loc='best', fontsize=9)

    # Show plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def compute_benchmark_returns_onlycall(df_signal, selected_leverage, selected_cost, k_opt):
    """
    Computes R_t_bench_temp and R_t_bench_comp using future prices.

    Parameters:
    df_signal : DataFrame with columns ["date", "Target", "R_t", "Signal", "open", "low", "high", "close"]
    selected_leverage : float
    selected_cost : float
    k_opt : int, horizon for closing price

    Returns:
    DataFrame with new columns: R_t_bench_temp and R_t_bench_comp
    """
    df = df_signal.copy()
    n = len(df)

    R_t_bench_temp = []

    for t in range(n):
        t_open_idx = t + 1
        t_close_idx = t + k_opt

        if t_open_idx < n and t_close_idx < n:
            open_t1 = df.loc[t_open_idx, 'open']
            close_tk = df.loc[t_close_idx, 'close']

            rt_bench = ((((close_tk - open_t1) / open_t1) / k_opt) * selected_leverage) - selected_cost
            R_t_bench_temp.append(rt_bench)

            print(f"t={t}: open_(t+1) = {open_t1:.4f}, close_(t+{k_opt}) = {close_tk:.4f}, R_t_bench_temp = {rt_bench:.4f}")
        else:
            R_t_bench_temp.append(np.nan)
            print(f"t={t}: Not enough future data for open_(t+1) and/or close_(t+{k_opt})")

    # Add temp returns
    df["R_t_bench_temp"] = R_t_bench_temp

    # Compute compounded returns
    R_t_bench_comp = []
    comp_return = 1.0
    for rt in R_t_bench_temp:
        if not np.isnan(rt):
            comp_return *= (1 + rt)
        R_t_bench_comp.append(comp_return)

    df["R_t_bench_comp"] = R_t_bench_comp

    return df

def compute_strategy_returns_onlycall(df_signal, selected_leverage, selected_cost, k_opt,
                              selected_signal, selected_slo, selected_sli):
    """
    Computes strategy compound return based on capped stop loss / take profit logic.

    Parameters:
    - df_signal : pd.DataFrame with market data (must include: open, close, low, high, Signal)
    - selected_leverage : float, leverage multiplier
    - selected_cost : float, transaction cost
    - k_opt : int, holding period (e.g., 3 means hold from t+1 to t+3)
    - selected_signal : float, minimum signal value to enter a trade
    - selected_slo : float, stop-loss return threshold (absolute, positive value)
    - selected_sli : float, take-profit return threshold (absolute, positive value)

    Returns:
    - Full DataFrame with added columns: R_t_temp, R_t_comp
    """
    df = df_signal.copy()
    n = len(df)

    R_t_temp = []
    R_t_comp = []
    comp_return = 1.0

    for t in range(n):
        signal = df.loc[t, "Signal"]
        
        if signal < selected_signal:
            R_t_temp.append(np.nan)
            R_t_comp.append(comp_return)
            continue

        open_t1_idx = t + 1
        close_tk_idx = t + k_opt
        low_range = range(t + 1, t + k_opt + 1)
        high_range = range(t + 1, t + k_opt + 1)

        if open_t1_idx >= n or close_tk_idx >= n or low_range[-1] >= n:
            R_t_temp.append(np.nan)
            R_t_comp.append(comp_return)
            print(f"t={t}: Not enough data for open or close or high/low window")
            continue

        open_t1 = df.loc[open_t1_idx, 'open']
        close_tk = df.loc[close_tk_idx, 'close']
        lows = df.loc[low_range, 'low']
        highs = df.loc[high_range, 'high']

        min_low = lows.min()
        max_high = highs.max()

        R_low = abs((min_low - open_t1) / open_t1)
        R_high = abs((max_high - open_t1) / open_t1)

        if R_low >= selected_slo:
            rt_temp = -1 * selected_slo
            reason = "Stop Loss triggered"
        elif R_high >= selected_sli:
            rt_temp = selected_sli
            reason = "Take Profit triggered"
        else:
            rt_temp = (close_tk - open_t1) / open_t1
            reason = "Normal exit"

        # Final return calculation
        rt = ((rt_temp / k_opt) * selected_leverage) - selected_cost
        comp_return *= (1 + rt)

        R_t_temp.append(rt_temp)
        R_t_comp.append(comp_return)

        print(f"""
t={t} | Signal={signal:.2f}
Reason: {reason}
Used open_(t+1): {open_t1:.4f}
Used close_(t+{k_opt}): {close_tk:.4f}
Min(low_(t+1 to t+{k_opt})): {min_low:.4f}
Max(high_(t+1 to t+{k_opt})): {max_high:.4f}
R_low: {R_low:.4f}, R_high: {R_high:.4f}
R_t_temp (before leverage): {rt_temp:.4f}
Final R_t (after leverage and cost): {rt:.4f}
Cumulative Compounded Return: {comp_return:.4f}
""")

    df["R_t_temp"] = R_t_temp
    df["R_t_comp"] = R_t_comp

    return df

def compute_benchmark_returns(df_signal, selected_leverage, selected_cost, k_opt, selected_model):
    df = df_signal.copy()
    n = len(df)
    
    R_t_bench_temp = []
    R_t_bench_comp = []
    comp_return = 1.0

    for t in range(n):
        open_idx = t + 1
        close_idx = t + k_opt

        if open_idx >= n or close_idx >= n:
            R_t_bench_temp.append(np.nan)
            R_t_bench_comp.append(comp_return)
            print(f"{df.loc[t, 'date']}: t={t} - Not enough data")
            continue

        open_t1 = df.loc[open_idx, "open"]
        close_tk = df.loc[close_idx, "close"]

        direction = -1 if "put" in selected_model.lower() else 1

        rt_temp = ((((close_tk - open_t1) / open_t1) / k_opt) * selected_leverage * direction) - selected_cost
        comp_return *= (1 + rt_temp)

        R_t_bench_temp.append(rt_temp)
        R_t_bench_comp.append(comp_return)

        print(f"{df.loc[t, 'date']} | Model: {selected_model.upper()} | t={t} | open_(t+1)={open_t1:.4f}, "
              f"close_(t+{k_opt})={close_tk:.4f}, R_t_bench_temp={rt_temp:.4f}, Comp={comp_return:.4f}")

    df["R_t_bench_temp"] = R_t_bench_temp
    df["R_t_bench_comp"] = R_t_bench_comp
    return df

def compute_strategy_returns(df_signal, selected_leverage, selected_cost, k_opt, selected_signal, selected_slo, selected_sli, selected_model):
    df = df_signal.copy()
    n = len(df)

    R_t_temp = []
    R_t_comp = []
    comp_return = 1.0

    for t in range(n):
        signal = df.loc[t, "Signal"]
        if signal < selected_signal:
            R_t_temp.append(np.nan)
            R_t_comp.append(comp_return)
            continue

        open_idx = t + 1
        close_idx = t + k_opt
        range_idx = range(t + 1, t + k_opt + 1)

        if open_idx >= n or close_idx >= n or range_idx[-1] >= n:
            R_t_temp.append(np.nan)
            R_t_comp.append(comp_return)
            print(f"{df.loc[t, 'date']}: t={t} - Not enough data for strategy")
            continue

        open_t1 = df.loc[open_idx, 'open']
        close_tk = df.loc[close_idx, 'close']
        lows = df.loc[range_idx, 'low']
        highs = df.loc[range_idx, 'high']

        min_low = lows.min()
        max_high = highs.max()

        R_low = abs((min_low - open_t1) / open_t1)
        R_high = abs((max_high - open_t1) / open_t1)

        model = selected_model.lower()

        if "put" in model:
            # Put logic
            if selected_slo <= R_high:
                rt_temp = -1 * selected_slo
                reason = "Stop Loss triggered (PUT)"
            elif selected_sli <= R_low:
                rt_temp = selected_sli
                reason = "Take Profit triggered (PUT)"
            else:
                rt_temp = ((close_tk - open_t1) / open_t1) * -1
                reason = "Normal exit (PUT)"
        else:
            # Call logic
            if selected_slo <= R_low:
                rt_temp = -1 * selected_slo
                reason = "Stop Loss triggered (CALL)"
            elif selected_sli <= R_high:
                rt_temp = selected_sli
                reason = "Take Profit triggered (CALL)"
            else:
                rt_temp = (close_tk - open_t1) / open_t1
                reason = "Normal exit (CALL)"

        R_t_temp.append(rt_temp)
        rt = ((rt_temp / k_opt) * selected_leverage) - selected_cost
        comp_return *= (1 + rt)
        R_t_comp.append(comp_return)

        print(f"""{df.loc[t, 'date']} | Model: {selected_model.upper()} | t={t}
  Signal={signal:.2f} | {reason}
  open_(t+1): {open_t1:.4f}, close_(t+{k_opt}): {close_tk:.4f}
  min_low: {min_low:.4f}, max_high: {max_high:.4f}
  R_low: {R_low:.4f}, R_high: {R_high:.4f}
  R_t_temp: {rt_temp:.4f}, Final R_t: {rt:.4f}, Comp Return: {comp_return:.4f}""")

    df["R_t_temp"] = R_t_temp
    df["R_t_comp"] = R_t_comp
    return df[df["Signal"] >= selected_signal].reset_index(drop=True)

def plot_strategy_vs_benchmark(df_result, selected_model, k_opt, selected_slo, selected_sli, selected_leverage, selected_cost):
    """
    Plots strategy vs. benchmark compound return over time.

    Parameters:
    - df_result : DataFrame with 'date', 'R_t_comp', 'R_t_bench_comp' columns.
    - selected_model : 'call' or 'put'
    - k_opt : int
    - selected_slo : float
    - selected_sli : float
    - selected_leverage : float
    - selected_cost : float
    """

    df = df_result.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["date"], df["R_t_bench_comp"], label="Benchmark", color="gray", linewidth=2, linestyle="--")
    ax.plot(df["date"], df["R_t_comp"], label="Strategy", color="blue", linewidth=2)

    ax.set_ylabel("Compound Return", fontsize=9, fontweight='normal', family='sans-serif')
    ax.set_xlabel("Date", fontsize=9, fontweight='normal', family='sans-serif')
    ax.tick_params(axis='x', rotation=90, labelsize=9)
    ax.grid(True)
    ax.legend(fontsize=8)

    # Title with parameters
    title = (f"{selected_model.upper()} | k={k_opt} | SLO={round(selected_slo,3)} | SLI={round(selected_sli,3)} | "
             f"Lev={selected_leverage} | Cost={selected_cost}")
    fig.suptitle(f"Strategy vs Benchmark Compound Return\n{title}", fontsize=10, fontweight='normal', family='sans-serif')

    # Streamlit display
    st.pyplot(fig)
    plt.close(fig)

def analyze_weekly_returns_df(df_result, selected_signal=None, return_column='R_t_comp'):
    label = 'strategy' if return_column == 'R_t_comp' else 'bench'
    df = df_result.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Weekly compound return snapshot (last value in week)
    weekly_comp = df[return_column].resample('W').last()

    # Identify active weeks where at least one Signal ≥ selected_signal
    if return_column == 'R_t_comp' and selected_signal is not None:
        signal_active = df['Signal'].resample('W').apply(lambda x: (x >= selected_signal).any())
        weekly_comp = weekly_comp[signal_active]    

    # Calculate weekly returns
    weekly_returns = weekly_comp.pct_change().dropna()
    returns = weekly_returns.values

    if len(returns) == 0:
        return pd.DataFrame({label: ["Not enough data"]}, index=["error"])

    # --- Metrics ---
    n_trades = (returns != 0).sum()
    n_share = (returns != 0).mean()
    n_pos = (returns > 0).sum()
    cum_return = (returns + 1).prod() - 1
    cagr = (1 + cum_return) ** (52 / len(returns)) - 1

    avg_loss = np.mean(returns[returns < 0]) if any(returns < 0) else 0
    avg_win = np.mean(returns[returns > 0]) if any(returns > 0) else 0
    avg_return = np.mean(returns)
    expected_return = avg_return
    expected_shortfall = np.mean(returns[returns <= np.percentile(returns, 5)])

    gain_to_pain_ratio = (
        returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if any(returns < 0) else np.nan
    )
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan
    profit_factor = gain_to_pain_ratio  # same as gain_to_pain_ratio
    profit_ratio = np.sum(returns > 0) / len(returns)

    win_rate = n_pos / n_trades if n_trades > 0 else np.nan
    win_loss_ratio = n_pos / (n_trades - n_pos) if (n_trades - n_pos) > 0 else np.nan

    risk_of_ruin = (
        (1 - win_rate) / (1 + win_loss_ratio) ** 2 if win_loss_ratio and win_rate and win_loss_ratio != 0 else np.nan
    )
    risk_return_ratio = np.std(returns) / avg_return if avg_return != 0 else np.nan
    value_at_risk = np.percentile(returns, 5)
    volatility = np.std(returns)
    implied_volatility = volatility * np.sqrt(52)

    sharpe = avg_return / np.std(returns) * np.sqrt(52) if np.std(returns) != 0 else np.nan
    downside = np.std(returns[returns < 0])
    sortino = avg_return / downside * np.sqrt(52) if downside != 0 else np.nan

    metrics = {
        "n_weeks": len(returns),
        "n_trades": n_trades,
        "n_share": round(n_share, 4),
        "n_pos": n_pos,
        "win_rate": round(win_rate, 4),
        "win_loss_ratio": round(win_loss_ratio, 4),
        "cum_return": round(cum_return, 4),
        "cagr": round(cagr, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "avg_return": round(avg_return, 4),
        "expected_return": round(expected_return, 4),
        "expected_shortfall": round(expected_shortfall, 4),
        "gain_to_pain_ratio": round(gain_to_pain_ratio, 4) if not np.isnan(gain_to_pain_ratio) else np.nan,
        "payoff_ratio": round(payoff_ratio, 4) if not np.isnan(payoff_ratio) else np.nan,
        "profit_factor": round(profit_factor, 4) if not np.isnan(profit_factor) else np.nan,
        "profit_ratio": round(profit_ratio, 4),
        "risk_of_ruin": round(risk_of_ruin, 4) if not np.isnan(risk_of_ruin) else np.nan,
        "risk_return_ratio": round(risk_return_ratio, 4) if not np.isnan(risk_return_ratio) else np.nan,
        "value_at_risk": round(value_at_risk, 4),
        "volatility": round(volatility, 4),
        "implied_volatility": round(implied_volatility, 4),
        "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        "sortino": round(sortino, 4) if not np.isnan(sortino) else np.nan,
    }

    return pd.DataFrame({label: metrics})

def plot_metrics_comparison(df_comparison, small_threshold=1.0, figsize=(10, 6)):
    """
    Plots horizontal bar charts comparing strategy and bench metrics.

    Parameters:
    - df_comparison: DataFrame from analyze_weekly_returns_df (combined)
    - small_threshold: float, used to separate small vs large metrics
    - figsize: tuple, size of each figure
    """

    # Separate metrics into small vs large based on threshold
    abs_max = df_comparison.abs().max(axis=1)
    small_metrics = abs_max[abs_max <= small_threshold].index.tolist()
    large_metrics = abs_max[abs_max > small_threshold].index.tolist()

    def _plot(metrics_subset, title_suffix):
        fig, ax = plt.subplots(figsize=figsize)

        # Subset and reorder for display
        data = df_comparison.loc[metrics_subset]
        data = data.sort_values(by="strategy", ascending=True)

        # Bar positions
        y_pos = range(len(data))

        ax.barh(y_pos, data["strategy"], height=0.4, label='Strategy', color='green', alpha=0.6)
        ax.barh([y + 0.4 for y in y_pos], data["bench"], height=0.4, label='Bench', color='steelblue', alpha=0.6)

        ax.set_yticks([y + 0.2 for y in y_pos])
        ax.set_yticklabels(data.index)
        ax.invert_yaxis()
        ax.set_title(f"Metric Comparison ({title_suffix})", fontsize=10, fontweight='normal')
        ax.set_xlabel("Metric Value", fontsize=9)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        st.pyplot(fig)
        plt.close(fig)

    # Plot small and large separately
    if small_metrics:
        _plot(small_metrics, "Small Metrics")

    if large_metrics:
        _plot(large_metrics, "Large Metrics")
