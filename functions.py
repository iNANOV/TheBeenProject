import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import morethemes as mt
import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import check_password_hash
from matplotlib.cm import ScalarMappable 
from scipy.stats import hypergeom

# Set the theme
# mt.set_theme("yellowish")

def calculate_weighted_signal(df):
    # Select S1 to S10 columns
    S_columns = df.filter(regex="^S\d+")
    
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
    
    required_cols = required_S_cols | required_N_cols | required_R_cols
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select S1 to S10 columns
    S_columns = df.filter(regex="^S\d+")
    
    # Number of S columns (should be 10 for S1 to S10)
    num_S = S_columns.shape[1]
    
    # Create exponential weights (S1 has highest weight, S10 lowest)
    weights = np.array([0.9 ** (i-1) for i in range(1, num_S+1)])
    
    # Normalize weights so they sum to 1 (optional, to keep scale consistent)
    weights /= weights.sum()
    
    # Compute weighted sum for all S1 to S10
    signal_balanced = sum(
        df[f"S{i}"] * (df[f"N{i}"] / 52) * df[f"R{i}"] * weights[i-1]
        for i in range(1, num_S + 1)
    )
    
    # Store result in the dataframe
    df["Signal_balanced"] = signal_balanced
    
    return df


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

    # Create a new DataFrame with only the selected signal, renamed to "Signal"
    df_signal = df[[signal_type]].rename(columns={signal_type: "Signal"})

    return df_signal, signal_type

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
    ax.set_title(f"Simulation {selection} progressing till adding new best models", fontsize=9, fontweight='normal', family='Arial')
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
    ax.set_title(f"Adding new best models from {selection} simulation", fontsize=9, fontweight='normal', family='Arial')
    #ax.set_xlabel('Datetime', fontsize=9)
    ax.set_ylabel('Cumsum', fontsize=9)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def plot_function_value_distribution_by_date(df, selection):
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
    ax.set_title(f"Method Value Distribution over Time for {selection}", fontsize=9, fontweight='normal', family='Arial')
    
    # Remove x-axis label
    ax.set_xlabel('')  # This removes the label

    ax.set_ylabel('Counts of Method Occurrences', fontsize=9)

    # Rotate the x-axis labels for better readability
    ax.set_xticklabels(function_value_counts.index.strftime('%Y-%m-%d'), rotation=90, fontsize=9)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)


def plot_signal_chart(df, selection, signal):
    
    # Blue scale: Light blue for low values, dark blue for high values
    bar_cmap = LinearSegmentedColormap.from_list("lightblue_darkblue", ["lightblue", "darkblue"])
    
    # Define color mappings for OHLC and Volume plots
    ohlc_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    volume_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]

    # Bar Chart: Adjusting color and aligning dates
    fig1, ax1 = plt.subplots(figsize=(12, 6))
   # df["Signal"] = df.filter(regex="^S\d+").apply(pd.to_numeric, errors="coerce").sum(axis=1)
    norm = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())
    bars = ax1.bar(df["date"], df["R_t"], color=bar_cmap(norm(df["Signal"])))

    # Set tick labels alignment and rotation
    ax1.set_xticklabels(df["date"], rotation=90, ha='center')  # Align labels at the center
    ax1.tick_params(labelsize=9)  # Set font size for ticks to 9
    ax1.grid(True)
    ax1.set_xlabel("")  # Removed 'Date' label
    ax1.set_ylabel("R_t", fontsize=9, fontweight='normal', family='Arial')  # Set label font size to 9
    ax1.set_title(f"Master model based on {selection} simulation / {signal}", fontsize=9, fontweight='normal', family='Arial')

    # Color bar (legend)
    sm = plt.cm.ScalarMappable(cmap=bar_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Signal')

    # Adding values on top of bars
    for bar, signal in zip(bars, df["Signal"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{signal:.2f}',
                 ha='center', va='center', fontsize=9, rotation=90)

    # Adding the legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=9, frameon=False)

    # Adjust layout to ensure the legend doesn't overlap with the plot
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig1)
    plt.close(fig1)



def extract_second_part(selection: str) -> str:
    """Extracts the second part of a string separated by underscores."""
    parts = selection.split("_")
    return parts[1] if len(parts) > 1 else selection  # Default to full string if no underscores

def plot_ohlc_volume_old(df, selection):
    """Plots OHLC and volume charts using green and red for positive/negative trends."""
    ohlc_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    volume_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    
    df['date'] = pd.to_datetime(df['date'])
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
    
    # Candlestick OHLC Plot
    body_width = pd.Timedelta(hours=72)
    for i in range(len(df)):
        color = 'green' if df['close'][i] >= df['open'][i] else 'red'
        ax2.plot([df['date'][i], df['date'][i]], [df['low'][i], df['high'][i]], color=color, lw=2)
        ax2.add_patch(plt.Rectangle(
            (df['date'][i] - body_width / 2, min(df['open'][i], df['close'][i])),
            body_width, abs(df['close'][i] - df['open'][i]),
            color=color, lw=2))
    
    ax2.set_ylabel("Price", fontsize=9, fontweight='normal', family='Arial')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=90, labelsize=9)
    ax2.grid(True)
    
    # Volume Plot
    ax3.bar(df['date'], df['volume'], color=volume_colors, alpha=0.7)
    for bar in ax3.patches:
        bar.set_width(0.25)
    
    ax3.set_xlabel("", fontsize=9, fontweight='normal', family='Arial')  # Removed 'Date' label
    ax3.set_ylabel("Volume", fontsize=9, fontweight='normal', family='Arial')
    ax3.tick_params(axis='both', labelsize=9)
    ax3.tick_params(axis='x', rotation=90)  # Rotate x-tick labels vertically
    ax3.grid(True)
    
    # Extract second part of the selection string for the title
    title_part = extract_second_part(selection).upper()
    fig.suptitle(f"{title_part}: OHLC & Volume Chart", fontsize=9, fontweight='normal', family='Arial')
    
    st.pyplot(fig)
    plt.close(fig)


def plot_ohlc_volume_best(df, selection, signal):
    """Plots OHLC and volume charts with round dots below lows, colored based on Signal values."""

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Define OHLC colors based on price movement
    ohlc_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]

    # Volume colors based on price movement
    volume_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]

    # Normalize Signal values for color mapping
    norm_signal = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())
    norm_r_t = Normalize(vmin=df["R_t"].min(), vmax=df["R_t"].max())

    # Create custom blue-scale colormap for Signal
    bar_cmap_signal = LinearSegmentedColormap.from_list("lightblue_darkblue", ["lightblue", "darkblue"])

    # Create figure and subplots (adjusting figure size for better visibility)
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

    # Candlestick OHLC Plot
    body_width = pd.Timedelta(hours=72)  # Adjust body width
    for i in range(len(df)):
        color = ohlc_colors[i]
        ax2.plot([df['date'][i], df['date'][i]], [df['low'][i], df['high'][i]], color=color, lw=2)
        ax2.add_patch(plt.Rectangle(
            (df['date'][i] - body_width / 2, min(df['open'][i], df['close'][i])),
            body_width, abs(df['close'][i] - df['open'][i]),
            color=color, lw=2))

    # Add round dots 1% below the Low, colored by Signal values
    ax2.scatter(df["date"], df["low"] * 0.99, c=bar_cmap_signal(norm_signal(df["Signal"])), s=50, edgecolors='black', alpha=0.8)

    # Add second dots 1% below the first dots, filled with R_t values
    ax2.scatter(df["date"], df["low"] * 0.99 * 0.99, c=plt.cm.viridis(norm_r_t(df["R_t"])), s=30, edgecolors='black', alpha=0.8)

    # Formatting for OHLC plot
    ax2.set_ylabel("Price", fontsize=9, fontweight='normal', family='Arial')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=90, labelsize=9)
    ax2.grid(True)

    # Volume Plot
    ax3.bar(df['date'], df['volume'], color=volume_colors, alpha=0.7)
    for bar in ax3.patches:
        bar.set_width(0.4)

    ax3.set_xlabel("", fontsize=9, fontweight='normal', family='Arial')  # Removed 'Date' label
    ax3.set_ylabel("Volume", fontsize=9, fontweight='normal', family='Arial')
    ax3.tick_params(axis='both', labelsize=9)
    ax3.tick_params(axis='x', rotation=90)  # Rotate x-tick labels vertically
    ax3.grid(True)

    # Manually adjust the x-axis limits for both plots to add more empty space
   # x_min = df['date'].min() - pd.Timedelta(days=20)  # Add more padding before the first date
    #x_max = df['date'].max() + pd.Timedelta(days=40)  # Add more padding after the last date (increase for more space)

    # Set the same x-axis limits for both plots with more padding
   # ax2.set_xlim(x_min, x_max)
    #ax3.set_xlim(x_min, x_max)

    # Extract second part of the selection string for the title
    title_part = selection.split("_")[-1].upper()
    fig.suptitle(f"{title_part}: OHLC & Volume Chart / {signal}", fontsize=9, fontweight='normal', family='Arial')

    # Add color bar for Signal (on the right of the first plot)
    sm_signal = ScalarMappable(cmap=bar_cmap_signal, norm=norm_signal)
    sm_signal.set_array([])
    plt.colorbar(sm_signal, ax=ax2, orientation='vertical', label="Signal")

    # Add color bar for R_t (on the right of the second plot)
    sm_r_t = ScalarMappable(cmap=plt.cm.viridis, norm=norm_r_t)
    sm_r_t.set_array([])
    plt.colorbar(sm_r_t, ax=ax3, orientation='vertical', label="R_t")

    # Display in Streamlit
    st.pyplot(fig)
    plt.close(fig)


def plot_ohlc_volume_working(df, selection, signal, signal_threshold=0):
    """Plots OHLC and volume charts with round dots below lows, colored based on Signal values.
    
    Dots (both sets) are displayed only for rows where Signal > signal_threshold.
    The first set of dots is colored based on Signal, and the second set (1% further down)
    is filled with R_t values. Everything else is displayed as it is.
    """

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Define OHLC colors based on price movement
    ohlc_colors = ['green' if close >= open_ else 'red' 
                   for close, open_ in zip(df['close'], df['open'])]

    # Volume colors based on price movement
    volume_colors = ['green' if close >= open_ else 'red' 
                     for close, open_ in zip(df['close'], df['open'])]

    # Normalize Signal values for color mapping
    norm_signal = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())
    norm_r_t = Normalize(vmin=df["R_t"].min(), vmax=df["R_t"].max())

    # Create custom blue-scale colormap for Signal
    bar_cmap_signal = LinearSegmentedColormap.from_list("lightblue_darkblue", ["lightblue", "darkblue"])

    # Create figure and subplots (adjusting figure size for better visibility)
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

    # Candlestick OHLC Plot
    body_width = pd.Timedelta(hours=72)  # Adjust body width
    for i in range(len(df)):
        color = ohlc_colors[i]
        ax2.plot([df['date'][i], df['date'][i]], [df['low'][i], df['high'][i]], color=color, lw=2)
        ax2.add_patch(plt.Rectangle(
            (df['date'][i] - body_width / 2, min(df['open'][i], df['close'][i])),
            body_width, abs(df['close'][i] - df['open'][i]),
            color=color, lw=2))

    # Filter rows where Signal is above the threshold
    signal_filtered = df["Signal"] > signal_threshold

    # Add round dots 1% below the Low, colored by Signal values
    ax2.scatter(df["date"][signal_filtered],
                df["low"][signal_filtered] * 0.99,
                c=bar_cmap_signal(norm_signal(df["Signal"][signal_filtered])),
                s=50, edgecolors='black', alpha=0.8)

    # Add second dots 1% below the first dots, filled with R_t values
    ax2.scatter(df["date"][signal_filtered],
                df["low"][signal_filtered] * 0.99 * 0.99,
                c=plt.cm.viridis(norm_r_t(df["R_t"][signal_filtered])),
                s=30, edgecolors='black', alpha=0.8)

    # Formatting for OHLC plot
    ax2.set_ylabel("Price", fontsize=9, fontweight='normal', family='Arial')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=90, labelsize=9)
    ax2.grid(True)

    # Volume Plot
    ax3.bar(df['date'], df['volume'], color=volume_colors, alpha=0.7)
    for bar in ax3.patches:
        bar.set_width(0.4)

    ax3.set_xlabel("", fontsize=9, fontweight='normal', family='Arial')  # Removed 'Date' label
    ax3.set_ylabel("Volume", fontsize=9, fontweight='normal', family='Arial')
    ax3.tick_params(axis='both', labelsize=9)
    ax3.tick_params(axis='x', rotation=90)  # Rotate x-tick labels vertically
    ax3.grid(True)

    # Manually adjust the x-axis limits for both plots to add more empty space
    # x_min = df['date'].min() - pd.Timedelta(days=20)  # Add more padding before the first date
    # x_max = df['date'].max() + pd.Timedelta(days=40)  # Add more padding after the last date (increase for more space)

    # Set the same x-axis limits for both plots with more padding
    # ax2.set_xlim(x_min, x_max)
    # ax3.set_xlim(x_min, x_max)

    # Extract second part of the selection string for the title
    title_part = selection.split("_")[-1].upper()
    fig.suptitle(f"{title_part}: OHLC & Volume Chart / {signal}", fontsize=9, fontweight='normal', family='Arial')

    # Add color bar for Signal (on the right of the first plot)
    sm_signal = ScalarMappable(cmap=bar_cmap_signal, norm=norm_signal)
    sm_signal.set_array([])
    plt.colorbar(sm_signal, ax=ax2, orientation='vertical', label="Signal")

    # Add color bar for R_t (on the right of the second plot)
    sm_r_t = ScalarMappable(cmap=plt.cm.viridis, norm=norm_r_t)
    sm_r_t.set_array([])
    plt.colorbar(sm_r_t, ax=ax3, orientation='vertical', label="R_t")

    # Display in Streamlit
    st.pyplot(fig)
    plt.close(fig)


def plot_ohlc_volume(df, selection, signal, signal_threshold=0):
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
    ax2.set_ylabel("Price", fontsize=9, fontweight='normal', family='Arial')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=90, labelsize=9)
    ax2.grid(True)

    # Volume Plot (bottom)
    ax3.bar(df['date'], df['volume'], color=volume_colors, alpha=0.7)
    for bar in ax3.patches:
        bar.set_width(0.4)

    ax3.set_xlabel("", fontsize=9, fontweight='normal', family='Arial')  # Removed 'Date' label
    ax3.set_ylabel("Volume", fontsize=9, fontweight='normal', family='Arial')
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
    fig.suptitle(f"{title_part}: OHLC & Volume Chart / {signal} >= {signal_threshold}", fontsize=9, fontweight='normal', family='Arial')

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
    ax.set_title(f"Simulation {selection} signal_threshold : {signal_threshold} / window {window}", fontsize=9, fontweight='normal', family='Arial')
    #ax.set_xlabel('Datetime', fontsize=9)
    ax.set_ylabel('p-value', fontsize=9)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

def plot_win_rate(df, selection, signal_threshold, window):    
    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'],  df['k'] / df['n_draws'], marker='o', linestyle='-', color='b', label='Sim_trials')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label="Threshold (0.5)")

    # Set title and labels
    ax.set_title(f"Simulation {selection} signal_threshold : {signal_threshold} / window {window}", fontsize=9, fontweight='normal', family='Arial')
    #ax.set_xlabel('Datetime', fontsize=9)
    ax.set_ylabel('win rate', fontsize=9)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)