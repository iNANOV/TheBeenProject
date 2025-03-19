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

# Set the theme
mt.set_theme("yellowish")

def plot_signal_chart(df, selection):
    # Blue scale: Light blue for low values, dark blue for high values
    bar_cmap = LinearSegmentedColormap.from_list("lightblue_darkblue", ["lightblue", "darkblue"])
    
    # Define color mappings for OHLC and Volume plots
    ohlc_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    volume_colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['close'], df['open'])]

    # Bar Chart: Adjusting color and aligning dates
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    df["Signal"] = df.filter(regex="^S\d+").apply(pd.to_numeric, errors="coerce").sum(axis=1)
    norm = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())
    bars = ax1.bar(df["date"], df["R_t"], color=bar_cmap(norm(df["Signal"])))

    # Set tick labels alignment and rotation
    ax1.set_xticklabels(df["date"], rotation=90, ha='center')  # Align labels at the center
    ax1.tick_params(labelsize=9)  # Set font size for ticks to 9
    ax1.grid(True)
    ax1.set_xlabel("")  # Removed 'Date' label
    ax1.set_ylabel("R_t", fontsize=9, fontweight='normal', family='Arial')  # Set label font size to 9
    ax1.set_title(f"{selection} Model: R_t Visualization", fontsize=9, fontweight='normal', family='Arial')

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

def plot_ohlc_volume(df, selection):
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
    title_part = extract_second_part(selection)
    fig.suptitle(f"{title_part}: OHLC & Volume Chart", fontsize=9, fontweight='normal', family='Arial')
    
    st.pyplot(fig)
    plt.close(fig)
