import streamlit as st
import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from werkzeug.security import check_password_hash

# Load MongoDB credentials from Streamlit secrets
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

# Streamlit Login System
def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        return True
    return False

st.title("Login Page")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

if st.session_state["authenticated"]:
    st.title("Dashboard")

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
    st.write(df)

    # Define a custom colormap (yellow to orange)
    colors = ["#FFFFE0", "#FFBF00", "#FF4500"]  
    cmap = LinearSegmentedColormap.from_list("yellow_orange", colors)

    # First Plot: R_t Bar Chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    df["Signal"] = df.filter(regex="^S\d+").apply(pd.to_numeric, errors="coerce").sum(axis=1)
    norm = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())

    bars = ax1.bar(df["date"], df["R_t"], color=cmap(norm(df["Signal"])), edgecolor='black')
    ax1.set_xticklabels(df["date"], rotation=90)
    ax1.grid(False)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("R_t")
    ax1.set_title(f"{selected_model} Model: R_t Visualization")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Signal')

    for bar, signal in zip(bars, df["Signal"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{signal:.2f}', 
                 ha='center', va='center', fontsize=10, color='black', rotation=90)

    st.pyplot(fig1)
    plt.close(fig1)

    # Second Plot: OHLC Candlestick + Volume
    df['date'] = pd.to_datetime(df['date'])
    fig2 = plt.figure(figsize=(12, 8))
    spec = fig2.add_gridspec(ncols=4, nrows=2, height_ratios=[3, 1])

    ax2 = fig2.add_subplot(spec[0, :])  # OHLC
    ax3 = fig2.add_subplot(spec[1, :])  # Volume

    # Candlestick OHLC Plot
    body_width = pd.Timedelta(hours=72)  

    for i in range(len(df)):
        color = 'green' if df['close'][i] >= df['open'][i] else 'red'
        ax2.plot([df['date'][i], df['date'][i]], [df['low'][i], df['high'][i]], color=color, lw=2)
        ax2.add_patch(plt.Rectangle(
            (df['date'][i] - body_width / 2, min(df['open'][i], df['close'][i])),
            body_width, abs(df['close'][i] - df['open'][i]),
            color=color, lw=2, edgecolor='black'))

    ax2.set_title("OHLC Chart (Candlestick)")
    ax2.set_ylabel("Price")
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=90)

    # Volume Plot
    ax3.bar(df['date'], df['volume'], color='green', alpha=0.7)
    for bar in ax3.patches:
        bar.set_width(0.25)  # Reduce bar width

    ax3.set_title("Volume Chart")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Volume")

    st.pyplot(fig2)
    plt.close(fig2)
