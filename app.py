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
users_collection = db["users"]

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
    st.title("Model Selection")
    
    # Retrieve all models containing '_master' in 'Sim'
    models = list(db.Models.find({"Sim": {"$regex": "_master"}}, {"Sim": 1}))
    model_options = {model["Sim"]: str(model["_id"]) for model in models}
    
    if model_options:
        selected_model = st.selectbox("Select a model:", list(model_options.keys()))
        
        if selected_model:
            model_id = model_options[selected_model]
            
            # Fetch data based on selected model
            data = db.Models.find_one({"_id": ObjectId(model_id)}, {"Now": 1, "_id": 0})
            
            if data and "Now" in data:
                df = pd.DataFrame(data["Now"])
                st.write(df)
            else:
                st.error("No data found for the selected model.")
            
            # Plot OHLC & Volume Chart (Existing Logic)
            df['date'] = pd.to_datetime(df['date'])
            
            fig = plt.figure(figsize=(12, 8))
            spec = fig.add_gridspec(ncols=4, nrows=2, height_ratios=[3, 1])
            ax1 = fig.add_subplot(spec[0, :])
            ax2 = fig.add_subplot(spec[1, :])
            
            body_width = pd.Timedelta(hours=72)
            
            for i in range(len(df)):
                color = 'green' if df['close'][i] >= df['open'][i] else 'red'
                ax1.plot([df['date'][i], df['date'][i]], [df['low'][i], df['high'][i]], color=color, lw=2)
                body_bottom = min(df['open'][i], df['close'][i])
                body_top = max(df['open'][i], df['close'][i])
                ax1.add_patch(plt.Rectangle((df['date'][i] - body_width / 2, body_bottom),
                                            body_width, body_top - body_bottom,
                                            color=color, lw=2, edgecolor='black'))
            
            ax1.set_title("OHLC Chart (Candlestick)")
            ax1.set_ylabel("Price")
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())
            ax1.tick_params(axis='x', rotation=90)
            
            ax2.bar(df['date'], df['volume'], color='green', alpha=0.7)
            for bar in ax2.patches:
                bar.set_width(0.25)
            
            ax2.set_title("Volume Chart")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Volume")
            
            fig.tight_layout(h_pad=3.0)
            st.pyplot(fig)
            plt.close(fig)
