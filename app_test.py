import streamlit as st
import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId
import matplotlib.pyplot as plt
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
users_collection = db["users"]  # Access users collection

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
    st.title("Test")
    st.write("Welcome to the test page!")

    # Retrieve and display data
    st.title("MongoDB Data Viewer")

    try:
        data = db.Models.find_one({"_id": ObjectId("67b73c19fc9a30b7cb7f1ae4")}, {"Now": 1, "_id": 0})
        if data and "Now" in data:
            df = pd.DataFrame(data["Now"])
            st.write(df)
        else:
            st.error("No data found for the given ID.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")


    # Define a custom colormap that transitions from light yellow to dark orange
    colors = ["#FFFFE0", "#FFBF00", "#FF4500"]  # Light Yellow to Dark Orange
    cmap = LinearSegmentedColormap.from_list("yellow_orange", colors)

    # Create a figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 6))

    df["Signal"] = df.filter(regex="^S\d+").apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # Normalize Signal values for color scaling
    norm = Normalize(vmin=df["Signal"].min(), vmax=df["Signal"].max())

    # Plot bars with color based on Signal using the yellow-to-orange colormap
    bars = ax.bar(df["date"], df["R_t"], 
                  color=cmap(norm(df["Signal"])), 
                  edgecolor='black')  # Add black border around each bar

    # Rotate x-axis labels (dates) for vertical display
    ax.set_xticklabels(df["date"], rotation=90)

    # Remove gridlines from background
    ax.grid(False)

    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("R_t")
    ax.set_title("GSPC call MASTER model, trained until Dec 2023, all")

    # Add color bar to show the scale of the Signal
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set empty array to make ScalarMappable valid
    plt.colorbar(sm, ax=ax, label='Signal')

    # Add Signal values vertically inside the bars
    for bar, signal in zip(bars, df["Signal"]):
        height = bar.get_height()
        # Place the text vertically inside the bars, centered
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{signal:.2f}', 
                ha='center', va='center', fontsize=10, color='black', rotation=90)

    # Show the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)  # Close the figure to avoid memory issues
