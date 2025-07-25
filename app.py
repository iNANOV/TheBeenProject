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
from overview import show_overview_page
from welcome import show_welcome_page
from research import show_research_page

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

# Function to authenticate users
def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and check_password_hash(user["password"], password):
        return True
    return False

st.title("The Been Project")
st.image("been.png", use_container_width=True)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None  # Store username globally

# Show Welcome page always (outside authentication)
show_welcome_page()

# Login system
if not st.session_state["authenticated"]:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username  # Store username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# If authenticated, show the main content
if st.session_state["authenticated"]:
    # Sidebar customization
    st.sidebar.title("Customization")
    available_themes = [
        "economist", "wsj", "urban", "minimal", "ft", 
        "nature", "retro", "yellowish", "darker", "monoblue"
    ]
    selected_theme = st.sidebar.selectbox("Choose a Matplotlib Theme", available_themes, index=0)
    mt.set_theme(selected_theme)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Research"])

    # Show the selected page
    if page == "Overview":
        show_overview_page()
    elif page == "Research":
        show_research_page()

        
