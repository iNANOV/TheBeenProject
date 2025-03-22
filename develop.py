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

import morethemes as mt
print(type(mt.themes))  # Check if it's a list, dict, or module
print(mt.themes)  # Print the content

usr = "admin"
pwd ="iXughds67"
url =  "85.235.65.151" #'mongo'
database = 'ptsc'


db_uri = f"mongodb://{usr}:{pwd}@{url}/{database}?authSource=admin&retryWrites=true&w=majority"
client = MongoClient(db_uri)
db = client.get_database(database)

pd.DataFrame(list(db.Models.find({}, {})))


