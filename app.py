import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("./model.pkl","rb"))
data= pd.read_csv("./clean.csv")

st.title("Car Price Predictor")


name = st.selectbox("Car Model",options=data["name"].sort_values().unique())
fuel = st.selectbox("Fuel",options=data["fuel_type"].unique())
year = st.selectbox("Year",options=data["year"].sort_values().unique())
kms = st.selectbox("Km driven",options=data["kms_driven"].sort_values().unique())
company = st.selectbox("Company",options=data["company"].sort_values().unique())

if st.button("Predict"):
  query = pd.DataFrame([[name,company,year,kms,fuel]],columns=["name","company","year","kms_driven","fuel_type"])
  result = model.predict(query)
  result = f'{result[0]:.2f}'
  st.write("INR = ",result)

