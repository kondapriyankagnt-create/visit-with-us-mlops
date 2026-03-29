
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("Wellness Tourism Purchase Predictor")

age = st.number_input("Age", 18, 70)
income = st.number_input("Income", 10000, 200000)
trips = st.number_input("Trips", 0, 20)
followups = st.number_input("Followups", 0, 10)

if st.button("Predict"):
    data = np.array([[age, income, trips, followups]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("Will Purchase ✅")
    else:
        st.error("Will NOT Purchase ❌")
