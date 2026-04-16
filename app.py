import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('models/model.pkl', 'rb'))

st.title("AI Remote Patient Monitoring System")

heart_rate = st.number_input("Heart Rate")
oxygen = st.number_input("Oxygen Level")
temp = st.number_input("Temperature")

if st.button("Predict Risk"):
    input_data = np.array([[heart_rate, oxygen, temp]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Health Risk Detected!")
    else:
        st.success("Patient is Stable")