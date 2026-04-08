import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("RDF_model.pkl")
scaler = joblib.load("Scaler.pkl")
encoder = joblib.load("encoder.pkl")
selected_features = joblib.load("Selected_features.pkl")

st.set_page_config(page_title="Airline Satisfaction Predictor", page_icon="✈", layout="wide")


st.title("✈ Airline Passenger Satisfaction Prediction")
st.write("Fill the passenger experience details below to predict satisfaction level.")

st.write("---")

with st.form("form"):

    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox("⚥ Gender", ["Male", "Female"])
        Customer_Type = st.selectbox("🛂 Customer Type", ["Loyal Customer", "disloyal Customer"])
        travel_type = st.selectbox("🛄 Type of Travel", ["Business travel", "Personal Travel"])
        Class = st.selectbox("🎟 Seat Class", ["Eco", "Eco Plus", "Business"])
        flight_distance = st.number_input("📏 Flight Distance (km)", min_value=30, max_value=20000)

    with col2:
        cleanliness = st.slider("🧼 Cleanliness", 0, 5)
        wifi = st.slider("📡 Inflight WiFi Service", 0, 5)
        entertainment = st.slider("🎬 Inflight Entertainment", 0, 5)
        seat_comfort = st.slider("💺 Seat Comfort", 0, 5)
        leg_room = st.slider("🦵 Leg Room Service", 0, 5)
        on_boarding = st.slider("🚪 Online Boarding", 0, 5)
        baggage = st.slider("🎒 Baggage Handling", 0, 5)
        infl_service = st.slider("💼 Inflight Service", 0, 5)
        onboard_service = st.slider("🛬 On-board Service", 0, 5)

    submit = st.form_submit_button("🔍 Predict Satisfaction")

if submit:

    input_df = pd.DataFrame({
        "Gender": [Gender],
        "Customer Type": [Customer_Type],
        "Class": [Class],
        "Inflight wifi service": [wifi],
        "Cleanliness": [cleanliness],
        "Baggage handling": [baggage],
        "Seat comfort": [seat_comfort],
        "Flight Distance": [flight_distance],
        "Leg room service": [leg_room],
        "Type of Travel": [travel_type],
        "Inflight service": [infl_service],
        "On-board service": [onboard_service],
        "Inflight entertainment": [entertainment],
        "Online boarding": [on_boarding]
    })

    categorical_cols = ["Gender", "Customer Type", "Class", "Type of Travel"]
    for col in categorical_cols:
        input_df[col] = encoder[col].transform(input_df[col])

    try:
        input_df = input_df[selected_features]
    except:
        missing = set(selected_features) - set(input_df.columns)
        st.error(f"❌ Missing required features: {missing}")
        st.stop()

    input_df = scaler.transform(input_df)
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]

    st.write("---")
    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        st.success(f"🙂 Passenger is **SATISFIED** ✔ ")
    else:
        st.error(f"🙁 Passenger is **DISSATISFIED** ❌ ")