import joblib
import pandas as pd
import streamlit as st

st.title("Student Progress Predictor")

model = joblib.load("progress_model.pkl")

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=4.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)
internal_marks = st.number_input("Internal Marks", min_value=0.0, max_value=100.0, value=22.0)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=78.0)

if st.button("Predict"):
  input_df = pd.DataFrame([{
      "study_hours": study_hours,
      "attendance": attendance,
      "internal_marks": internal_marks,
      "previous_score": previous_score,
  }])
  prediction = model.predict(input_df)[0]
  st.success(f"Predicted Final Percentage: {round(prediction, 2)}")