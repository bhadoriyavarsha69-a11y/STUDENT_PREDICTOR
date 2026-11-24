import joblib
import pandas as pd

model = joblib.load("progress_model.pkl")


new_student = pd.DataFrame([{
    "study_hours": 5,
    "attendance": 85,
    "internal_marks": 25,
    "previous_score": 78
}])

prediction = model.predict(new_student)[0]
print("Predicted Final Percentage:", round(prediction, 2))