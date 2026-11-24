import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("student_data.csv")

# Clean basic missing or duplicate rows if any
df = df.drop_duplicates().dropna()

# Features (inputs) and target (output)
X = df[["study_hours", "attendance", "internal_marks", "previous_score"]]
y = df["final_percentage"]

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on unseen test rows
predictions = model.predict(X_test)
print("R2 Score:", round(r2_score(y_test, predictions), 3))
print("Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, predictions), 2))

# Inspect learned weights for your PBL explanation
for feature, coef in zip(X.columns, model.coef_):
    print(f"Weight for {feature}: {round(float(coef), 3)}")
print("Intercept:", round(float(model.intercept_), 3))

# Save trained model
joblib.dump(model, "progress_model.pkl")
print("Model saved successfully as progress_model.pkl")