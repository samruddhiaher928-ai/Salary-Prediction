# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("salary_dataset.csv")

# Convert Education column into numbers
encoder = LabelEncoder()
df["Education"] = encoder.fit_transform(df["Education"])

# Features and Target
X = df[["YearsExperience", "Education", "Age"]]
y = df["Salary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "salary_model.pkl")
joblib.dump(encoder, "education_encoder.pkl")

# Accuracy
accuracy = model.score(X_test, y_test)

print("Model trained successfully!")
print(f"Accuracy (R² Score): {accuracy:.2f}")