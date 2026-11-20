
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Find the project folder and load the dataset
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "hospital_billing_synthetic_dataset.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select input features and target
features = [
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results",
    "Treatment Type",
    "Room Type",
    "Length of Stay",
    "ICU Days",
    "Number of Lab Tests"
]

X = df[features]
y = df["Billing Amount"]

# 3. Separate numerical and categorical columns
categorical_features = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results",
    "Treatment Type",
    "Room Type"
]

numerical_features = [
    "Age",
    "Length of Stay",
    "ICU Days",
    "Number of Lab Tests"
]

# 4. Convert text categories into numerical values
preprocessor = ColumnTransformer(
    transformer = [
        ( "categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
         ("numerical","passthrough",numerical_features)
    ]
)


# 5. Create the machine learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 8. Predict hospital bills
predictions = model.predict(X_test)

# 9. Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("-------------------------")
print(f"Mean Absolute Error: Rs. {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# 10. Save the trained model
model_path = project_dir / "models" / "hospital_billing_model.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)
