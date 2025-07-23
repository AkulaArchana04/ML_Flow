import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
import mlflow.sklearn 
import joblib

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv("airbnb_listings.csv")
x = df.drop(columns=["ListingID", "PricePerNight"])
y = df["PricePerNight"]

# Define the column transformer for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), ["City", "RoomType"])
    ],
    remainder="passthrough"
)

# Build the pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ]
)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Train the model inside an MLflow run
with mlflow.start_run():
    model.fit(x_train, y_train)
    joblib.dump(model, "airbnb.pkl")
    print("✅ Model trained and saved as airbnb.pkl")
