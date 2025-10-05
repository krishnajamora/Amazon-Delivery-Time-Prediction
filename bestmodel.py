import mlflow.sklearn
import joblib

# Replace <run_id> with your actual run ID from MLflow
run_id = "8f5d6667d289456f85b51fc64b5cbfb5"
model_name = "XGBoost"

# Load model from MLflow
model_uri = f"runs:/{run_id}/{model_name}"
model = mlflow.sklearn.load_model(model_uri)

# Save the model locally
joblib.dump(model, "best_model.pkl")
print("Model saved as best_model.pkl")
