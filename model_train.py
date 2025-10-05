import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv('amazondelivery_processed.csv')

# Define features and target
features = ['Agent_Age', 'Agent_Rating', 'Distance', 'Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'OrderHour', 'OrderDayOfWeek']
target = 'Delivery_Time'

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor()
}

# MLflow experiment logging
for name, model in models.items():
    mlflow.start_run(run_name=name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)
    
    mlflow.sklearn.log_model(model, artifact_path=name)
    mlflow.end_run()
    
    print(f'{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
