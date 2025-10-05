Amazon Delivery Time Prediction
Project Description
This project predicts delivery times for e-commerce orders using machine learning models trained on features like agent info, distance, traffic, weather, and order time. It includes data preprocessing, model training with MLflow experiment tracking, and a Streamlit app for predictions.

How to Run

1. Data Preparation : 
python data_prep.py

2. Exploratory Data Analysis : 
python eda.py

3. Model Training & Logging with MLflow : 
python model_train.py

4. Using MLflow UI (Experiment Tracking Dashboard)
Launch the MLflow UI locally to visualize and compare models and experiments : 

mlflow ui

Open your browser and go to: http://localhost:5000

The UI lets you:

View metrics (RMSE, MAE, R2) for each run

Compare models side-by-side

Inspect parameters, artifacts, and more

Select the best model for deployment

5. Run the Streamlit app for predictions : 
streamlit run app.py

This workflow helps track experiments rigorously and use the best model interactively.
