import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def train_model():
    mlflow.set_tracking_uri("file:///home/airr/airflow/mlruns")
    
    if not os.path.exists("/home/airr/airflow/mlruns"):
        os.makedirs("/home/airr/airflow/mlruns")

    df = pd.read_csv('/home/airr/airflow/dags/cars_project/df_clear.csv')
    
    target_col = [c for c in df.columns if 'Price' in c or 'price' in c]
    
    col_name = target_col[0]
    X = df.drop(col_name, axis=1)
    y = df[col_name]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    mlflow.set_experiment("linear_model_cars")
    
    with mlflow.start_run():
        params = {'alpha': [0.01], 'penalty': ['l2'], 'max_iter': [1000]}
        sgd = SGDRegressor(random_state=42)
        grid = GridSearchCV(sgd, params, cv=5)
        grid.fit(X_scaled, y)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_scaled)
        
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y, y_pred)))
        mlflow.log_metric("r2", r2_score(y, y_pred))
        mlflow.sklearn.log_model(best_model, "model")
        
        joblib.dump(best_model, '/home/airr/airflow/dags/cars_project/lr_cars.pkl')

if __name__ == "__main__":
    train_model()
