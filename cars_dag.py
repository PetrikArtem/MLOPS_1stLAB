import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from train_model import train_model

DATA_PATH = "/home/airr/airflow/dags/cars_project"

def download_data():
    url = 'https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv'
    df = pd.read_csv(url)
    df.to_csv(f"{DATA_PATH}/cars.csv", index=False)

def clear_data():
    df = pd.read_csv(f"{DATA_PATH}/cars.csv")
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    
    df = df[(df.Year > 1970) & (df.Year < 2022)]
    df = df[(df.Distance < 1e6) & (df.Distance > 1000)]
    df = df[(df['Engine_capacity(cm3)'] > 200) & (df['Engine_capacity(cm3)'] < 5000)]
    df = df[(df['Price(euro)'] > 100) & (df['Price(euro)'] < 1e5)]
    
    df = df.reset_index(drop=True)
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    df.to_csv(f"{DATA_PATH}/df_clear.csv", index=False)

with DAG(
    dag_id="cars_ml_pipeline",
    start_date=datetime(2026, 3, 22),
    schedule_interval=None,
    catchup=False
) as dag:

    download_task = PythonOperator(task_id="download_cars", python_callable=download_data)
    clear_task = PythonOperator(task_id="clear_cars", python_callable=clear_data)
    train_task = PythonOperator(task_id="train_cars", python_callable=train_model)

    download_task >> clear_task >> train_task
