from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import json
from google.cloud import storage
import os

# Google Cloud Storage settings
GCS_BUCKET = "mlb-pipeline-deji-mlb-pipeline"
GCS_FOLDER = "mlb_schedule"

# MLB API endpoint
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Default Airflow arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 3, 5),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "mlb_schedule_to_gcs",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
)

def fetch_mlb_schedule():
    """Fetch today's MLB schedule from the MLB Stats API."""
    today = datetime.today().strftime("%Y-%m-%d")
    params = {"sportId": 1, "date": today}  # sportId=1 for MLB
    response = requests.get(MLB_SCHEDULE_URL, params=params)
    
    if response.status_code == 200:
        schedule_data = response.json()
        local_file_path = f"/tmp/mlb_schedule_{today}.json"

        # Save to a local file
        with open(local_file_path, "w") as f:
            json.dump(schedule_data, f, indent=4)
        
        return local_file_path
    else:
        raise Exception(f"Failed to fetch MLB schedule: {response.status_code}")

def upload_to_gcs(local_file_path):
    """Upload a file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    
    today = datetime.today().strftime("%Y-%m-%d")
    gcs_path = f"{GCS_FOLDER}/mlb_schedule_{today}.json"
    
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_file_path)
    
    print(f"Uploaded {local_file_path} to gs://{GCS_BUCKET}/{gcs_path}")

fetch_schedule_task = PythonOperator(
    task_id="fetch_mlb_schedule",
    python_callable=fetch_mlb_schedule,
    dag=dag,
)

upload_to_gcs_task = PythonOperator(
    task_id="upload_to_gcs",
    python_callable=upload_to_gcs,
    op_args=["/tmp/mlb_schedule_{{ ds }}.json"],
    dag=dag,
)

fetch_schedule_task >> upload_to_gcs_task

