# dags/mlb_pipeline_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlb_data_pipeline',
    default_args=default_args,
    description='MLB Data Pipeline',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Task to extract data
    extract_data = BashOperator(
        task_id='extract_data',
        bash_command='echo "Extracting MLB data" && sleep 5',
    )

    # Task to run dbt
    run_dbt = BashOperator(
        task_id='run_dbt',
        bash_command='echo "Running dbt models" && sleep 5',
    )

    # Define task dependencies
    extract_data >> run_dbt