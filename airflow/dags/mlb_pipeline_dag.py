from datetime import datetime, timedelta
import os
import time

from dotenv import load_dotenv

# Explicitly specify the path if needed
#load_dotenv("/opt/airflow/.env")
load_dotenv()

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Import functions from your module
from mlb_pipeline.pipeline import (
    scrape_article,
    store_in_gcs,
    get_chroma_collection,
    embed_and_insert,
    test_query,
    rag_pipeline,
    generate_podcast_script
)

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 15),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "mlb_pipeline_dag",
    default_args=default_args,
    description="Orchestrates the MLB podcast automation pipeline",
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    def scrape_and_store(**kwargs):
        article_urls = [
            "https://www.mlb.com/news/mlb-power-rankings-before-opening-day-2025",
            # Add more URLs as needed
        ]
        scraped_data = []
        for url in article_urls:
            kwargs["ti"].log.info(f"Scraping: {url}")
            try:
                article = scrape_article(url)
                scraped_data.append(article)
                time.sleep(2)
            except Exception as e:
                kwargs["ti"].log.error(f"Error scraping {url}: {e}")
        blob_name = f"articles/{datetime.now().strftime('%Y-%m-%d')}/articles_batch.json"
        store_in_gcs(scraped_data, os.getenv("GCS_BUCKET_NAME"), blob_name)
        kwargs["ti"].xcom_push(key="scraped_data", value=scraped_data)

    scrape_and_store_task = PythonOperator(
        task_id="scrape_and_store",
        python_callable=scrape_and_store,
        provide_context=True,
    )

    def embed_update_vector_db(**kwargs):
        scraped_data = kwargs["ti"].xcom_pull(key="scraped_data", task_ids="scrape_and_store")
        collection = get_chroma_collection()
        embed_and_insert(scraped_data, collection)

    embed_update_task = PythonOperator(
        task_id="embed_update_vector_db",
        python_callable=embed_update_vector_db,
        provide_context=True,
    )

    def run_rag_demo(**kwargs):
        query_str = "Who is the 2nd team in MLB's power rankings?"
        answer = rag_pipeline(query_str)
        kwargs["ti"].log.info(f"Final Answer: {answer}")

    rag_query_task = PythonOperator(
        task_id="rag_query_demo",
        python_callable=run_rag_demo,
        provide_context=True,
    )

    def generate_script_task(**kwargs):
        query = "Generate a podcast script about MLB power rankings before opening day."
        script = generate_podcast_script(query)
        kwargs['ti'].log.info(f"Generated Podcast Script:\n{script}")

    generate_script = PythonOperator(
        task_id="generate_podcast_script",
        python_callable=generate_script_task,
        provide_context=True,
    )

    scrape_and_store_task >> embed_update_task >> rag_query_task >> generate_script
