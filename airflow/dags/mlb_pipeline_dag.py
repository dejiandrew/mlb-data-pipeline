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
    generate_podcast_script,
    format_script_for_tts,
    generate_audio_with_your_voice,
    upload_podcast_to_gcs
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

    def fetch_daily_article_urls(**kwargs):
        import feedparser
        from datetime import datetime

        feed = feedparser.parse("https://www.mlb.com/feeds/news/rss.xml")

        today_str = datetime.utcnow().strftime("%a, %d %b %Y")  # e.g., "Sun, 23 Mar 2025"
        urls = [
            entry.link for entry in feed.entries
            if entry.published.startswith(today_str)
        ]

        # Fallback: if no articles match today’s date, just take the top 3
        if not urls:
            urls = [entry.link for entry in feed.entries[:3]]

        kwargs["ti"].xcom_push(key="article_urls", value=urls)


    fetch_urls_task = PythonOperator(
        task_id="fetch_daily_article_urls",
        python_callable=fetch_daily_article_urls,
        provide_context=True,
    )


    def scrape_and_store(**kwargs):
        from mlb_pipeline.pipeline import scrape_article, store_in_gcs
        import time
        import os
        from datetime import datetime

        # Pull URLs from the previous task
        article_urls = kwargs["ti"].xcom_pull(
            key="article_urls", task_ids="fetch_daily_article_urls"
        )

        if not article_urls:
            kwargs["ti"].log.warning("No article URLs found to scrape.")
            return

        scraped_data = []
        for url in article_urls:
            kwargs["ti"].log.info(f"Scraping: {url}")
            try:
                article = scrape_article(url)
                scraped_data.append(article)
                time.sleep(2)
            except Exception as e:
                kwargs["ti"].log.error(f"Error scraping {url}: {e}")

        # Store scraped data to GCS
        blob_name = f"articles/{datetime.now().strftime('%Y-%m-%d')}/articles_batch.json"
        store_in_gcs(scraped_data, os.getenv("GCS_BUCKET_NAME"), blob_name)

        # Push scraped data forward for other tasks
        kwargs["ti"].xcom_push(key="scraped_data", value=scraped_data)

    # def scrape_and_store(**kwargs):
    #     article_urls = [
    #         #"https://www.mlb.com/news/mlb-power-rankings-before-opening-day-2025",
    #         #"https://www.mlb.com/news/spencer-strider-makes-first-2025-spring-training-start-after-surgery",
    #         "https://www.mlb.com/news/guardians-acquire-nolan-jones-from-rockies-for-tyler-freeman",
    #         # Add more URLs as needed
    #     ]
    #     scraped_data = []
    #     for url in article_urls:
    #         kwargs["ti"].log.info(f"Scraping: {url}")
    #         try:
    #             article = scrape_article(url)
    #             scraped_data.append(article)
    #             time.sleep(2)
    #         except Exception as e:
    #             kwargs["ti"].log.error(f"Error scraping {url}: {e}")
    #     blob_name = f"articles/{datetime.now().strftime('%Y-%m-%d')}/articles_batch.json"
    #     store_in_gcs(scraped_data, os.getenv("GCS_BUCKET_NAME"), blob_name)
    #     kwargs["ti"].xcom_push(key="scraped_data", value=scraped_data)

    scrape_and_store_task = PythonOperator(
        task_id="scrape_and_store",
        python_callable=scrape_and_store,
        provide_context=True,
    )

    def build_query_prompt(**kwargs):
        scraped_data = kwargs["ti"].xcom_pull(key="scraped_data", task_ids="scrape_and_store")

        if not scraped_data:
            kwargs["ti"].log.warning("No scraped data found for prompt generation.")
            return

        titles = [article.get("title", "MLB Update") for article in scraped_data]
        formatted_titles = "; ".join(titles)

        prompt = (
            "Generate a podcast script summarizing these MLB headlines:\n"
            f"{formatted_titles}\n"
            "The tone should be informative, engaging, and friendly — like a host doing a daily baseball roundup. "
            "Include context from the articles where possible."
        )

        kwargs["ti"].log.info(f"Auto-generated prompt:\n{prompt}")
        kwargs["ti"].xcom_push(key="podcast_prompt", value=prompt)
    
    build_prompt_task = PythonOperator(
        task_id="build_query_prompt",
        python_callable=build_query_prompt,
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
        query = kwargs['ti'].xcom_pull(key="podcast_prompt", task_ids="build_query_prompt")
        script = generate_podcast_script(query)
        kwargs['ti'].log.info(f"Generated Podcast Script:\n{script}")
        kwargs['ti'].xcom_push(key="podcast_script", value=script)
        return script

    generate_script = PythonOperator(
        task_id="generate_podcast_script",
        python_callable=generate_script_task,
        provide_context=True,
    )
    
    def generate_audio_task(**kwargs):
        # Get the script from XCom
        script = kwargs['ti'].xcom_pull(key="podcast_script", task_ids="generate_podcast_script")
        
        # Optimize the script for TTS
        optimized_script = format_script_for_tts(script)
        
        # Log the differences for debugging
        kwargs['ti'].log.info("Original script length: %d", len(script))
        kwargs['ti'].log.info("Optimized script length: %d", len(optimized_script))
        
        # Create output directory if it doesn't exist
        output_dir = "/opt/airflow/podcast_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/podcast_{timestamp}.mp3"
        
        # Save the optimized script for reference
        script_file = f"{output_dir}/script_{timestamp}.txt"
        with open(script_file, 'w') as f:
            f.write(optimized_script)
        
        # Generate the audio with the optimized script
        result = generate_audio_with_your_voice(optimized_script, output_file)
        
        if result:
            kwargs['ti'].log.info(f"Successfully generated audio at: {result}")
            kwargs['ti'].log.info(f"Script saved at: {script_file}")
            return result
        else:
            kwargs['ti'].log.error("Failed to generate audio")
            return None

    generate_audio = PythonOperator(
        task_id="generate_audio",
        python_callable=generate_audio_task,
        provide_context=True,
    )

    def upload_podcast_files_to_gcs(**kwargs):
        """Upload generated podcast files to GCS"""
        audio_file = kwargs['ti'].xcom_pull(task_ids='generate_audio')
        if not audio_file or not os.path.exists(audio_file):
            kwargs['ti'].log.error("Audio file not found")
            return None
        
        # Derive script file path from audio file path
        # The script is in the same directory as the audio file
        directory = os.path.dirname(audio_file)
        base_filename = os.path.basename(audio_file).replace('podcast_', 'script_').replace('.mp3', '.txt')
        script_file = os.path.join(directory, base_filename)
        
        # Log the paths to verify
        kwargs['ti'].log.info(f"Audio file path: {audio_file}")
        kwargs['ti'].log.info(f"Script file path: {script_file}")
        
        # Check if script file exists
        if not os.path.exists(script_file):
            kwargs['ti'].log.error(f"Script file not found: {script_file}")
            # Continue with audio upload only
            bucket_name = os.getenv("GCS_BUCKET_NAME")
            timestamp = datetime.now().strftime("%Y%m%d")
            audio_blob = f"podcasts/audio/{timestamp}/{os.path.basename(audio_file)}"
            audio_gcs_path = upload_podcast_to_gcs(audio_file, bucket_name, audio_blob)
            kwargs['ti'].log.info(f"Uploaded podcast audio to: {audio_gcs_path}")
            return {"audio_gcs_path": audio_gcs_path}
        
        # Upload both files to GCS
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        timestamp = datetime.now().strftime("%Y%m%d")
        
        audio_blob = f"podcasts/audio/{timestamp}/{os.path.basename(audio_file)}"
        script_blob = f"podcasts/scripts/{timestamp}/{os.path.basename(script_file)}"
        
        audio_gcs_path = upload_podcast_to_gcs(audio_file, bucket_name, audio_blob)
        script_gcs_path = upload_podcast_to_gcs(script_file, bucket_name, script_blob)
        
        kwargs['ti'].log.info(f"Uploaded podcast audio to: {audio_gcs_path}")
        kwargs['ti'].log.info(f"Uploaded podcast script to: {script_gcs_path}")
        
        return {
            "audio_gcs_path": audio_gcs_path,
            "script_gcs_path": script_gcs_path
        }

    upload_to_gcs_task = PythonOperator(
        task_id="upload_podcast_to_gcs",
        python_callable=upload_podcast_files_to_gcs,
        provide_context=True,
    )
    
    # Set task dependencies
    #scrape_and_store_task >> embed_update_task >> rag_query_task >> generate_script >> generate_audio >> upload_to_gcs_task
    #scrape_and_store_task >> embed_update_task >> generate_script >> generate_audio >> upload_to_gcs_task
    fetch_urls_task >> scrape_and_store_task >> build_prompt_task >> embed_update_task >> generate_script >> generate_audio >> upload_to_gcs_task
