from datetime import datetime, timedelta
import os
import time
import requests
from bs4 import BeautifulSoup
import sys
sys.path.append('/opt/airflow')
#from agent_framework.orchestrator import OrchestratorAgent
from agent_framework.utils import save_json #, format_plan_as_markdown

from dotenv import load_dotenv

# Explicitly specify the path if needed
load_dotenv()

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

# Import functions from your module
# from mlb_pipeline.pipeline import (
#     #scrape_article,
#     #store_in_gcs,
#     #get_chroma_collection,
#     #embed_and_insert,
#     #format_script_for_tts,
#     #generate_audio_with_your_voice,
#     #upload_podcast_to_gcs
# )

# Import agent framework components
#from agent_framework.orchestrator import OrchestratorAgent
#from agent_framework.utils import save_json, format_plan_as_markdown

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 15),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "mlb_agent_framework_dag",
    default_args=default_args,
    description="Orchestrates the MLB podcast automation with agent framework",
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    def get_latest_articles_from_rss(limit: int = 5) -> list[str]:
        import xml.etree.ElementTree as ET
        """
        Pulls latest article URLs from MLB.com RSS feed.
        """
        rss_url = "https://www.mlb.com/feeds/news/rss.xml"
        response = requests.get(rss_url)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall("./channel/item")

        article_urls = []
        for item in items[:limit]:
            link = item.find("link").text
            if link:
                article_urls.append(link)

        return article_urls


    def scrape_and_store(**kwargs):
        from mlb_pipeline.pipeline import scrape_article, store_in_gcs
        """Scrape MLB articles and store them in GCS."""
        article_urls = [
        "https://www.mlb.com/news/mlb-2025-bbwaa-award-winners-predictions",
        "https://www.mlb.com/news/aaron-judge-hits-first-spring-training-home-run",
        "https://www.mlb.com/news/mlb-power-rankings-before-opening-day-2025",
        
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
        
        # Store in GCS
        blob_name = f"articles/{datetime.now().strftime('%Y-%m-%d')}/articles_batch.json"
        store_in_gcs(scraped_data, os.getenv("GCS_BUCKET_NAME"), blob_name)
        kwargs["ti"].xcom_push(key="scraped_data", value=scraped_data)
        kwargs["ti"].xcom_push(key="article_urls", value=article_urls)

    scrape_and_store_task = PythonOperator(
        task_id="scrape_and_store",
        python_callable=scrape_and_store,
        provide_context=True,
    )

    def embed_update_vector_db(**kwargs):
        from mlb_pipeline.pipeline import get_chroma_collection, embed_and_insert
        """Embed articles and update the vector database."""
        scraped_data = kwargs["ti"].xcom_pull(key="scraped_data", task_ids="scrape_and_store")
        collection = get_chroma_collection()
        embed_and_insert(scraped_data, collection)

    embed_update_task = PythonOperator(
        task_id="embed_update_vector_db",
        python_callable=embed_update_vector_db,
        provide_context=True,
    )

    def create_podcast_plan(**kwargs):
        from agent_framework.orchestrator import OrchestratorAgent
        from agent_framework.utils import format_plan_as_markdown

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")

        # You can still pull scraped data if needed
        scraped_data = kwargs["ti"].xcom_pull(key="scraped_data", task_ids="scrape_and_store")

        # Instead of generating a context, hardcode a focused topic
        custom_topic = "Spring training storylines: Judge's power, BBWAA predictions, and 2025 Opening Day power rankings"

        kwargs["ti"].log.info("Initializing orchestrator agent")
        orchestrator = OrchestratorAgent(openai_api_key=openai_api_key)

        kwargs["ti"].log.info("Creating podcast plan with custom topic")
        plan = orchestrator.create_podcast_plan(
            topic=custom_topic,
            special_focus="insightful narrative around key preseason developments"
        )

        kwargs["ti"].xcom_push(key="podcast_plan", value=plan.dict())

        plan_md = format_plan_as_markdown(plan.dict())
        kwargs["ti"].log.info(f"Podcast Plan:\n{plan_md}")

        return plan.dict()

    create_plan_task = PythonOperator(
        task_id="create_podcast_plan",
        python_callable=create_podcast_plan,
        provide_context=True,
    )

    def execute_podcast_plan(**kwargs):
        from agent_framework.orchestrator import OrchestratorAgent, PodcastTaskPlan
        """Execute the podcast plan created by the orchestrator agent."""
        # Retrieve the podcast plan
        plan_dict = kwargs["ti"].xcom_pull(key="podcast_plan", task_ids="create_podcast_plan")
        
        if not plan_dict:
            kwargs["ti"].log.error("No podcast plan found in XCom")
            return None
            
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")
            
        # Create PodcastTaskPlan object from dict
        from agent_framework.orchestrator import PodcastTaskPlan
        plan = PodcastTaskPlan(**plan_dict)
        
        # Initialize the orchestrator agent
        orchestrator = OrchestratorAgent(openai_api_key=openai_api_key)
        
        # Execute the podcast plan
        kwargs["ti"].log.info("Executing podcast plan")
        results = orchestrator.execute_podcast_plan(plan)
        
        # Save key results to XCom
        kwargs["ti"].xcom_push(key="podcast_script", value=results.get("final_script", ""))
        
        # Save the complete results dictionary
        kwargs["ti"].xcom_push(key="podcast_results", value=results)
        
        return results

    execute_plan_task = PythonOperator(
        task_id="execute_podcast_plan",
        python_callable=execute_podcast_plan,
        provide_context=True,
    )

    def generate_audio_task(**kwargs):
        from mlb_pipeline.pipeline import format_script_for_tts, generate_audio_with_your_voice
        """Generate audio from the podcast script."""
        # Get the script from XCom
        script = kwargs['ti'].xcom_pull(key="podcast_script", task_ids="execute_podcast_plan")
        
        if not script:
            kwargs['ti'].log.error("No podcast script found in XCom")
            return None
        
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
            # Save the file paths to XCom
            kwargs['ti'].xcom_push(key="audio_file", value=result)
            kwargs['ti'].xcom_push(key="script_file", value=script_file)
            return result
        else:
            kwargs['ti'].log.error("Failed to generate audio")
            return None

    generate_audio = PythonOperator(
        task_id="generate_audio",
        python_callable=generate_audio_task,
        provide_context=True,
    )

    def upload_to_gcs_task(**kwargs):
        from mlb_pipeline.pipeline import upload_podcast_to_gcs
        """Upload generated podcast files to GCS."""
        audio_file = kwargs['ti'].xcom_pull(key="audio_file", task_ids="generate_audio")
        script_file = kwargs['ti'].xcom_pull(key="script_file", task_ids="generate_audio")
        
        if not audio_file or not os.path.exists(audio_file):
            kwargs['ti'].log.error("Audio file not found")
            return None
            
        if not script_file or not os.path.exists(script_file):
            kwargs['ti'].log.error("Script file not found")
            # Continue with audio upload only
        
        # Upload files to GCS
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        timestamp = datetime.now().strftime("%Y%m%d")
        
        audio_blob = f"podcasts/audio/{timestamp}/{os.path.basename(audio_file)}"
        script_blob = f"podcasts/scripts/{timestamp}/{os.path.basename(script_file)}"
        
        audio_gcs_path = upload_podcast_to_gcs(audio_file, bucket_name, audio_blob)
        kwargs['ti'].log.info(f"Uploaded podcast audio to: {audio_gcs_path}")
        
        if script_file and os.path.exists(script_file):
            script_gcs_path = upload_podcast_to_gcs(script_file, bucket_name, script_blob)
            kwargs['ti'].log.info(f"Uploaded podcast script to: {script_gcs_path}")
            return {"audio": audio_gcs_path, "script": script_gcs_path}
        
        return {"audio": audio_gcs_path}

    upload_to_gcs = PythonOperator(
        task_id="upload_to_gcs",
        python_callable=upload_to_gcs_task,
        provide_context=True,
    )

    # Create a task for monitoring and collecting feedback
    def collect_feedback(**kwargs):
        from mlb_pipeline.pipeline import upload_podcast_to_gcs

        """Placeholder for a feedback collection system."""
        # In a real implementation, this would integrate with your monitoring system
        # to collect feedback on the podcast and store it for future improvements
        kwargs['ti'].log.info("Collecting feedback on the podcast (placeholder)")
        
        # Here you could implement:
        # 1. Sending out a feedback survey to subscribers
        # 2. Analyzing download/streaming metrics
        # 3. Gathering social media mentions
        # 4. Storing this data for future analysis
        
        podcast_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "downloads": 0,  # placeholder
                "shares": 0,     # placeholder
                "comments": 0,   # placeholder
                "likes": 0       # placeholder
            },
            "feedback": []  # would contain actual feedback in production
        }
        
        # Save this to a feedback collection system
        output_dir = "/opt/airflow/feedback_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        feedback_file = f"{output_dir}/feedback_{datetime.now().strftime('%Y%m%d')}.json"
        
        import json
        with open(feedback_file, 'w') as f:
            json.dump(podcast_info, f, indent=2)
            
        kwargs['ti'].log.info(f"Saved feedback data to {feedback_file}")

        bucket_name = os.getenv("GCS_BUCKET_NAME")
        blob_name = f"feedback/{os.path.basename(feedback_file)}"

        upload_podcast_to_gcs(feedback_file, bucket_name, blob_name)
        kwargs["ti"].log.info(f"Uploaded feedback to GCS: gs://{bucket_name}/{blob_name}")

        return feedback_file

    collect_feedback_task = PythonOperator(
        task_id="collect_feedback",
        python_callable=collect_feedback,
        provide_context=True,
    )
    
    # Set task dependencies
    scrape_and_store_task >> embed_update_task >> create_plan_task >> execute_plan_task >> generate_audio >> upload_to_gcs >> collect_feedback_task