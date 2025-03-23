# main.py
"""
Entry point for the MLB podcast agent framework
"""
import os
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from agent_framework.orchestrator import OrchestratorAgent
from agent_framework.utils import save_json, format_plan_as_markdown
from mlb_pipeline.pipeline import generate_audio_with_your_voice, upload_podcast_to_gcs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_framework.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main(topic: Optional[str] = None, special_focus: Optional[str] = None):
    """Run the MLB podcast agent framework pipeline."""
    # Load environment variables
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables")
    
    # Initialize the orchestrator agent
    logger.info("Initializing orchestrator agent")
    orchestrator = OrchestratorAgent(openai_api_key=openai_api_key)
    
    # Create a podcast plan
    logger.info(f"Creating podcast plan on topic: {topic or 'latest MLB news'}")
    plan = orchestrator.create_podcast_plan(
        topic=topic,
        special_focus=special_focus
    )
    
    # Save the plan to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = f"output/plans/podcast_plan_{timestamp}.json"
    save_json(plan.dict(), plan_file)
    logger.info(f"Saved podcast plan to {plan_file}")
    
    # Display the plan in a readable format
    plan_md = format_plan_as_markdown(plan.dict())
    logger.info(f"Podcast Plan:\n{plan_md}")
    
    # Execute the podcast plan
    logger.info("Executing podcast plan")
    results = orchestrator.execute_podcast_plan(plan)
    
    # Save the results
    results_file = f"output/results/podcast_results_{timestamp}.json"
    save_json(results, results_file)
    logger.info(f"Saved podcast results to {results_file}")
    
    # Generate audio from the final script
    logger.info("Generating podcast audio")
    script = results["final_script"]
    
    # Create output directory if it doesn't exist
    output_dir = "output/audio"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    audio_file = f"{output_dir}/podcast_{timestamp}.mp3"
    
    # Save the script for reference
    script_file = f"output/scripts/script_{timestamp}.txt"
    os.makedirs(os.path.dirname(script_file), exist_ok=True)
    with open(script_file, 'w') as f:
        f.write(script)
    
    # Generate the audio with the script
    audio_result = generate_audio_with_your_voice(script, audio_file)
    
    if audio_result:
        logger.info(f"Successfully generated audio at: {audio_result}")
        
        # Upload to GCS
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if bucket_name:
            logger.info("Uploading podcast to GCS")
            audio_blob = f"podcasts/audio/{timestamp}/{os.path.basename(audio_file)}"
            script_blob = f"podcasts/scripts/{timestamp}/{os.path.basename(script_file)}"
            
            gcs_audio_path = upload_podcast_to_gcs(audio_file, bucket_name, audio_blob)
            gcs_script_path = upload_podcast_to_gcs(script_file, bucket_name, script_blob)
            
            logger.info(f"Uploaded podcast audio to: {gcs_audio_path}")
            logger.info(f"Uploaded podcast script to: {gcs_script_path}")
    else:
        logger.error("Failed to generate audio")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the MLB podcast agent framework")
    parser.add_argument("--topic", type=str, help="Specific topic for the podcast")
    parser.add_argument("--focus", type=str, help="Special focus area for the podcast")
    
    args = parser.parse_args()
    
    main(topic=args.topic, special_focus=args.focus)