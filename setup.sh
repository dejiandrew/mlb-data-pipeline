#!/bin/bash
# setup.sh - MLB Pipeline Setup Script

# Exit on error
set -e

echo "MLB Pipeline Setup Script - Starting..."

# Create required directories
echo "Creating directories..."
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/data
mkdir -p dbt/profiles
mkdir -p vector_db

# Create .env file
echo "Creating .env file..."
cat > .env << EOL
GCS_BUCKET_NAME=mlb-pipeline-deji-mlb-pipeline
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/gcp-key.json
OPENAI_API_KEY=your-openai-api-key
ELEVENLABS_API_KEY=your-elevenlabs-api-key
ELEVENLABS_VOICE_ID=your-voice-id
CHROMA_PERSIST_DIR=/opt/airflow/vector_db
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EOL

echo "Please edit the .env file with your actual API keys"

# Download GCP service account key
echo "IMPORTANT: You need to download your GCP service account key"
echo "1. Go to GCP Console > IAM & Admin > Service Accounts"
echo "2. Find the 'airflow-service' account"
echo "3. Create a new key and download it as 'gcp-key.json'"
echo "4. Place it in the root directory of this project"

# Fix Airflow permissions
echo "Setting correct permissions for Airflow directories..."
if [ -d "airflow/dags" ] && [ -d "airflow/logs" ] && [ -d "airflow/plugins" ]; then
    sudo chown -R 50000:0 airflow/dags airflow/logs airflow/plugins
    echo "Permissions set successfully"
else
    echo "Warning: Some Airflow directories don't exist yet"
fi

# Install dependencies (if needed)
if [ ! -f "requirements.txt" ]; then
    echo "Creating requirements.txt file..."
    cat > requirements.txt << EOL
apache-airflow
google-cloud-storage
python-dotenv
sentence-transformers
chromadb
openai
requests
beautifulsoup4
EOL
fi

echo "Setup complete! Next steps:"
echo "1. Make sure gcp-key.json is in the root directory"
echo "2. Edit .env file with your actual API keys"
echo "3. Run 'docker-compose up -d' to start the containers"
echo "4. Access Airflow UI at http://localhost:8080"