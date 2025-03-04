# MLB Data Pipeline

## Overview
A comprehensive data engineering pipeline that extracts, processes, and visualizes MLB (Major League Baseball) data from various sources. This project demonstrates end-to-end data engineering practices from data ingestion to visualization.

## Architecture
This project follows modern data engineering architecture as illustrated in the DE-Flowchart-030325:

- **Data Sources**: MLB.com Articles, Box Scores APIs, Game Statistics
- **Data Ingestion**: Python scrapers and API clients
- **Workflow Orchestration**: Apache Airflow
- **Data Lake**: Google Cloud Storage (Raw and Processed zones)
- **Data Warehouse**: Google BigQuery (Raw, Staging, and Analytics layers)
- **Transformations**: dbt models
- **Visualization**: Looker Studio dashboards

## Tech Stack
- **Infrastructure**: Terraform, Google Cloud Platform
- **Data Processing**: Python, Pandas
- **Orchestration**: Apache Airflow
- **Data Storage**: Google Cloud Storage, BigQuery
- **Transformations**: dbt
- **Containerization**: Docker, Docker Compose
- **Visualization**: Looker Studio

## Project Structure
mlb-data-pipeline/
├── README.md
├── deploy.sh
├── cloudbuild.yaml
├── docker-compose.yml
│
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── monitoring.tf
│
├── scrapers/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── mlb_scraper.py
│   ├── boxscore_client.py
│   └── game_stats_client.py
│
├── airflow/
│   ├── docker-compose.yaml
│   ├── dags/
│   │   ├── mlb_ingestion_dag.py
│   │   └── dbt_transformation_dag.py
│   └── plugins/
│       └── operators/
│           └── custom_operators.py
│
└── dbt/
└── mlb_transformations/
├── dbt_project.yml
├── packages.yml
├── models/
│   ├── schema.yml
│   ├── staging/
│   │   ├── stg_articles.sql
│   │   ├── stg_boxscores.sql
│   │   └── stg_game_stats.sql
│   ├── intermediate/
│   │   ├── int_team_games.sql
│   │   └── int_team_stats.sql
│   └── marts/
│       ├── team_performance.sql
│       └── game_statistics.sql
├── analysis/
│   └── data_quality_checks.sql
├── macros/
│   └── date_utils.sql
├── seeds/
│   └── team_metadata.csv
└── tests/
└── assert_positive_runs.sql

## Setup Instructions

### Prerequisites
- GitHub account (for Codespaces)
- Google Cloud Platform account
- gcloud CLI installed (if working locally)

### Development Environment
This project uses GitHub Codespaces for development to provide a consistent environment and avoid resource constraints on local machines.

1. Clone this repository
2. Open in GitHub Codespaces
3. The devcontainer configuration will automatically set up all necessary tools and dependencies

### Local Development Alternative
If you prefer local development:
```bash```
# Clone the repository
git clone https://github.com/yourusername/mlb-data-pipeline.git
cd mlb-data-pipeline

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start local services
docker-compose up -d

# Authenticate with GCP
gcloud auth login

# Set your project
gcloud config set project your-gcp-project-id

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable composer.googleapis.com
gcloud services enable monitoring.googleapis.com

# Navigate to terraform directory
cd terraform

# Initialize terraform
terraform init

# Apply the configuration
terraform apply -var="project_id=your-gcp-project-id"
