flowchart TD
    %% Data Sources
    subgraph DataSources["Data Sources"]
        MLB[MLB.com Articles]
        BoxScore[Box Scores APIs]
        GameStats[Game Statistics]
    end
    
    %% Infrastructure as Code
    subgraph IaC["Infrastructure as Code"]
        Terraform[Terraform]
    end
    
    %% Data Ingestion
    subgraph DataIngestion["Data Ingestion"]
        direction TB
        Scrapers[Web Scrapers & API Clients]
        
        subgraph Orchestration["Workflow Orchestration"]
            Airflow[Airflow DAGs]
        end
    end
    
    %% Data Lake
    subgraph DataLake["Data Lake (GCS)"]
        GCSRaw[(Raw Zone)]
        GCSProcessed[(Processed Zone)]
    end
    
    %% Data Warehouse
    subgraph DataWarehouse["Data Warehouse (BigQuery)"]
        BQRaw[(BigQuery Raw Tables)]
        BQStaging[(BigQuery Staging Tables)]
        
        subgraph Transformations["Data Transformations"]
            dbt[dbt Models]
        end
        
        BQAnalytics[(BigQuery Analytics Tables)]
    end
    
    %% Dashboard
    subgraph Visualization["Visualization"]
        LookerStudio[Looker Studio]
        TeamDashboard[Team Performance Dashboard]
        StatsDashboard[Game Statistics Dashboard]
    end
    
    %% Connections
    Terraform -->|Provisions| DataLake
    Terraform -->|Provisions| DataWarehouse
    Terraform -->|Provisions| Orchestration
    
    MLB --> Scrapers
    BoxScore --> Scrapers
    GameStats --> Scrapers
    
    Scrapers -->|Extract Data| Airflow
    Airflow -->|Load Raw Data| GCSRaw
    Airflow -->|Transform| GCSProcessed
    
    GCSRaw -->|Load| BQRaw
    GCSProcessed -->|Load| BQStaging
    
    BQStaging --> dbt
    dbt -->|Transform| BQAnalytics
    
    BQAnalytics --> LookerStudio
    LookerStudio --> TeamDashboard
    LookerStudio --> StatsDashboard