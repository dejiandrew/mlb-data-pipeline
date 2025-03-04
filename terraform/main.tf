provider "google" {
  project     = var.project_id
  region      = var.region
  credentials = file(var.credentials_file)  # Your local GCP authentication
}

# 1️⃣ Create GCS Bucket
resource "google_storage_bucket" "mlb_pipeline_bucket" {
  name          = "${var.project_id}-mlb-pipeline"
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = true  # Deletes bucket on `terraform destroy`
}

# 2️⃣ Create a Service Account for Airflow/dbt
resource "google_service_account" "airflow_service" {
  account_id   = "airflow-service"
  display_name = "Airflow Service Account"
}

# 3️⃣ Assign IAM Roles (Storage & BigQuery)
resource "google_project_iam_member" "gcs_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.airflow_service.email}"
}

resource "google_project_iam_member" "bigquery_admin" {
  project = var.project_id
  role    = "roles/bigquery.admin"
  member  = "serviceAccount:${google_service_account.airflow_service.email}"
}

# 4️⃣ Generate a Service Account Key (JSON file)
resource "google_service_account_key" "airflow_key" {
  service_account_id = google_service_account.airflow_service.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}

# 5️⃣ Save the Key Locally
resource "local_file" "airflow_key_json" {
  filename = "${path.module}/gcp-key.json"
  content  = base64decode(google_service_account_key.airflow_key.private_key)
}

resource "google_bigquery_dataset" "raw" {
  dataset_id  = "raw"
  project     = var.project_id
  location    = "US"
  description = "Raw data for MLB pipeline"
}

resource "google_bigquery_dataset" "staging" {
  dataset_id  = "staging"
  project     = var.project_id
  location    = "US"
  description = "Staging data for MLB pipeline"
}

resource "google_bigquery_dataset" "analytics" {
  dataset_id  = "analytics"
  project     = var.project_id
  location    = "US"
  description = "Analytics data for MLB pipeline"
}
