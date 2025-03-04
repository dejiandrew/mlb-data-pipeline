variable "project_id" {
  description = "GCP Project ID"
  default     = "mlb-pipeline-deji"  # UPDATE THIS
}

variable "region" {
  description = "GCP region"
  default     = "us-central1"  # You can change this if needed
}

variable "credentials_file" {
  description = "Path to GCP service account key"
  default     = "./terraform-key.json"  # We will generate this next
}
