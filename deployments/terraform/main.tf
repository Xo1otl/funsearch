terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "cloud_run_api" {
  service = "run.googleapis.com"
}

resource "google_project_service" "artifact_registry_api" {
  service = "artifactregistry.googleapis.com"
}

# Artifact Registry repository
resource "google_artifact_registry_repository" "funsearch" {
  repository_id = "funsearch"
  location      = var.region
  format        = "DOCKER"
  description   = "FunSearch Docker repository"

  depends_on = [google_project_service.artifact_registry_api]
}

# Cloud Run service
resource "google_cloud_run_service" "funsearch" {
  name     = var.service_name
  location = var.region

  template {
    metadata {
      annotations = {
        "run.googleapis.com/ingress" = "all"
      }
    }
    
    spec {
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/funsearch/funsearch:latest"
        
        ports {
          container_port = 7860
        }

        env {
          name  = "GOOGLE_CLOUD_API_KEY"
          value = var.google_cloud_api_key
        }

      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.cloud_run_api]
}

