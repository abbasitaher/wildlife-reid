#!/usr/bin/env bash
#
# Deploy the Wildlife Re-Identification API to Google Cloud Run.
#
# Prerequisites:
#   - gcloud auth login
#   - Gallery index at gs://wildlifereidentification/sea_turtle/index/v1
#     (run ./deploy/upload_artifacts.sh first)
#
# Usage:
#   ./deploy/deploy.sh
#   # or override: PROJECT_ID=... BUCKET=... ./deploy/deploy.sh
#
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-wildlife-498300}"
PROJECT_NUMBER="${PROJECT_NUMBER:-716002528696}"
BUCKET="${BUCKET:-wildlifereidentification}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-wildlife-reid}"
SERVICE="${SERVICE:-wildlife-reid-api}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"
MODEL_VERSION="${MODEL_VERSION:-v1}"
INDEX_URI="gs://${BUCKET}/sea_turtle/index/${MODEL_VERSION}"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "==> Project: ${PROJECT_ID}  Bucket: gs://${BUCKET}"

echo "==> Enabling required services"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT_ID}"

echo "==> Ensuring Artifact Registry repository exists"
gcloud artifacts repositories describe "${REPO}" \
  --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" \
  --repository-format docker \
  --location "${REGION}" \
  --project "${PROJECT_ID}"

echo "==> Granting Cloud Run runtime read access to gs://${BUCKET}"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role roles/storage.objectViewer \
  --project "${PROJECT_ID}" \
  >/dev/null 2>&1 || true

echo "==> Building and pushing image with Cloud Build"
gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT_ID}"

echo "==> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --cpu-boost \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --set-env-vars "WILDLIFE_REID_CONFIG=configs/sea_turtle.gcp.yaml,WILDLIFE_REID_INDEX=${INDEX_URI}"

echo "==> Done. Service URL:"
gcloud run services describe "${SERVICE}" \
  --region "${REGION}" --project "${PROJECT_ID}" \
  --format "value(status.url)"
