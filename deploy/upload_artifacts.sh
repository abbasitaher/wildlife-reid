#!/usr/bin/env bash
#
# Build the FAISS gallery index and upload to Cloud Storage.
#
# Usage:
#   ./deploy/upload_artifacts.sh
#
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-wildlife-498300}"
BUCKET="${BUCKET:-wildlifereidentification}"
REGION="${REGION:-us-central1}"
MODEL_VERSION="${MODEL_VERSION:-v1}"
INDEX_URI="gs://${BUCKET}/sea_turtle/index/${MODEL_VERSION}"

echo "==> Project: ${PROJECT_ID}  Index: ${INDEX_URI}"

echo "==> Verifying bucket gs://${BUCKET}"
gcloud storage buckets describe "gs://${BUCKET}" --project "${PROJECT_ID}"

echo "==> Building index and uploading to ${INDEX_URI}"
python scripts/build_index.py --config configs/sea_turtle.gcp.yaml --output "${INDEX_URI}"

echo "==> Uploaded artifacts:"
gcloud storage ls "${INDEX_URI}/"
