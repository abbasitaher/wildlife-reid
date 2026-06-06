#!/usr/bin/env bash
#
# Upload a trained model checkpoint to Cloud Storage at a versioned path.
#
# Usage:
#   ./deploy/upload_model.sh [LOCAL_CHECKPOINT]
#   # defaults to artifacts/sea_turtle/checkpoints/best.pt
#
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-wildlife-498300}"
BUCKET="${BUCKET:-wildlifereidentification}"
MODEL_URI="${MODEL_URI:-gs://${BUCKET}/sea_turtle/models/v1/best.pt}"
LOCAL_CHECKPOINT="${1:-artifacts/sea_turtle/checkpoints/best.pt}"

echo "==> Uploading ${LOCAL_CHECKPOINT} -> ${MODEL_URI}"
gcloud storage cp "${LOCAL_CHECKPOINT}" "${MODEL_URI}" --project "${PROJECT_ID}"

echo "==> Done. Model available at ${MODEL_URI}"
