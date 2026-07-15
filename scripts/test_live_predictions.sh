#!/usr/bin/env bash
# Sends sample tweets to the running /predict API to generate live MLflow metrics.
# Usage: bash scripts/test_live_predictions.sh
# Prereq: docker-compose up --build (mlflow UI at http://localhost:5000)

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000/predict}"

tweets=(
  "Thank you so much for the amazing flight, best crew ever!"
  "My flight got cancelled and nobody told me anything, terrible service."
  "Landed on time, nothing special either way."
  "Lost my luggage again, this is the third time this year."
  "Great legroom and friendly staff, will fly again."
)

for t in "${tweets[@]}"; do
  echo "--- $t ---"
  curl -s -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$t\"}"
  echo
  sleep 2
done
