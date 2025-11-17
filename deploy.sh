#!/bin/bash
# Deploy to Cloud Run

PROJECT_ID="your-gcp-project"
SERVICE_NAME="sidekick-adk-agent"
REGION="us-central1"

# Build image
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

# Push to registry
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars "ENABLE_GCP_TRACING=true,ENABLE_GCP_METRICS=true"

echo "Deployment complete!"
