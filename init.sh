#!/bin/bash

MOUNT_DOWNLOAD_FOLDER=/mlruns
GCS_BUCKET=ml-project-hlh
# Ensure you understand the security implications of the next line before uncommenting
# chmod -R 777 ${MOUNT_DOWNLOAD_FOLDER}

mkdir -p ${MOUNT_DOWNLOAD_FOLDER}

# Get the first day of the current month
DATE=$(date -d "$(date +'%Y-%m-01')" '+%Y%m%d')
echo "Syncing for date: $DATE"

# Sync the GCS bucket with the local directory
gsutil rsync -d -r gs://${GCS_BUCKET}/mlruns/ ${MOUNT_DOWNLOAD_FOLDER}

# Start the MLflow UI in the background and log output
nohup mlflow ui --host 0.0.0.0 --backend-store-uri ${MOUNT_DOWNLOAD_FOLDER}/${DATE} >> mlflow.log
