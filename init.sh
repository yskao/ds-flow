#!/bin/bash

MOUNT_DOWNLOAD_FOLDER=/mlruns
GCS_BUCKET=ml-project-hlh

mkdir -p ${MOUNT_DOWNLOAD_FOLDER}
chmod -R 777 ${MOUNT_DOWNLOAD_FOLDER}

nohup mlflow ui --host 0.0.0.0 >> mlflow.log &

while [ 1 ]
do
    gsutil rsync -d -r gs://ml-project-hlh/mlruns/ /mlruns
    sleep 1
done
