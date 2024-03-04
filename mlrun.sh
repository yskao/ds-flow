#!/bin/bash
# docker exec -it {container_name} /bin/bash where the mlruns folder is there.
docker run -it -d --privileged -p 5050:5000 mlflow-docker:latest
