FROM ghcr.io/mlflow/mlflow

USER root

COPY init.sh /
COPY cloud-run-secret.json /

RUN apt-get update
RUN apt-get install curl apt-transport-https ca-certificates gnupg -y
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && apt-get install google-cloud-cli -y

RUN gcloud auth activate-service-account --key-file /cloud-run-secret.json
RUN gcloud config set project data-warehouse-369301
RUN export GOOGLE_APPLICATION_CREDENTIALS=/cloud-run-secret.json

RUN apt-get update -y && apt-get install -y tini lsb-release
RUN GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`; \
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -s
RUN apt-get update -y
RUN apt-get install -y fuse gcsfuse && apt-get clean

CMD ["sh", "init.sh"]
