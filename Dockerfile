FROM python:3.11-slim-bullseye

RUN apt-get update && \
    apt-get install git curl -y

COPY install_odbc.sh ./
RUN chmod +x install_odbc.sh
RUN ./install_odbc.sh
RUN sed -i 's/MinProtocol = TLSv1\.2/MinProtocol = TLSv1/' /etc/ssl/openssl.cnf

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ./src/ app/src/

ENV PYTHONPATH="$PYTHONPATH:/app/src"
