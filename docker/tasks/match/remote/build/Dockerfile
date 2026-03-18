ARG TARGETPLATFORM
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY shared/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

COPY shared/data/survey_io.py .
RUN mkdir -p data/datasets

COPY tasks/match/shared/pipeline.py .
COPY tasks/match/remote/build/config/*.py .
COPY tasks/match/remote/envs/gcp.env .
COPY tasks/match/remote/envs/data.env .
COPY tasks/match/remote/build/config/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]
