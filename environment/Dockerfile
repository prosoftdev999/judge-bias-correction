FROM python:3.13-slim-bookworm

RUN pip install --no-cache-dir numpy==2.5.2 scipy==1.18.0

WORKDIR /app

COPY data/judge_transcripts.jsonl /app/data/judge_transcripts.jsonl
COPY data/human_calibration.jsonl /app/data/human_calibration.jsonl
COPY data/models_metadata.json /app/data/models_metadata.json
