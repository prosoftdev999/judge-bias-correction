FROM python:3.13-slim-bookworm

# Pre-install everything the verifier needs -- test.sh must never apt-get /
# curl / pip install at verify time
RUN pip install --no-cache-dir numpy==2.5.2 scipy==1.18.0 pytest==9.1.1 pytest-json-ctrf==0.5.2

# Bake tests into the image -- harbor does NOT upload tests/ at verify time
COPY . /tests/

# Pre-create the parent dir of every declared artifact, or harbor's upload
# fails with "Could not find the file /app/output.json in container"
RUN mkdir -p /app

WORKDIR /tests
