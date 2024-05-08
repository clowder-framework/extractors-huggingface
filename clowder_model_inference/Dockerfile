## USAGE:
# 1. Build
# docker build -t kastanday/huggingface-single-file-extractor .
# 2. Run (with clowder in full development mode.)
# docker run -t -i --rm --net clowder2_clowder2 \
#   -e CLOWDER_URL=http://host.docker.internal:80/ \
#   -e "CLOWDER_VERSION=2" \
#   -e RABBITMQ_URI="amqp://guest:guest@clowder2-rabbitmq-1:5672/%2F" \
#   --name clowder-file-finetuning \
#   mohanar/clowder-file-finetuning

# Start with a Python 3.9 slim base image
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /extractor

# Install necessary system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed for pip install
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the scripts needed at runtime
COPY clowder_inference.py clowder_model_inference_extractor.py extractor_info.json ./

# Use a second stage to keep only what's necessary
FROM python:3.9-slim

WORKDIR /extractor

# Copy installed Python packages and scripts from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /extractor /extractor

# Environment variables
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Run the Python script
CMD ["python3", "-u", "clowder_model_inference_extractor.py", "--max-retry", "1", "--heartbeat", "5", "--connector", "RabbitMQ"]
