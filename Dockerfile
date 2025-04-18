# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Copy only the requirements file first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Upgrade pip and install requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY --chown=appuser:appuser . .

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Create data directory
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8002

# Use the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
