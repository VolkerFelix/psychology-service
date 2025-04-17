#!/bin/bash
set -e

# Wait for the database to be ready
echo "Waiting for database to be ready..."
while ! pg_isready -h postgres -p 5432 -U postgres; do
  sleep 1
done

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the application
echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8002
