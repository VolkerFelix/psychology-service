#!/bin/bash
set -e

# Check which database backend to use
if [ "$DATABASE_BACKEND" = "postgres" ] || [ -z "$DATABASE_BACKEND" ]; then
    echo "Using PostgreSQL database backend..."

    # Wait for PostgreSQL to be ready
    echo "Waiting for database to be ready..."
    if command -v pg_isready &> /dev/null; then
        until pg_isready -h ${DB_HOST:-postgres} -p ${DB_PORT:-5432} -U ${DB_USER:-postgres}; do
            echo "PostgreSQL is unavailable - sleeping"
            sleep 1
        done
        echo "PostgreSQL is up - continuing"
    else
        echo "pg_isready not found, skipping PostgreSQL readiness check"
        # Sleep a bit to give the database a chance to start
        sleep 5
    fi
else
    echo "Using SQLite database backend..."
    # Ensure the directory for SQLite exists
    mkdir -p $(dirname ${SQLITE_DB_PATH:-psychology_data.db})
fi

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the application
echo "Starting application..."
exec uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8002}
