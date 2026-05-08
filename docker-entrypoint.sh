#!/bin/bash
set -e

echo "=== Market Signal Engine ==="
echo "Running database migrations..."
cd /app
alembic upgrade head || python -c "from market_signal_engine.database.connection import init_db; init_db(); print('init_db fallback OK')"

echo "Starting application..."
exec uvicorn run:app --host 0.0.0.0 --port 8000
