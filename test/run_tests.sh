#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Ensuring dependencies are installed..."
pip install -q -r requirements.txt pytest

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading .env from $PROJECT_ROOT/.env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "Running unit and integration tests..."
pytest -v test/test_units.py test/test_integration.py

if [ -z "$PINECONE_API_KEY" ]; then
    echo "Error: PINECONE_API_KEY not set. System tests will fail."
    exit 1
fi
echo "Running system tests..."
pytest -v test/test_system.py
