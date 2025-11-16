#!/bin/bash

# Start the FastAPI backend server in the background
echo "Starting FastAPI backend..."
uvicorn src.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend in the foreground
# This keeps the container running
echo "Starting Streamlit frontend..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false