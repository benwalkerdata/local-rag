#!/bin/bash

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
while ! nc -z ollama 11434; do 
  sleep 1
done
echo "✅ Ollama is ready!"

# Run vectorization
python /app/rag_pipeline.py

# Start Open WebUI
exec /app/entrypoint.sh "$@"
