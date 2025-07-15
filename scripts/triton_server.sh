#!/bin/bash
set -e

IMAGE="triton-custom:25.04"

echo "🔍 Checking for Docker image: $IMAGE"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "📦 Image not found locally. Pulling from NVIDIA NGC..."
  docker pull "$IMAGE"
else
  echo "✅ Image already exists locally."
fi

echo "🚀 Starting Triton Inference Server container..."

docker run --rm -it --gpus=all \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "$(pwd)/model_repository:/models" \
  -v "$(pwd):/workspace" \
  "$IMAGE" bash