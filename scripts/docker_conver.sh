#!/bin/bash
set -e

IMAGE="tensorrt-custom:25.04"
TAG="25.04-py3"

echo "🔍 Checking for Docker image: $IMAGE"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "📦 Image not found locally. Pulling from NVIDIA NGC..."
  docker pull "$IMAGE"
else
  echo "✅ Image already available locally."
fi

echo "🚀 Launching TensorRT container for model conversion..."
docker run --rm -it --gpus all \
  -v "$(pwd)/weights:/workspace/weights" \
  -v "$(pwd)/model_repository:/workspace/model_repository" \
  -v "$(pwd)/scripts:/workspace/scripts" \
  -v "$(pwd)/assets:/workspace/assets" \
  -v "$(pwd)/src:/workspace/src" \
  "$IMAGE" bash

# chuyển quyền lại bình thường
# sudo chown -R tony:tony ./model_repository
