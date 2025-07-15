#!/bin/bash
set -e

IMAGE="tritonserver-custom:25.04-py3-sdk"
CLIENT_SCRIPT="/workspace/src/inference.py"
TRITON_URL="172.17.0.1:8001"
MODEL_NAME="${1:-yolov_ensemble}"
IMAGE_PATH="${2:-assets/test_01.jpg}"
IMAGE_DIR="${2:-assets}"
LABEL_PATH="${4:-src/labels.txt}"

echo "🔍 Checking for Docker image: $IMAGE"
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "📦 Image not found locally. Pulling now..."
  docker pull "$IMAGE"
else
  echo "✅ Image already exists locally."
fi

echo "🚀 Running Triton client inference..."

docker run --rm -v "${PWD}:/workspace" \
  "$IMAGE" \
  bash -c "
    pip install opencv-python-headless numpy > /dev/null &&
    python $CLIENT_SCRIPT \
      --model_name $MODEL_NAME \
      --data $IMAGE_DIR \
      --url $TRITON_URL \
      --label-file $LABEL_PATH \
      --save-image 
  "

# Lấy hostname
# hostname -I
#CLIENT_SCRIPT="/workspace/src/client.py"
#CLIENT_SCRIPT="/workspace/test/client_direct_trt.py"