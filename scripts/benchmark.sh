#!/bin/bash
# scripts/benchmark.sh
# Benchmark Triton ensemble model using perf_analyzer in Docker container

set -e

MODEL_NAME=ensemble
TRITON_GRPC=<ip_host>:8069
OUTPUT_DIR=perf_results
NAME=perf_model_bz1

mkdir -p $OUTPUT_DIR

docker run --rm -it \
  --gpus all \
  -v $(pwd)/model_repository:/models \
  -v $(pwd)/$OUTPUT_DIR:/perf_results \
  nvcr.io/nvidia/tritonserver:25.04-py3-sdk \
  perf_analyzer \
    -m $MODEL_NAME \
    -u $TRITON_GRPC \
    -i grpc \
    -b 1 \
    --percentile=95 \
    --concurrency-range 1:8 \
    --shape INPUT:640,640,3 \
    --input-data random \
    --measurement-interval 5000 \
    --verbose-csv \
    -f /perf_results/$NAME.csv

echo "Benchmark finished. Results saved to $OUTPUT_DIR/$NAME.csv"
