#!/bin/bash
VERSION=1                        # set version convert
INPUT_SIZE=3x640x640             # set input size (C x H x W)
MIN_BZ=1
OPT_BZ=4
MAX_BZ=16

ONNX_FILE=${1:-/workspace/weights/model.onnx} # change name onnx model
OUTPUT_PATH=${2:-/workspace/model_repository/model/$VERSION}

echo "[INFO] ONNX file: $ONNX_FILE"
echo "[INFO] Save engine to: $OUTPUT_PATH"
MODEL_NAME=$(basename "$ONNX_FILE" .onnx)
OUTDIR="$OUTPUT_PATH"
mkdir -p "$OUTDIR"
echo "[INFO] Converting $MODEL_NAME → $OUTDIR/model.plan"

# Auto get input name from ONNX
INPUT_NAME=$(python3 -c "import onnx;m=onnx.load('$ONNX_FILE');print(m.graph.input[0].name)")
echo "[INFO] Input: $INPUT_NAME"

MIN_SHAPE="${INPUT_NAME}:${MIN_BZ}x${INPUT_SIZE}"
OPT_SHAPE="${INPUT_NAME}:${OPT_BZ}x${INPUT_SIZE}"
MAX_SHAPE="${INPUT_NAME}:${MAX_BZ}x${INPUT_SIZE}"

trtexec \
  --onnx="$ONNX_FILE" \
  --saveEngine="$OUTDIR/model.plan" \
  --minShapes=$MIN_SHAPE \
  --optShapes=$OPT_SHAPE \
  --maxShapes=$MAX_SHAPE \
  --fp16

echo "Model converted sucessfully."
# chmod -R 777 "$OUTDIR"
