#!/bin/bash
set -e

INPUT_PATH=${1:-/workspace/weights}
OUTPUT_PATH=${2:-/workspace/model_repository}

echo "🔍 Read ONNX file from: $INPUT_PATH"
echo "📁 Save engine to: $OUTPUT_PATH"

for onnx in "$INPUT_PATH"/*.onnx; do
  name=$(basename "$onnx" .onnx)
  outdir="$OUTPUT_PATH/${name}_trt/1"
  mkdir -p "$outdir"

  echo "🚀 Converting $onnx → $outdir/model.trt"

  INPUT_NAME=$(python3 -c "import onnx; m=onnx.load('$onnx'); print(m.graph.input[0].name)")
  OUTPUT_NAME=$(python3 -c "import onnx; m=onnx.load('$onnx'); print(m.graph.output[0].name)")
  echo "   ↪ Input name: $INPUT_NAME"
  echo "   ↪ Output: $outdir/model.trt"
  python3 src/generate_config.py "$onnx" "$OUTPUT_PATH/${name}_trt"
  
  trtexec \
    --onnx="$onnx" \
    --saveEngine="$outdir/model.plan" \
    --minShapes=${INPUT_NAME}:1x3x640x640 \
    --optShapes=${INPUT_NAME}:1x3x640x640 \
    --maxShapes=${INPUT_NAME}:16x3x640x640 \
    --fp16 \
    --memPoolSize=workspace:1024
done

echo "✅ All model converted sucessfully."


# apt-get update && apt-get install -y libgl1
# chuyển quyền lại bình thường
# sudo chown -R tony:tony ./model_repository