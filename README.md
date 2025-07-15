# odlab-triton
A lightweight and modular package for research and development of Object Detection models using NVIDIA Triton Inference Server.

## Project structure
```bash
odlab-triton/
├── model_repository/     # Triton models (preprocess, inference, postprocess, ensemble)
├── weights/              # ONNX and TensorRT engine files
├── scripts/              # Conversion and helper scripts
├── assets/               # Test images
└── src/inference.py         # Triton client script for inference
```

## Setup environment

## Convert onnx to tensortRT ([more here](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html#converting-onnx-to-a-tensorrt-engine))

- Using trtexec inside a TensorRT containerc
  ```bash
  docker run --rm -it --gpus all \
    -v $(pwd)/weights:/workspace/weights \
    -v $(pwd)/model_repository:/workspace/model_repository \
    -v $(pwd)/scripts:/workspace/scripts \
    nvcr.io/nvidia/tensorrt:25.04-py3 bash
  ```

- Prepare weight models file in folder weights. After that, run scripts/convert.sh or 

  ```bash
  INPUT_PATH=${1:-/workspace/weights}
  OUTPUT_PATH=${2:-/workspace/model_repository}

  echo "🔍 Read ONNX file from: $INPUT_PATH"
  echo "📁 Save engine to: $OUTPUT_PATH"

  for onnx in "$INPUT_PATH"/*.onnx; do
    name=$(basename "$onnx" .onnx)
    outdir="$OUTPUT_PATH/${name}_trt/1"
    mkdir -p "$outdir"

    echo "🚀 Converting $onnx → $outdir/model.plan"

    INPUT_NAME=$(python3 -c "import onnx; m=onnx.load('$onnx'); print(m.graph.input[0].name)")
    echo "   ↪ Input name: $INPUT_NAME"
    echo "   ↪ Output: $outdir/model.plan"
    
    trtexec \
      --onnx="$onnx" \
      --saveEngine="$outdir/model.plan" \
      --minShapes=${INPUT_NAME}:1x3x640x640 \
      --optShapes=${INPUT_NAME}:1x3x640x640 \
      --maxShapes=${INPUT_NAME}:16x3x640x640 \
      --memPoolSize=workspace:2048
  done

  echo "✅ All model converted sucessfully."
  ```

## Start Triton Inference Server
1. Serve from local model repository:
    ```bash
    docker run --rm -it --gpus=all 
      -p8000:8000 -p8001:8001 -p8002:8002 
      -v "${PWD}\model_repository:/models" 
      -v "${PWD}:/workspace" 
      nvcr.io/nvidia/tritonserver:25.04-py3 bash
    ```
2. Inside the container:
- Install package
  ```bash
  #  Python packages (for running local client)
  pip install opencv-python numpy opencv-python
  apt-get update && apt-get install -y libgl1
  ```
- Run commandline build model

  ```tritonserver --model-repository=/models```
  > 📡 Ports:
  > - `8000`: HTTP  
  > - `8001`: gRPC  
  > - `8002`: Metrics (Prometheus)

## Run Inference via Client
- Using Python client in SDK container:
  ```bash
  #!/bin/bash

  docker run --rm -v ${PWD}:/workspace \
    nvcr.io/nvidia/tritonserver:24.04-py3-sdk \
    python /workspace/client.py \
      --image /workspace/assets/test_01.jpg \
      --url localhost:8001
  ```
  ---
  > ⚠️ Notes:
  > - Make sure `client.py` matches the model's input/output specs.  
  > - Replace `localhost` with the IP of your local Triton server (search ipconfig get `IPv4 Address`)
  ---

## 📌 Notes

- ✅ Tested on: `Windows + WSL2 + Docker + GPU (NVIDIA Container Toolkit)`
- 🧠 Model used: `YOLOv9-c` (for object detection), wrapped in Triton ensemble
- 📦 Containers:
  - `nvcr.io/nvidia/tritonserver:25.04-py3` – Triton server
  - `nvcr.io/nvidia/tritonserver:24.04-py3-sdk` – Python client SDK
  - `nvcr.io/nvidia/tensorrt:25.04-py3` – Model conversion (`trtexec`)

---