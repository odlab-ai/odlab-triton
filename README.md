# odlab-triton

**odlab-triton** is a lightweight, modular toolkit for deploying **Computer Vision models** using [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), supporting:
- ONNX → TensorRT conversion  
- Auto-generated model configuration  
- Easy deployment via Docker  
- Perf_analyser the performance 

For detailed configuration instructions, refer to the [official Triton documentation](<https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1120/triton-inference-server-guide/docs/index.html>).

---

## Project structure
```
odlab-triton/
├── docker-compose.yml        # Compose file for Triton server deployment
├── Docker.triton             # Dockerfile for Triton server image
├── Docker.convert            # Dockerfile for ONNX → TensorRT conversion
├── model_repository/         # Auto-generated Triton model configs
├── weights/                  # Store ONNX models
├── scripts/                  # Helper scripts
├── .env                      # Environment variables
└── README.md
```

---

## Quick Start

### 1. Prepare Model  
Put your ONNX model in `weights/`:

```
weights/
└── model.onnx
```

### 2. Create the Model Repository

Configure the model ensemble pipeline and its configuration files (`config.pbtxt`), including settings such as the number of GPUs to deploy, the maximum batch size, and the input/output specifications for each step. The ensemble model includes a preprocessing model (`preprocess`) that handles image preprocessing, a postprocessing model (`postprocess`) that performs tasks such as NMS, and a TensorRT model (`inference`) that executes the main inference.


### 3. Convert ONNX to TensorRT  

Use `scripts/onnx2trt.sh` to convert ONNX to TensorRT.

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/weights:/workspace/weights \
  -v $(pwd)/model_repository:/workspace/model_repository \
  -v $(pwd)/scripts:/workspace/scripts \
  nvcr.io/nvidia/tensorrt:25.04-py3 \
  /bin/bash -c "pip install --no-cache-dir onnx && bash /workspace/scripts/onnx2trt.sh"
```

### 4. Configure Model Repository  

Setup environment
```
conda create -n odlab-env python==3.11 -y
conda activate odlab-env

pip install onnx Jinja2
```

Model configuration files in `model_repository/` are automatically generated.  
To create or update a configuration for a specific model **version** (model ID), run:
```bash
python scripts/generate_config.py weights/model.onnx model_repository/model
```


### 5. Environment Variables  

Create a `.env` file in the root:

```env
IMAGE_NAME=odlab-triton-<model-name>
IMAGE_VERSION=latest
```

### 6. Deploy Triton Inference Server  

Build and run using your `Docker.triton` and docker-compose:

```bash
docker-compose up --build
```

Triton will be available on:

| Port | Protocol |
|------|----------|
| 8068 | HTTP     |
| 8069 | gRPC     |
| 8070 | Metrics  |


### 7. Benchmark with `perf_analyzer`
Install perf_analyzer
[recommended method](https://github.com/triton-inference-server/perf_analyzer/blob/main/docs/install.md) or simply install via
`pip install perf-analyzer`
and run the performance analyzer:
```bash
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
```
---

## License  

Licensed under the MIT.  
Copyright © 2025 [odlab-ai](https://github.com/odlab-ai). All rights reserved.
