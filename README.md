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

### 2️. Convert ONNX to TensorRT  

Use `Docker.convert` image/container to convert ONNX to TensorRT.


### 3️. Configure Model Repository  

Setup environment
```
conda create -n odlab-env python==3.11 -y
conda activate odlab-env

pip install onnx perf-analyzer Jinja2
```

Model configuration files in `model_repository/` are automatically generated.  
To create or update a configuration for a specific model **version** (model ID), run:
```bash
python scripts/generate_config.py weights/model.onnx model_repository/model/<version>
```


### 4️. Environment Variables  

Create a `.env` file in the root:

```env
IMAGE_NAME=odlab-triton-detection
IMAGE_VERSION=latest
```

### 5️. Deploy Triton Inference Server  

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


### 6. Benchmark with `perf_analyzer`
Install perf_analyzer
[recommended method](https://github.com/triton-inference-server/perf_analyzer/blob/main/docs/install.md) or simply install via
`pip install perf-analyzer`
and run the benchmark: 
```bash
OUTPUT_DIR=perf_results
NAME=perf_model_bz1

mkdir -p $OUTPUT_DIR

perf_analyzer \
    -m ensemble \                      # Model name (as in model_repository)
    -u 0.0.0.0:8069 \                  # Triton server address (gRPC endpoint)
    -i grpc \                          # Protocol: grpc or http
    --percentile=95 \                  # Report 95th percentile latency
    --concurrency-range 1:64 \         # Number of concurrent requests (1–64)
    --shape INPUT:640,640,3 \          # Input tensor name and shape (no batch dim)
    --batch-size 1 \                   # Batch size per inference request (<= model max_batch_size)
    --input-data random \              # Use random input data for testing
    --measurement-interval 5000 \      # Measurement interval in ms (5 seconds)
    --verbose-csv \                    # Output detailed CSV results
    -f $OUTPUT_DIR/$NAME.csv           # Path to save the performance results
```
---

## License  

Licensed under the MIT.  
Copyright © 2025 [Tien Nguyen Van](https://github.com/tien-ngnvan). All rights reserved.
