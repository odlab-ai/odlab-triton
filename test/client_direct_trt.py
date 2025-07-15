import cv2
import numpy as np
import tritonclient.grpc as tritongrpcclient

def preprocess(image_path, size=(640, 640)):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Không thể đọc ảnh: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # → NCHW
    return img

# --- Config ---
image_path = "assets/test_01.jpg"
model_name = "yolov9-c_trt"
triton_url = "172.17.0.1:8001"  # Sửa lại nếu cần

# --- Step 1: Kết nối Triton ---
triton_client = tritongrpcclient.InferenceServerClient(url=triton_url, verbose=False)
print("✅ Connected to Triton Server")

# --- Step 2: Preprocess ảnh ---
preprocessed_img = preprocess(image_path)

# --- Step 3: Chuẩn bị input/output ---
inputs = [tritongrpcclient.InferInput("images", preprocessed_img.shape, "FP32")]
inputs[0].set_data_from_numpy(preprocessed_img)

outputs = [tritongrpcclient.InferRequestedOutput("output0")]

# --- Step 4: Gửi yêu cầu infer ---
results = triton_client.infer(
    model_name=model_name,
    inputs=inputs,
    outputs=outputs
)

# --- Step 5: In kết quả ---
output = results.as_numpy("output0")
print("📦 Output shape:", output.shape)
print("Output values:\n", output)
print("🔍 Output sample (first 5 boxes):\n", output[:5])