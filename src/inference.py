import os
import sys
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional

import tritonclient.grpc as tritongrpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput

from utils.plot import draw_detection

def load_image(img_path: str):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return image

def get_max_batch_size(triton_client, model_name, model_version=""):
    """
    Query Triton server to get the max_batch_size for the given model.
    """
    try:
        full_config = triton_client.get_model_config(
            model_name=model_name, model_version=model_version, as_json=True
        )
        config = full_config["config"]  
        max_batch_size = config.get("max_batch_size", 0)
        if max_batch_size < 1:
            print(f"Model '{model_name}' does not support batching (max_batch_size={max_batch_size})")
        return max_batch_size
    except Exception as e:
        print(f"Failed to get model config for '{model_name}': {e}")
        return 0


def run(
    triton_client,
    model_name,
    image_paths: Optional[list] = None,
    output_dir: str = "runs",
    save_txt: bool = False,
    save_image: bool = False,
    id2label: dict = None,
    batch_size: int = 1,
):
    inp_name, opt_name = "INPUT", "OUTPUT"

    max_batch_size = get_max_batch_size(triton_client, model_name)
    batch_size = min(batch_size, max_batch_size)  # Ensure batch_size <= max_batch_size
    print(f"\nUsing batch_size={batch_size} (max_batch_size={max_batch_size})")

    images = []
    valid_image_paths = []
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            img = load_image(img_path)
            images.append(img)
            valid_image_paths.append(img_path)
        except ValueError as e:
            print(e)
            continue

    if not images:
        print("No valid images to process")
        return

    for batch_idx in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
        batch_images = images[batch_idx:batch_idx + batch_size]
        batch_paths = valid_image_paths[batch_idx:batch_idx + batch_size]

        max_height = max(img.shape[0] for img in batch_images)
        max_width = max(img.shape[1] for img in batch_images)
        padded_images = []
        for img in batch_images:
            h, w = img.shape[:2]
            padded_img = np.full((max_height, max_width, 3), 128, dtype=np.uint8)
            padded_img[:h, :w, :] = img
            padded_images.append(padded_img)

        batch_data = np.stack(padded_images, axis=0)  # Shape: [batch_size, max_height, max_width, 3]
        infer_input = InferInput(inp_name, batch_data.shape, "UINT8")
        infer_input.set_data_from_numpy(batch_data)

        results = triton_client.infer(
            model_name=model_name,
            inputs=[infer_input],
            outputs=[InferRequestedOutput(opt_name)],
        )

        detections = results.as_numpy(opt_name)  # Shape: [batch_size, max_detections, 6]
        for idx, (img_path, detection) in enumerate(zip(batch_paths, detections)):
            file_stem = os.path.splitext(os.path.basename(img_path))[0]

            if save_txt:
                output_txt_path = os.path.join(output_dir, "labels", f"{file_stem}.txt")
                with open(output_txt_path, "w") as f:
                    for *xyxy, conf, cls in detection:
                        line = (*xyxy, conf, cls)
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            
            if save_image:
                image = cv2.imread(img_path)  # Reload original image for visualization
                image = draw_detection(image, detection, id2label)
                cv2.imwrite(os.path.join(output_dir, f"{file_stem}.jpg"), image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolov_deyo_ensemble", help="Model name")
    parser.add_argument("--url", type=str, default="localhost:8001", help="Triton inference server URL")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose output")
    parser.add_argument("--data", type=str, required=True, help="Path to image file or folder containing images")
    parser.add_argument("--output", type=str, default="runs", help="Output directory")
    parser.add_argument("--name", type=str, default="predicts", help="Output subdirectory name")
    parser.add_argument("--label-file", type=str, default="./labels.txt", help="Path to labels.txt")
    parser.add_argument("--save-txt", action="store_true", help="Save detection results as text files")
    parser.add_argument("--save-image", action="store_true", help="Save images with detections")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (will be capped at model's max_batch_size)")

    args = parser.parse_args()

    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=args.url, verbose=args.verbose)
    except Exception as e:
        print(f"Failed to connect to Triton server at {args.url}: {e}")
        sys.exit(1)

    id2label = {}
    if os.path.isfile(args.label_file):
        with open(args.label_file) as f:
            id2label = {idx: line.strip() for idx, line in enumerate(f)}

    output_dir = os.path.join(args.output, args.name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_txt:
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    if os.path.isfile(args.data) and args.data.lower().endswith(image_extensions):
        image_paths = [args.data]
    elif os.path.isdir(args.data):
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.data, f'*{ext}')))
            image_paths.extend(glob.glob(os.path.join(args.data, f'*{ext.upper()}')))
    else:
        print(f"Invalid input: {args.data} is neither a valid image file nor a directory")
        sys.exit(1)

    if not image_paths:
        print(f"No images found in path: {args.data}")
        sys.exit(1)

    run(
        triton_client=triton_client,
        model_name=args.model_name,
        image_paths=image_paths,
        output_dir=output_dir,
        save_txt=args.save_txt,
        save_image=args.save_image,
        id2label=id2label,
        batch_size=args.batch_size,
    )