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
    return np.fromfile(img_path, dtype="uint8")


def run(
    triton_client,
    model_name,
    image_paths: Optional[list] = None,
    output_dir: str = "runs",
    save_txt: bool = False,
    save_image: bool = False,
    id2label: dict = None,
):
    inp_name, opt_name = "INPUT", "OUTPUT"

    for img_path in tqdm(image_paths, desc="Processing images"):
        image_data = load_image(img_path)
        image_data = np.expand_dims(image_data, axis=0)

        infer_input = InferInput(inp_name, image_data.shape, "UINT8")
        infer_input.set_data_from_numpy(image_data)

        results = triton_client.infer(
            model_name=model_name,
            inputs=[infer_input],
            outputs=[InferRequestedOutput(opt_name)],
        )

        detections = results.as_numpy(opt_name)[0]

        file_stem = os.path.splitext(os.path.basename(img_path))[0]

        if save_txt:
            output_txt_path = os.path.join(output_dir, "labels", f"{file_stem}.txt")
            with open(output_txt_path, "w") as f:
                for *xyxy, conf, cls in detections:
                    line = (*xyxy, conf, cls)
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
        if save_image:
            image = cv2.imread(img_path)
            image = draw_detection(image, detections, id2label)
            cv2.imwrite(os.path.join(output_dir, f"{file_stem}.jpg"), image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolov9_c_ensemble", help="Model name")
    parser.add_argument("--url", type=str, default="localhost:8001", help="Triton inference server URL")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose output")
    parser.add_argument("--data", type=str, help="Path to images, e.g., ./images/*.jpg")
    parser.add_argument("--output", type=str, default="runs", help="")
    parser.add_argument("--name", type=str, default="predicts", help="")
    parser.add_argument("--label-file", type=str, default="./labels.txt", help="Path to labels.txt")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-image", action="store_true")

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

    image_paths = glob.glob(f"{args.data}/*")
    if not image_paths:
        print(f"No images found in path: {args.data_path}")
        sys.exit(1)

    run(
        triton_client=triton_client,
        model_name=args.model_name,
        image_paths=image_paths,
        output_dir=output_dir,
        save_txt=args.save_txt,
        save_image=args.save_image,
        id2label=id2label,
    )