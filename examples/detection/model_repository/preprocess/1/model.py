import json
import cv2
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the model with the provided arguments.
        """
        self.model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT_0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        """
        Process the incoming requests and return the preprocessed data.
        """
        output0_dtype = self.output0_dtype
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            imgs_np = in_0.as_numpy()  # Shape: [batch_size, height, width, 3]

            batch_out = []
            original_shapes, ratios, paddings = [], [], []

            # Process each image in the batch
            for img in imgs_np:  # img shape: [height, width, 3]
                if img is None or len(img.shape) != 3 or img.shape[2] != 3:
                    raise pb_utils.TritonModelException(
                        f"Invalid input image shape: {img.shape if img is not None else None}. Expected [height, width, 3]."
                    )
                
                img = self.decode(img)
                h, w = img.shape[:2]
                original_shapes.append([h, w])

                # Preprocess image (expects BGR from cv2.imread)
                img_out, ratio, dwdh = self.preprocess(img)

                batch_out.append(img_out)
                ratios.append([ratio])
                paddings.append(dwdh)

            batch_out = np.concatenate(batch_out, axis=0).astype(output0_dtype)
            ratios = np.array(ratios, dtype=np.float32)
            paddings = np.array(paddings, dtype=np.float32)
            original_shapes = np.array(original_shapes, dtype=np.int32)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT_0", batch_out),
                    pb_utils.Tensor("OUTPUT_1", ratios),
                    pb_utils.Tensor("OUTPUT_2", paddings),
                    pb_utils.Tensor("OUTPUT_3", original_shapes)
                ],
            )
            responses.append(inference_response)
        return responses

    def preprocess(self, im: np.array, new_shape=(640, 640), ftype=32, color=(114, 114, 114), scaleup=True) -> list:
        """
        Preprocess image: resize and pad while meeting stride-multiple constraints.
        """
        shape = im.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        im = im.transpose((2, 0, 1))  # [H, W, C] -> [C, H, W]
        im = np.expand_dims(im, 0)  # [C, H, W] -> [1, C, H, W]
        if ftype == 32:
            im = np.ascontiguousarray(im, dtype=np.float32)
        elif ftype == 16:
            im = np.ascontiguousarray(im, dtype=np.float16)
        else:
            raise ValueError(f"Unsupported ftype: {ftype}")
        im /= 255.0  # Normalize to [0, 1]

        return im, r, (dw, dh)
    
    def decode(self, img: np.ndarray, pad_value: int = 128) -> np.ndarray:
        """
        Auto-remove padding from image based on constant pad_value (e.g. 128).
        Works when padding is uniformly applied with a known color.
        """
        mask = ~(img == pad_value).all(axis=2)
        ys, xs = np.where(mask)
        if len(ys) == 0 or len(xs) == 0:
            return img
        top, bottom = ys.min(), ys.max()
        left, right = xs.min(), xs.max()
        return img[top:bottom+1, left:right+1]