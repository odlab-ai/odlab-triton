import json
import cv2
import numpy as np
import triton_python_backend_utils as pb_utils



class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the model with the provided arguments.
        """
        # parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT_0"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        """
        Process the incoming requests and return the preprocessed data.
        """
        output0_dtype = self.output0_dtype
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            imgs_np = in_0.as_numpy()

            batch_out = []
            original_shapes, ratios, paddings = [], [], []

            for img_bytes in imgs_np:
                np_arr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # -> ndarray [H, W, 3] (BGR)

                if image is None:
                    raise pb_utils.TritonModelException(
                        "Failed to decode image. Please check the input format."
                    )
                
                h, w = image.shape[:2]
                original_shapes.append([h, w])

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_out, ratio, dwdh = self.preprocess(image)

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

    def preprocess(self, im:np.array, new_shape=(640, 640), ftype=32, color=(114, 114, 114), scaleup=True) -> list:
        # Resize and pad image while meeting stride-multiple constraints
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
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color
                                )  # add border

        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        if (ftype == 32):
            im = np.ascontiguousarray(im, dtype=np.float32)
        elif (ftype == 16):
            im = np.ascontiguousarray(im, dtype=np.float16)  # half precision float16
        else:
            raise ValueError(f"Unsupported ftype: {ftype}")
        im /= 255

        return im, r, (dw, dh)