import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT_0"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            pred = pb_utils.get_input_tensor_by_name(request, "OUTPUT_0").as_numpy()  # [B, 300, 7]
            ratio = pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy()  # [B, 1]
            dwdh = pb_utils.get_input_tensor_by_name(request, "INPUT_1").as_numpy()   # [B, 2]
            original_shape = pb_utils.get_input_tensor_by_name(request, "INPUT_2").as_numpy()  # [B, 2]

            results = []
            for i in range(pred.shape[0]):
                result = self.postprocess(pred[i], original_shape[i], ratio[i][0], dwdh[i])
                results.append(result)

            max_detections = max(len(r) for r in results)
            output_array = np.zeros((len(results), max_detections, 6), dtype=self.output0_dtype)

            for i, dets in enumerate(results):
                if len(dets) > 0:
                    output_array[i, :len(dets)] = dets

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUTPUT_0", output_array)]
            )
            responses.append(inference_response)

        return responses

    def postprocess(self, pred, image_shape, ratio, dwdh, conf_thres=0.4):
        boxes = pred[:, :4]
        cls_scores = pred[:, 4:]

        class_ids = np.argmax(cls_scores, axis=1)
        scores = np.max(cls_scores, axis=1)

        keep = scores > conf_thres
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if boxes.shape[0] == 0:
            return []

        # xywh → xyxy
        xy = boxes[:, :2]
        wh = boxes[:, 2:]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

        # ✅ scale về ảnh gốc
        padding = np.array([dwdh[0], dwdh[1], dwdh[0], dwdh[1]])
        xyxy = ((xyxy - padding) / ratio).round()
        self.clip_coords(xyxy, image_shape)

        final = np.concatenate([xyxy, scores[:, None], class_ids[:, None]], axis=1)
        return final

    def clip_coords(self, boxes, img_shape):
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2
