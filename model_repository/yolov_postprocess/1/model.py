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
            pred = pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy()  # [B, 84, 8400]
            ratio = pb_utils.get_input_tensor_by_name(request, "INPUT_1").as_numpy()  # [B, 1]
            dwdh = pb_utils.get_input_tensor_by_name(request, "INPUT_2").as_numpy()   # [B, 2]
            original_shape = pb_utils.get_input_tensor_by_name(request, "INPUT_3").as_numpy()  # [B, 2]

            results = []
            for i in range(pred.shape[0]):
                result = self.postprocess(
                    pred[i],  # shape: [84, 8400]
                    original_shape[i],
                    ratio[i][0],
                    dwdh[i],
                    det_thres=0.25,
                    iou_thres=0.45,
                )
                results.append(result)

            # pad to max_detections in batch
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

    def postprocess(self, pred, image_shape, ratio, dwdh, det_thres=0.25, iou_thres=0.45):
        """
        pred: [84, 8400]
        image_shape: [H, W]
        ratio: float
        dwdh: [dw, dh]
        Returns: [[x1, y1, x2, y2, score, cls], ...]
        """
        pred = pred.transpose(1, 0)  # [8400, 84]
        boxes = pred[:, :4]  # xywh
        scores = pred[:, 4:]  # class scores (80)

        class_ids = np.argmax(scores, axis=1)
        class_scores = np.max(scores, axis=1)
        conf = class_scores  # assuming obj_conf is included in class_scores

        keep = conf > det_thres
        boxes = boxes[keep]
        scores = conf[keep]
        class_ids = class_ids[keep]

        if boxes.shape[0] == 0:
            return []

        # Convert xywh to xyxy
        xy = boxes[:, :2]
        wh = boxes[:, 2:]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

        # Scale back to original image size
        padding = np.array([dwdh[0], dwdh[1], dwdh[0], dwdh[1]])
        xyxy = ((xyxy - padding) / ratio).round()
        self.clip_coords(xyxy, image_shape)

        # Apply NMS
        keep_indices = self.nms_numpy(xyxy, scores, iou_thres)

        final = []
        for idx in keep_indices:
            final.append(np.concatenate([xyxy[idx], [scores[idx], class_ids[idx]]]))

        return final

    def clip_coords(self, boxes, img_shape):
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2

    def nms_numpy(self, boxes, scores, iou_threshold):
        """
        Pure NumPy NMS for deployment (no PyTorch dependency)
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep
