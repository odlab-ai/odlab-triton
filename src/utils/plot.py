import cv2
import numpy as np
from typing import List, Dict, Optional


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def draw_detection(
        image: np.ndarray,
        detections: List[List[float]],
        id2label: Optional[Dict[int, str]] = None,
        colors: Optional[List[int]] = None
) -> np.ndarray:
    """
    Draws a single detection box on the image.

    Args:
        image (np.ndarray) Input image
        detections (List[List[float]]): A list containing the bounding box coordinates and class ID.
                Each detection as [x1, y1, x2, y2, conf, class_id].
        labels (Optional[Dict[int, str]]): Optional dictionary mapping class_id to label string.
    Returns:
        image (np.ndarray): The image with the detection box drawn.
    """
    for *xyxy, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        cls_id = int(cls_id)
        label = id2label.get(cls_id, "Unknow") if id2label else f"Unknow"

        if colors is None:
            draw_color = Colors()(cls_id, True)  # Default color based on class ID
        else:
            draw_color = colors[cls_id % len(colors)] if cls_id < len(colors) else (255, 0, 0)  # Fallback color

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), draw_color, 2)

        # Draw label background
        label_text = f"{label} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 4), (x1 + text_width, y1), draw_color, -1)

        # Draw label text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return image
