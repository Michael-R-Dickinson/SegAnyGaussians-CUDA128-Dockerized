"""YOLO inference for the SAGA GUI viewpoint overlay.

Loads an Ultralytics YOLO checkpoint once and runs detection on a single
viewpoint frame, returning the per-box class labels, confidence scores and
bounding boxes for the GUI to draw.
"""
import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, weights="yolo26l.pt", device="cuda"):
        self.model = YOLO(weights)        # loaded once
        self.device = device
        self.names = self.model.names     # {id: class_name}

    def predict(self, frame_rgb, conf=0.25):
        """Run detection on an (H, W, 3) uint8 RGB frame.

        Returns (labels, scores, boxes):
            labels: list[str]               per-box class names
            scores: np.ndarray (N,)         confidence in [0, 1]
            boxes:  np.ndarray (N, 4)        pixel xyxy in frame coords
        """
        # Ultralytics treats numpy sources as BGR; our buffer is RGB -> flip.
        frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        r = self.model.predict(frame_bgr, device=self.device,
                               conf=conf, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()                # (N, 4) pixel xyxy
        scores = r.boxes.conf.cpu().numpy()               # (N,)
        labels = [self.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
        return labels, scores, boxes
