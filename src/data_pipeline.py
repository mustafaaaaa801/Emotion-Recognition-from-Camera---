import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class FaceExtractor:
    def __init__(self, min_detection_confidence=0.5):
        import mediapipe as mp
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(min_detection_confidence=min_detection_confidence)

    def extract(self, frame):
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = []
        if results.detections:
            h, w, _ = frame.shape
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                boxes.append((x1, y1, x2, y2))
        return boxes

class EmotionDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.samples = []
        self.classes = classes
        self.transform = transform
        for cls_idx, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, f), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224,224,3), dtype=np.uint8)
        if self.transform:
            img = self.transform(img)
        return img, label
