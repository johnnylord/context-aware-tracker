import cv2
import numpy as np

from .base import BaseStream


class VideoStream(BaseStream):

    def __init__(self, src, **kwargs):
        self.src = src

        # Video stream
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open video stream '{src}'")

        # Metadata
        self.fps = int(self.stream.get(cv2.CAP_PROP_FPS))
        self.length = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        super().__init__()

    def __len__(self):
        return self.length

    def seek(self, idx):
        if idx >= self.length:
            raise RuntimeError(f"Cannot move stream pointer to index {idx}")
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def read(self):
        ret, frame = self.stream.read()
        if not ret:
            return None
        return frame

    def close(self):
        self.stream.release()
        del self.stream

    def save(self):
        pass
