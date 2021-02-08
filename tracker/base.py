from abc import ABC, abstractmethod
import numpy as np

from multimedia.container.msv import MultiStreamVideo

class MSVTracker(ABC):
    """Tracker that can handle video file of type 'msv'"""
    def __init__(self, target_video, **kwargs):
        assert isinstance(target_video, MultiStreamVideo)
        self.video = target_video
        # Processing speed
        self.start_time = 0
        self.end_time = 0

    @property
    def fps(self):
        elapsed_time = self.end_time - self.start_time
        fps = int(1/elapsed_time) + 1e-3
        return fps

    def __len__(self):
        return len(self.video)

    def __str__(self):
        content = f"{self.__class__.__name__} on video '{self.video.path}'"
        return content

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass
