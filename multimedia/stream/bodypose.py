import pickle
import numpy as np

from .base import BaseStream


class BodyposeStream(BaseStream):

    def __init__(self, src, **kwargs):
        self.src = src

        # Read stream from src path
        try:
            with open(src, 'rb') as f:
                self.stream = pickle.load(f)
        except Exception as e:
            print(e)
            self.stream = None

        super().__init__(**kwargs)

    def close(self):
        del self.stream

    def save(self):
        with open(self.src, 'wb') as f:
            pickle.dump(self.stream, f)
