from abc import ABC, abstractmethod


class BaseStream(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pointer = 0

        # Derived class need to initialize the stream value
        if self.stream is None:
            msg = f"Derived class {self.__class__.__name__} needs to initialize this stream"
            raise RuntimeError(msg)

    def __len__(self):
        return len(self.stream)

    def seek(self, idx):
        if idx >= len(self.stream):
            raise RuntimeError(f"Cannot move stream pointer to index {idx}")
        self.pointer = idx

    def read(self):
        try:
            frame = self.stream[self.pointer]
            self.pointer += 1
        except Exception as e:
            return None

        return frame

    @abstractmethod
    def close(self):
        raise RuntimeError("You didn't implement close method")

    @abstractmethod
    def save(self):
        raise RuntimeError("You didn't implement save method")
