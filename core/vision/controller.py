from abc import ABC, abstractmethod
from .cv import HandDetector

class Controller(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def update(self, image, ctx:HandDetector, debug:bool):
        pass    