from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def update(self, image, hand, ctx, handedness='Right'):
        pass