from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def update(self, image, hand, ctx, handedness='Right'):
        pass

class InputController(ABC):
    @abstractmethod
    def process_input(self, key: int):
        pass

class WaveformController(ABC):
    @abstractmethod
    def change_waveform(self, wave_type: str):
        pass