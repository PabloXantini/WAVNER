from core.vision.controller import InputController, WaveformController
from core.audio.oscillator import OscillatorChannel

class KeyboardController(InputController):
    """
    KeyboardController handles user inputs and routes them to the appropriate
    systems: camera exit, and staged waveform selection for the main instrument.

    Keys 1-4 select a waveform on a local staging oscillator (preview only).
    Enter commits the staged waveform to the main synth instrument via the handler.
    ESC exits the application.
    """
    def __init__(self, handler: WaveformController, camera, sample_rate: int = 44100):
        self.handler = handler
        self.camera = camera
        # Staging oscillator — isolated preview of the pending waveform
        self.oscillator = OscillatorChannel(sample_rate=sample_rate, waveform_type='sine')
        self._staged_wave = 'sine'
        self.wave_map = {
            ord('1'): 'sine',
            ord('2'): 'square',
            ord('3'): 'triangle',
            ord('4'): 'saw'
        }

    def process_input(self, key: int):
        if key in self.wave_map:
            # Stage the waveform on the local oscillator (no commit yet)
            self._staged_wave = self.wave_map[key]
            self.oscillator.set_waveform_type(self._staged_wave)
            print(f"[KEYBOARD] Staged waveform: {self._staged_wave} — press Enter to apply")
        elif key == 13:  # Enter — commit staged waveform to the main instrument
            self.handler.change_waveform(self._staged_wave)
            print(f"[KEYBOARD] Applied waveform '{self._staged_wave}' to instrument")
        elif key == 27:  # ESC to exit
            if self.camera:
                self.camera.work = False

