import numpy as np
import threading
from .dsp import DSPChannel

class OscillatorChannel(DSPChannel):
    """
    A real-time synthesizer channel that generates dynamic waveforms
    and applies stateful DSP filters and effects via DSPChannel.

    Designed by: PabloXantini
    """
    def __init__(self, sample_rate=44100, waveform_type='sine'):
        self.sample_rate = sample_rate
        self.waveform_type = waveform_type

        # Core synth parameters
        self.frequency = 440.0
        self.volume = 0.5
        self.is_playing = False

        # DSP phase accumulator
        self.phase = 0.0
        self._lock = threading.RLock()

        # Initialize shared DSP state (filters, reverb, distortion)
        self._init_dsp(sample_rate)

    def play(self):
        with self._lock:
            self.is_playing = True

    def stop(self):
        with self._lock:
            self.is_playing = False
            self.phase = 0.0

    def pause(self):
        with self._lock:
            self.is_playing = False

    def set_volume(self, volume):
        with self._lock:
            self.volume = max(0.0, min(1.0, volume))

    def set_frequency(self, frequency):
        with self._lock:
            self.frequency = max(20.0, min(2000.0, frequency))

    def set_waveform_type(self, waveform_type):
        with self._lock:
            if waveform_type in ['sine', 'square', 'triangle', 'saw']:
                self.waveform_type = waveform_type

    def generate_audio(self, frame_count):
        """
        Generates the next block of audio frames.
        Returns a numpy array of shape (frame_count, 2) normalized to [-1, 1].
        """
        with self._lock:
            if not self.is_playing:
                return np.zeros((frame_count, 2), dtype=np.float32)

            # --- 1. Waveform Generation ---
            t = np.arange(frame_count)
            phase_step = self.frequency / self.sample_rate
            phases = (self.phase + t * phase_step) % 1.0
            self.phase = (self.phase + frame_count * phase_step) % 1.0

            if self.waveform_type == 'sine':
                wave = np.sin(2.0 * np.pi * phases)
            elif self.waveform_type == 'square':
                wave = np.sign(np.sin(2.0 * np.pi * phases))
            elif self.waveform_type == 'triangle':
                wave = 2.0 * np.abs(2.0 * (phases - np.floor(phases + 0.5))) - 1.0
            elif self.waveform_type == 'saw':
                wave = 2.0 * (phases - np.floor(phases + 0.5))
            else:
                wave = np.zeros(frame_count, dtype=np.float32)

            # Create stereo mix
            out = np.column_stack([wave, wave]).astype(np.float32)

            # --- 2. Apply shared DSP chain (distortion → filters → reverb) ---
            return self._apply_dsp(out, frame_count)
