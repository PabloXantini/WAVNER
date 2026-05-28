import pyaudio
import numpy as np
import librosa
import threading
import time
from .dsp import DSPChannel

class AudioChannel(DSPChannel):
    def __init__(self, data, sample_rate, loop=False, base_frequency=440.0):
        """
        data: numpy array of shape (channels, samples) normalized to [-1, 1]
        base_frequency: the pitch the sample is recorded at (Hz). Changing
                        frequency via set_frequency() shifts pitch relative to this.
        """
        self.data = data
        self.sample_rate = sample_rate
        self.loop = loop
        self.is_playing = False
        self.volume = 1.0
        self._lock = threading.RLock()

        # Fractional playback position and pitch control
        self.position = 0.0          # Float — advances by playback_rate per output sample
        self.base_frequency = base_frequency
        self.frequency = base_frequency
        self.playback_rate = 1.0     # frequency / base_frequency

        # Initialize shared DSP state (filters, reverb, distortion)
        self._init_dsp(sample_rate)

    def play(self):
        with self._lock:
            self.is_playing = True

    def stop(self):
        with self._lock:
            self.is_playing = False
            self.position = 0.0

    def pause(self):
        with self._lock:
            self.is_playing = False

    def set_volume(self, volume):
        with self._lock:
            self.volume = max(0.0, min(1.0, volume))

    def set_frequency(self, frequency):
        """Shift pitch by changing the playback rate relative to base_frequency."""
        with self._lock:
            self.frequency = max(20.0, min(20000.0, frequency))
            self.playback_rate = self.frequency / self.base_frequency

    def generate_audio(self, frame_count):
        """
        Returns the next (frame_count, 2) float32 block from the sample buffer
        with pitch-shifted playback (via fractional position + linear interpolation)
        and the full DSP chain applied.
        Called by AudioEngine._callback — must NOT acquire self._lock (already held).
        """
        if not self.is_playing:
            return np.zeros((frame_count, 2), dtype=np.float32)

        data_len = self.data.shape[1]

        # Compute fractional read positions for each output sample
        positions = self.position + np.arange(frame_count, dtype=np.float64) * self.playback_rate

        if self.loop:
            positions = positions % data_len
        else:
            # Stop playback when past end
            past_end = positions >= data_len
            if past_end[0]:
                self.is_playing = False
                self.position = 0.0
                return np.zeros((frame_count, 2), dtype=np.float32)
            positions = np.minimum(positions, data_len - 1.0)

        # Linear interpolation between adjacent samples
        idx0 = positions.astype(np.int64)
        idx1 = np.minimum(idx0 + 1, data_len - 1)
        frac = (positions - idx0).astype(np.float32)

        out = np.empty((frame_count, 2), dtype=np.float32)
        n_ch = min(2, self.data.shape[0])
        for ch in range(n_ch):
            out[:, ch] = self.data[ch, idx0] * (1.0 - frac) + self.data[ch, idx1] * frac
        if n_ch < 2:
            out[:, 1] = out[:, 0]  # Upmix mono to stereo

        # Advance fractional position
        new_pos = self.position + frame_count * self.playback_rate
        if self.loop:
            self.position = new_pos % data_len
        elif new_pos >= data_len:
            self.is_playing = False
            self.position = 0.0
        else:
            self.position = new_pos

        return self._apply_dsp(out, frame_count)


class AudioEngine:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.channels_list = []
        self._lock = threading.RLock()
        
        # Buffer to keep the latest mixed audio block for visualizer usage
        self.latest_block = np.zeros((512, self.channels), dtype=np.float32)
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            stream_callback=self._callback
        )

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            # Initialize output buffer with silence
            out_data = np.zeros((frame_count, self.channels), dtype=np.float32)
            
            with self._lock:
                active_channels = [c for c in self.channels_list if c.is_playing]
                
                for channel in active_channels:
                    with channel._lock:
                        if callable(getattr(channel, 'generate_audio', None)):
                            chunk = channel.generate_audio(frame_count)
                            out_data += chunk * channel.volume
                        else:
                            # Fallback: raw buffer read for channels without generate_audio (no DSP)
                            remaining = channel.data.shape[1] - channel.position
                            take = min(frame_count, remaining)

                            # Mix in the data
                            chunk = channel.data[:, channel.position : channel.position + take].T
                            out_data[:take] += chunk * channel.volume

                            channel.position += take

                            # Handle looping or stopping
                            if channel.position >= channel.data.shape[1]:
                                if channel.loop:
                                    channel.position = 0
                                else:
                                    channel.is_playing = False
                                    channel.position = 0
                                    
            # Clipping to avoid distortion
            out_data = np.clip(out_data, -1.0, 1.0)
            self.latest_block = out_data.copy()
            return (out_data.tobytes(), pyaudio.paContinue)
        except Exception as e:
            import traceback
            print(f"[AUDIO ENGINE CALLBACK EXCEPTION]: {e}")
            traceback.print_exc()
            # Return silence instead of crashing the stream callback
            silence = np.zeros((frame_count, self.channels), dtype=np.float32)
            return (silence.tobytes(), pyaudio.paContinue)

    def load_sound(self, file_path, loop=False):
        """Loads a sound file using librosa and returns an AudioChannel."""
        print(f"Loading {file_path}...")
        # Load as stereo (mono=False) and resample to engine rate
        data, sr = librosa.load(file_path, sr=self.sample_rate, mono=False)
        
        # Ensure it's 2D (channels, samples)
        if data.ndim == 1:
            data = np.tile(data, (2, 1))
            
        channel = AudioChannel(data, self.sample_rate, loop=loop)
        with self._lock:
            self.channels_list.append(channel)
        return channel

    def start(self):
        if self.stream.is_stopped():
            self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
