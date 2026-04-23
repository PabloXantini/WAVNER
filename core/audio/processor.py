import pyaudio
import numpy as np
import librosa
import threading
import time

class AudioChannel:
    def __init__(self, data, sample_rate, loop=False):
        """
        data: numpy array of shape (channels, samples) normalized to [-1, 1]
        """
        self.data = data
        self.sample_rate = sample_rate
        self.loop = loop
        self.position = 0
        self.is_playing = False
        self.volume = 1.0
        self._lock = threading.Lock()

    def play(self):
        with self._lock:
            self.is_playing = True

    def stop(self):
        with self._lock:
            self.is_playing = False
            self.position = 0

    def pause(self):
        with self._lock:
            self.is_playing = False

    def set_volume(self, volume):
        with self._lock:
            self.volume = max(0.0, min(1.0, volume))

class AudioEngine:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.channels_list = []
        self._lock = threading.Lock()
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            stream_callback=self._callback
        )

    def _callback(self, in_data, frame_count, time_info, status):
        # Initialize output buffer with silence
        out_data = np.zeros((frame_count, self.channels), dtype=np.float32)
        
        with self._lock:
            active_channels = [c for c in self.channels_list if c.is_playing]
            
            for channel in active_channels:
                with channel._lock:
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
                            # If we still need frames, we could recursively call or loop here
                            # For simplicity in this chunk, we just wait for next callback
                        else:
                            channel.is_playing = False
                            channel.position = 0
                            
        # Clipping to avoid distortion
        out_data = np.clip(out_data, -1.0, 1.0)
        return (out_data.tobytes(), pyaudio.paContinue)

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
