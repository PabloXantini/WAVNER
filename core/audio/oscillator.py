import numpy as np
import threading

class OscillatorChannel:
    """
    A real-time synthesizer channel that generates dynamic waveforms
    and applies stateful DSP filters and effects.
    
    Designed by: PabloXantini
    """
    def __init__(self, sample_rate=44100, waveform_type='sine'):
        self.sample_rate = sample_rate
        self.waveform_type = waveform_type
        
        # Core synth parameters
        self.frequency = 440.0
        self.volume = 0.5
        self.is_playing = False
        
        # DSP States
        self.phase = 0.0
        self._lock = threading.RLock()
        
        # Filter cutoffs (Hz)
        self.lowpass_cutoff = 20000.0  # Default fully open
        self.highpass_cutoff = 0.0     # Default fully open
        self.lp_intensity = 1.0
        self.hp_intensity = 1.0
        
        # Filter States (stereo)
        self.lp_state = np.zeros(2, dtype=np.float32)
        self.hp_x_state = np.zeros(2, dtype=np.float32)
        self.hp_y_state = np.zeros(2, dtype=np.float32)
        
        # Effects parameters (0.0 to 1.0)
        self.reverb_intensity = 0.0
        self.distortion_intensity = 0.0
        
        # Reverb / Delay buffer states (1 second max delay)
        self.delay_buffer_size = sample_rate
        self.delay_buffer = np.zeros((self.delay_buffer_size, 2), dtype=np.float32)
        self.delay_ptr = 0

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

    def set_filters(self, lowpass=None, highpass=None, lp_intensity=None, hp_intensity=None):
        with self._lock:
            if lowpass is not None:
                # lowpass should go from ~50Hz to 20kHz
                self.lowpass_cutoff = max(50.0, min(20000.0, lowpass))
            if lp_intensity is not None:
                self.lp_intensity = max(0.0, min(1.0, lp_intensity))
            if highpass is not None:
                # highpass should go from 0Hz to ~5kHz
                self.highpass_cutoff = max(0.0, min(5000.0, highpass))
            if hp_intensity is not None:
                self.hp_intensity = max(0.0, min(1.0, hp_intensity))

    def set_effects(self, reverb=None, distortion=None):
        with self._lock:
            if reverb is not None:
                self.reverb_intensity = max(0.0, min(1.0, reverb))
            if distortion is not None:
                self.distortion_intensity = max(0.0, min(1.0, distortion))

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
            
            # --- 2. Distortion (Soft-Clipping tanh) ---
            if self.distortion_intensity > 0.0:
                # Map intensity to drive coefficient k: [1, 20]
                k = 1.0 + 19.0 * self.distortion_intensity
                out = np.tanh(k * out) / np.tanh(k)
                
            # --- 3. Filter Coefficients ---
            # Lowpass (exponential smoothing)
            alpha_lp = 2.0 * np.pi * self.lowpass_cutoff / self.sample_rate
            alpha_lp = min(1.0, max(0.0, alpha_lp))
            
            # Highpass (1-pole IIR)
            b1_hp = np.exp(-2.0 * np.pi * self.highpass_cutoff / self.sample_rate)
            a0_hp = (1.0 + b1_hp) / 2.0
            
            # --- 4. Reverb Settings ---
            # Let Left channel delay be ~60ms, Right channel ~85ms for wide stereo delay reverb
            D_L = int(0.060 * self.sample_rate)
            D_R = int(0.085 * self.sample_rate)
            feedback = self.reverb_intensity * 0.75  # Limit feedback to 0.75 to prevent instability
            
            # --- 5. Integrated Sample Loop (Highly Optimized) ---
            for i in range(frame_count):
                x = out[i]
                
                # Lowpass Filter
                self.lp_state = self.lp_state + alpha_lp * (x - self.lp_state)
                x_lp = self.lp_intensity * self.lp_state + (1.0 - self.lp_intensity) * x
                
                # Highpass Filter
                y_hp = a0_hp * (x_lp - self.hp_x_state) + b1_hp * self.hp_y_state
                self.hp_x_state = x_lp.copy()
                self.hp_y_state = y_hp.copy()
                x_filtered = self.hp_intensity * y_hp + (1.0 - self.hp_intensity) * x_lp
                
                # Stereo Delay Reverb
                idx_l = (self.delay_ptr - D_L) % self.delay_buffer_size
                idx_r = (self.delay_ptr - D_R) % self.delay_buffer_size
                
                delayed_l = self.delay_buffer[idx_l, 0]
                delayed_r = self.delay_buffer[idx_r, 1]
                
                out_l = x_filtered[0] + feedback * delayed_l
                out_r = x_filtered[1] + feedback * delayed_r
                
                # Save back to circular delay buffer
                self.delay_buffer[self.delay_ptr, 0] = out_l
                self.delay_buffer[self.delay_ptr, 1] = out_r
                self.delay_ptr = (self.delay_ptr + 1) % self.delay_buffer_size
                
                out[i, 0] = out_l
                out[i, 1] = out_r
                
            return out
