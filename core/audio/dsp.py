import numpy as np
from scipy.signal import lfilter

class DSPChannel:
    """
    Mixin providing stateful DSP(Digital Signal Processing): lowpass/highpass filters,
    soft-clip distortion, and stereo delay reverb.

    All operations are fully vectorized — no Python sample loops.

    Call _init_dsp(sample_rate) inside the subclass __init__ after
    self._lock and self.sample_rate are set.

    Designed by: PabloXantini
    """

    def _init_dsp(self, sample_rate: int):
        # Filter cutoffs (Hz)
        self.lowpass_cutoff = 20000.0  # Default fully open
        self.highpass_cutoff = 0.0     # Default fully open
        self.lp_intensity = 1.0
        self.hp_intensity = 1.0

        # Filter states (stereo, one value per channel)
        self.lp_state = np.zeros(2, dtype=np.float64)
        self.hp_x_state = np.zeros(2, dtype=np.float64)  # Last highpass input
        self.hp_y_state = np.zeros(2, dtype=np.float64)  # Last highpass output

        # Effects parameters (0.0 to 1.0)
        self.reverb_intensity = 0.0
        self.distortion_intensity = 0.0

        # Reverb / delay circular buffer (1 second max)
        self.delay_buffer_size = sample_rate
        self.delay_buffer = np.zeros((self.delay_buffer_size, 2), dtype=np.float32)
        self.delay_ptr = 0

    def set_filters(self, lowpass=None, highpass=None, lp_intensity=None, hp_intensity=None):
        with self._lock:
            if lowpass is not None:
                # Lowpass: ~50 Hz to 20 kHz
                self.lowpass_cutoff = max(50.0, min(20000.0, lowpass))
            if lp_intensity is not None:
                self.lp_intensity = max(0.0, min(1.0, lp_intensity))
            if highpass is not None:
                # Highpass: 0 Hz to ~5 kHz
                self.highpass_cutoff = max(0.0, min(5000.0, highpass))
            if hp_intensity is not None:
                self.hp_intensity = max(0.0, min(1.0, hp_intensity))

    def set_effects(self, reverb=None, distortion=None):
        with self._lock:
            if reverb is not None:
                self.reverb_intensity = max(0.0, min(1.0, reverb))
            if distortion is not None:
                self.distortion_intensity = max(0.0, min(1.0, distortion))

    def _apply_dsp(self, out: np.ndarray, frame_count: int) -> np.ndarray:
        """
        Applies the full DSP chain to a (frame_count, 2) float32 array.
        All stages are vectorized — no Python per-sample loops.
        Order: distortion → lowpass → highpass → stereo reverb.
        """
        out = out.astype(np.float64)

        # --- 1. Distortion (soft-clip tanh) ---
        if self.distortion_intensity > 0.0:
            # Map intensity to drive coefficient k: [1, 20]
            k = 1.0 + 19.0 * self.distortion_intensity
            out = np.tanh(k * out) / np.tanh(k)

        # --- 2. Lowpass filter (1-pole exponential smoothing IIR) ---
        # H(z) = alpha / (1 - (1 - alpha)*z^-1)
        alpha_lp = float(min(1.0, max(1e-6, 2.0 * np.pi * self.lowpass_cutoff / self.sample_rate)))
        b_lp = np.array([alpha_lp])
        a_lp = np.array([1.0, -(1.0 - alpha_lp)])

        x_lp = np.empty_like(out)
        for ch in range(2):
            zi = np.array([self.lp_state[ch]])
            y, zf = lfilter(b_lp, a_lp, out[:, ch], zi=zi)
            x_lp[:, ch] = y
            self.lp_state[ch] = float(zf[0])

        # Blend filtered with dry signal
        x_lp = self.lp_intensity * x_lp + (1.0 - self.lp_intensity) * out

        # --- 3. Highpass filter (1-pole IIR) ---
        # y[n] = a0*(x[n] - x[n-1]) + b1*y[n-1]
        # Transfer function: b = [a0, -a0], a = [1, -b1]
        b1_hp = float(np.exp(-2.0 * np.pi * self.highpass_cutoff / self.sample_rate))
        a0_hp = (1.0 + b1_hp) / 2.0
        b_hp = np.array([a0_hp, -a0_hp])
        a_hp = np.array([1.0, -b1_hp])

        x_filtered = np.empty_like(x_lp)
        for ch in range(2):
            # Transposed direct form II initial condition:
            # zi[0] = b[1]*x_prev - a[1]*y_prev = -a0*x_prev + b1*y_prev
            zi = np.array([-a0_hp * self.hp_x_state[ch] + b1_hp * self.hp_y_state[ch]])
            y, _ = lfilter(b_hp, a_hp, x_lp[:, ch], zi=zi)
            x_filtered[:, ch] = y
            self.hp_x_state[ch] = float(x_lp[-1, ch])
            self.hp_y_state[ch] = float(y[-1])

        # Blend filtered with dry signal
        x_filtered = self.hp_intensity * x_filtered + (1.0 - self.hp_intensity) * x_lp

        # --- 4. Stereo delay reverb (fully vectorized circular buffer) ---
        # Left delay ~60 ms, Right ~85 ms for wide stereo reverb.
        # Safe to vectorize: D_L/D_R >> frame_count in practice, so reads and writes
        # never overlap within the same frame.
        feedback = self.reverb_intensity * 0.75  # Limit to 0.75 to prevent instability
        D_L = int(0.060 * self.sample_rate)
        D_R = int(0.085 * self.sample_rate)

        if feedback > 0.0:
            i_vec = np.arange(frame_count, dtype=np.int64)

            # Read delayed samples in one shot (before any writes)
            read_idx_l = (self.delay_ptr - D_L + i_vec) % self.delay_buffer_size
            read_idx_r = (self.delay_ptr - D_R + i_vec) % self.delay_buffer_size
            delayed_l = self.delay_buffer[read_idx_l, 0]
            delayed_r = self.delay_buffer[read_idx_r, 1]

            out_l = x_filtered[:, 0] + feedback * delayed_l
            out_r = x_filtered[:, 1] + feedback * delayed_r

            # Write back to circular buffer in one shot
            write_idx = (self.delay_ptr + i_vec) % self.delay_buffer_size
            self.delay_buffer[write_idx, 0] = out_l
            self.delay_buffer[write_idx, 1] = out_r

            x_filtered[:, 0] = out_l
            x_filtered[:, 1] = out_r

        self.delay_ptr = (self.delay_ptr + frame_count) % self.delay_buffer_size

        return x_filtered.astype(np.float32)
