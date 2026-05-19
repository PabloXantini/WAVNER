from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np

class BaseVisualizer(ABC):
    @abstractmethod
    def draw(self, image, audio_data, synth_state: dict):
        pass

class TelemetryHUD(BaseVisualizer):
    # Renders neat futuristic HUD displays, target crosshairs around hands, 
    # and connection lines indicating exact filter cutoffs and hand levels.
    # Supports horizontal mirroring coordinates.
    def draw(self, image, audio_data, synth_state: dict):
        h, w, _ = image.shape
        
        # 1. Draw Master Dashboard header
        cv.rectangle(image, (0, 0), (w, 55), (10, 10, 10), -1)
        cv.line(image, (0, 55), (w, 55), (0, 255, 255), 1)
        
        cv.putText(image, "W A V N E R  ::  S Y N T H  H U D", (30, 35), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        
        # Display active waveforms for ch1 and ch2
        ch1_wave = synth_state.get('ch1_wave', 'sine').upper()
        ch2_wave = synth_state.get('ch2_wave', 'sine').upper()
        cv.putText(image, f"CH1 WAVE: {ch1_wave} | CH2 WAVE: {ch2_wave}", (w - 450, 35), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv.LINE_AA)
        
        # Display general control info
        cv.putText(image, "KEYS: 1:SINE | 2:SQUARE | 3:TRIANGLE | 4:SAW", (30, h - 25), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv.LINE_AA)
        
        # 2. Draw Hand Telemetry overlay
        for hand in synth_state.get('hands_data', []):
            label = hand['label']
            landmarks = hand['landmarks']
            if len(landmarks) < 21:
                continue
            
            # Get wrist (0) position in pixels (horizontally mirrored!)
            wrist_x = int((1.0 - landmarks[0].x) * w)
            wrist_y = int(landmarks[0].y * h)
            
            # Neon HUD colors: Cyan for Left hand (effects/vol), Pink for Right hand (pitch/filters)
            color = (255, 255, 0) if label == 'Left' else (255, 0, 255)
            
            # Draw tracking crosshair circle
            cv.circle(image, (wrist_x, wrist_y), 35, color, 1, cv.LINE_AA)
            cv.circle(image, (wrist_x, wrist_y), 4, color, -1)
            
            # Floating label text
            cv.putText(image, f"{label.upper()} HAND", (wrist_x - 45, wrist_y - 45), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv.LINE_AA)
                       
            # Draw finger connection lines with values (horizontally mirrored!)
            if label == 'Right':
                # Lowpass pinch line (Thumb 4 to Index 8)
                t_x = int((1.0 - landmarks[4].x) * w)
                t_y = int(landmarks[4].y * h)
                i_x = int((1.0 - landmarks[8].x) * w)
                i_y = int(landmarks[8].y * h)
                
                cv.line(image, (t_x, t_y), (i_x, i_y), (0, 255, 0), 2, cv.LINE_AA)
                lp_cutoff = synth_state.get('lowpass', 20000.0)
                cv.putText(image, f"LP: {int(lp_cutoff)}Hz", (min(t_x, i_x), min(t_y, i_y) - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv.LINE_AA)
                
                # Highpass pinch line (Middle/Ring mid to Pinky 20)
                mr_x = int((1.0 - (landmarks[12].x + landmarks[16].x) / 2.0) * w)
                mr_y = int(((landmarks[12].y + landmarks[16].y) / 2.0) * h)
                p_x = int((1.0 - landmarks[20].x) * w)
                p_y = int(landmarks[20].y * h)
                
                cv.line(image, (mr_x, mr_y), (p_x, p_y), (0, 100, 255), 2, cv.LINE_AA)
                hp_cutoff = synth_state.get('highpass', 0.0)
                cv.putText(image, f"HP: {int(hp_cutoff)}Hz", (min(mr_x, p_x), min(mr_y, p_y) - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1, cv.LINE_AA)
            else:
                # Left Hand gating values and master volume
                mv = synth_state.get('master_volume', 1.0)
                cv.putText(image, f"M.Vol: {int(mv*100)}%", (wrist_x - 45, wrist_y + 55), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv.LINE_AA)
                
                ch1_gate = synth_state.get('ch1_gate', 0.0)
                ch2_gate = synth_state.get('ch2_gate', 0.0)
                cv.putText(image, f"CH1: {int(ch1_gate*100)}% | CH2: {int(ch2_gate*100)}%", (wrist_x - 65, wrist_y + 75), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1, cv.LINE_AA)

class SkeletonVisualizer(BaseVisualizer):
    # Draws a stunning, custom-designed glowing futuristic hand skeleton on the mirrored camera frame.
    # Connects key joints and outlines the hand in premium neon cyan and magenta colors.
    def draw(self, image, audio_data, synth_state: dict):
        h, w, _ = image.shape
        
        # Knuckle/finger joint connections in MediaPipe indices order
        connections = [
            # Wrist to Palm base/pinky
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index Finger
            (9, 10), (10, 11), (11, 12),         # Middle Finger
            (13, 14), (14, 15), (15, 16),        # Ring Finger
            (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
            # Palm Knuckle connectors
            (5, 9), (9, 13), (13, 17)
        ]
        
        for hand in synth_state.get('hands_data', []):
            label = hand['label']
            landmarks = hand['landmarks']
            if len(landmarks) < 21:
                continue
                
            color = (255, 255, 0) if label == 'Left' else (255, 0, 255)
            
            # 1. Draw glowing bones manually (mirrored)
            for conn in connections:
                p1, p2 = conn
                x1 = int((1.0 - landmarks[p1].x) * w)
                y1 = int(landmarks[p1].y * h)
                x2 = int((1.0 - landmarks[p2].x) * w)
                y2 = int(landmarks[p2].y * h)
                
                # Draw thick background glow
                cv.line(image, (x1, y1), (x2, y2), color, 4, cv.LINE_AA)
                # Draw sharp center line
                cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1, cv.LINE_AA)
                
            # 2. Draw glowing knuckle joint nodes
            for lm in landmarks:
                x = int((1.0 - lm.x) * w)
                y = int(lm.y * h)
                cv.circle(image, (x, y), 5, (255, 255, 255), -1, cv.LINE_AA)
                cv.circle(image, (x, y), 7, color, 1, cv.LINE_AA)

class WaveformVisualizer(BaseVisualizer):
    # Plots the real-time audio waveform as a continuous glowing cyan trace.
    def draw(self, image, audio_data, synth_state: dict):
        h, w, _ = image.shape
        if audio_data is None or len(audio_data) == 0:
            return
            
        # Downsample waveform for plotting
        points_count = min(150, len(audio_data))
        step = len(audio_data) // points_count
        
        # Center Y for the wave
        center_y = h - 100
        
        pts = []
        for i in range(points_count):
            x_pos = int((i / points_count) * w)
            val = audio_data[i * step, 0] # Left channel wave
            y_pos = int(center_y + val * 65) # Scale wave height
            pts.append([x_pos, y_pos])
            
        pts = np.array(pts, dtype=np.int32)
        
        # Draw glowing background trace
        cv.polylines(image, [pts], False, (0, 100, 255), 4, cv.LINE_AA)
        # Draw sharp foreground trace
        cv.polylines(image, [pts], False, (0, 255, 255), 1, cv.LINE_AA)

class SpectrumVisualizer(BaseVisualizer):
    # Computes a real-time FFT on the audio block and draws responsive 
    # neon equalizing spectrum bars on the bottom. Protected against empty arrays and NaNs.
    def draw(self, image, audio_data, synth_state: dict):
        h, w, _ = image.shape
        if audio_data is None or len(audio_data) == 0:
            return
            
        # Get Left channel data
        data = audio_data[:, 0]
        
        # Compute real FFT
        fft_res = np.abs(np.fft.rfft(data))
        
        # Bin into 40 bars
        num_bars = 40
        bar_width = int(w / num_bars) - 2
        
        if len(fft_res) < num_bars:
            return
            
        bins = np.array_split(fft_res, num_bars)
        
        # Smooth and scale bins
        for i, b in enumerate(bins):
            if len(b) == 0:
                continue
            amp = np.mean(b) * 15.0 # Amplify FFT bin
            if np.isnan(amp) or np.isinf(amp):
                amp = 0.0
                
            bar_height = int(min(120, amp))
            
            x_start = i * (bar_width + 2)
            y_start = h - 55 - bar_height
            
            # Draw glowing gradient bars
            # Bottom color (0, 255, 100) -> top color (0, 255, 255)
            color = (0, 255, int(100 + (bar_height / 120.0) * 155))
            cv.rectangle(image, (x_start, y_start), (x_start + bar_width, h - 55), color, -1)

class MusicReactiveShapeVisualizer(BaseVisualizer):
    # A stunning shape-based visualizer where geometric circles, glowing polygons, 
    # and expanding particle stars float and pulse directly in sync with the audio 
    # amplitude (RMS value) and current pitch frequency.
    def __init__(self):
        # Store particles: list of [x, y, vx, vy, radius, color, life]
        self.particles = []
        # Pulse angle for animations
        self.angle = 0.0

    def draw(self, image, audio_data, synth_state: dict):
        h, w, _ = image.shape
        
        # 1. Compute Audio RMS (loudness amplitude)
        if audio_data is not None and len(audio_data) > 0:
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            if np.isnan(rms) or np.isinf(rms):
                rms = 0.0
        else:
            rms = 0.0
            
        # Peak scale based on loudness
        rms_scale = min(1.0, rms * 4.0) # Scale RMS value for nice responsiveness
        
        # Pitch color mapping (using frequency to map to HSV space)
        freq = synth_state.get('ch1_freq', 440.0)
        # Map 100Hz-1200Hz to Hue 0-179
        hue = int(((freq - 100.0) / 1100.0) * 179)
        hue = max(0, min(179, hue))
        
        # Convert HSV to BGR color for CV
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0, 0]
        color_bgr = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
        
        # 2. Draw Central Music-Reactive Ring
        center_x, center_y = w // 2, h // 2
        base_radius = 80
        pulse_radius = int(base_radius + rms_scale * 120)
        
        # Create a glowing radial overlay effect
        overlay = image.copy()
        cv.circle(overlay, (center_x, center_y), pulse_radius + 20, color_bgr, 6, cv.LINE_AA)
        cv.circle(overlay, (center_x, center_y), pulse_radius, (255, 255, 255), 2, cv.LINE_AA)
        
        # Add beautiful inner spinning polygon
        self.angle = (self.angle + 0.05 + rms_scale * 0.1) % (2.0 * np.pi)
        num_vertices = 6
        poly_pts = []
        inner_r = int(pulse_radius * 0.65)
        for i in range(num_vertices):
            ang = self.angle + i * (2.0 * np.pi / num_vertices)
            px = int(center_x + inner_r * np.cos(ang))
            py = int(center_y + inner_r * np.sin(ang))
            poly_pts.append([px, py])
        
        cv.polylines(overlay, [np.array(poly_pts, dtype=np.int32)], True, color_bgr, 2, cv.LINE_AA)
        
        # Alpha blend overlay for a beautiful high-end glowing look
        cv.addWeighted(overlay, 0.45, image, 0.55, 0, image)
        
        # 3. Dynamic Particle System
        # Emit new particles if music is active/playing
        if rms_scale > 0.08 and len(self.particles) < 80:
            # Emit particles from center outward
            angle_part = np.random.uniform(0, 2.0 * np.pi)
            speed = (1.0 + rms_scale * 15.0) * np.random.uniform(0.5, 1.5)
            vx = speed * np.cos(angle_part)
            vy = speed * np.sin(angle_part)
            r = int(2 + np.random.uniform(1, 5) * rms_scale * 2.0)
            
            self.particles.append({
                'x': center_x,
                'y': center_y,
                'vx': vx,
                'vy': vy,
                'r': r,
                'color': color_bgr,
                'life': 1.0  # Life starts at 1.0 (100%)
            })
            
        # Update and draw particles
        alive_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 0.02 # Fade out life
            
            # Slowly decay radius
            radius = int(p['r'] * p['life'])
            
            # Check boundaries and life
            if p['life'] > 0 and 0 <= p['x'] < w and 0 <= p['y'] < h and radius > 0:
                cv.circle(image, (int(p['x']), int(p['y'])), radius, p['color'], -1, cv.LINE_AA)
                alive_particles.append(p)
                
        self.particles = alive_particles
        
        # 4. If in Debug mode, overlay floating metrics on the visualizer canvas
        if synth_state.get('debug', False):
            # Overlay simple clean high-end metrics directly on the visualizer window
            # Box overlay
            cv.rectangle(image, (20, 20), (280, 200), (10, 10, 10), -1)
            cv.rectangle(image, (20, 20), (280, 200), color_bgr, 1)
            
            cv.putText(image, "METRICS (DEBUG)", (35, 45), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
            cv.line(image, (25, 52), (275, 52), (100, 100, 100), 1)
            
            lp = synth_state.get('lowpass', 20000.0)
            hp = synth_state.get('highpass', 0.0)
            freq = synth_state.get('ch1_freq', 440.0)
            vol = synth_state.get('master_volume', 1.0)
            
            cv.putText(image, f"Pitch Freq: {int(freq)} Hz", (35, 75), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)
            cv.line(image, (35, 83), (265, 83), (40, 40, 40), 1)
            cv.putText(image, f"Lowpass  : {int(lp)} Hz", (35, 105), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv.LINE_AA)
            cv.line(image, (35, 113), (265, 113), (40, 40, 40), 1)
            cv.putText(image, f"Highpass : {int(hp)} Hz", (35, 135), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv.LINE_AA)
            cv.line(image, (35, 143), (265, 143), (40, 40, 40), 1)
            cv.putText(image, f"M. Volume: {int(vol * 100)}%", (35, 165), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv.LINE_AA)
            cv.line(image, (35, 173), (265, 173), (40, 40, 40), 1)
            
            # Show channel gates
            ch1_gate = synth_state.get('ch1_gate', 0.0)
            ch2_gate = synth_state.get('ch2_gate', 0.0)
            cv.putText(image, f"Gate 1: {int(ch1_gate*100)}% | Gate 2: {int(ch2_gate*100)}%", (35, 190), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 150), 1, cv.LINE_AA)
