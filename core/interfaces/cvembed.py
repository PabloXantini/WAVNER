from ..vision.controller import Controller
from ..vision.gesture import GestureRecognizer
from .visualizer import (
    MusicReactiveShapeVisualizer, 
    SpectrumVisualizer, 
    WaveformVisualizer, 
    TelemetryHUD,
    SkeletonVisualizer
)
import numpy as np

class HandSynthetizer(Controller):
    # Glue controller that maps detected hand gesture landmarks to real-time 
    # synthesizer parameters and coordinates the beautiful visualizer overlays.
    # Supports a separate aesthetic visualizer window and a camera debug view.
    def __init__(self, synth_controller, audio_engine=None, debug=False):
        super().__init__()
        self.synth_controller = synth_controller
        self.audio_engine = audio_engine
        self.debug = debug
        
        # Extensible Gesture Engine
        self.gesture_recognizer = GestureRecognizer()
        
        # Tracked hands telemetry storage (cleared every camera frame)
        self.hands_data = []
        
        # Aesthetic Visualizers rendered on a separate dark canvas (Window 2)
        self.canvas_visualizers = [
            MusicReactiveShapeVisualizer(),
            SpectrumVisualizer(),
            WaveformVisualizer()
        ]
        
        # Debug/Skeleton Visualizers rendered on the camera feed (Window 1)
        self.debug_visualizers = [
            SkeletonVisualizer(),
            TelemetryHUD()
        ]
        
        # Synthesizer control states
        self.master_volume = 0.5
        self.ch1_wave = 'sine'
        self.ch2_wave = 'square' # Set default wave shapes

    def clear_hands_data(self):
        self.hands_data = []

    def update(self, image, hand, ctx, handedness='Right'):
        # Evaluates the landmarks of a detected hand and routes parameters to 
        # the active synthesis channels. Wraps core updates in a resilient 
        # try...except block to diagnose any unexpected desynchronization issues.
        try:
            if hand is None or not hasattr(hand, 'landmark'):
                return
                
            # CAST TO STANDARD PYTHON LIST to prevent protobuf desynchronization/indexing blocking errors
            landmarks_list = [lm for lm in hand.landmark]
            if len(landmarks_list) < 21:
                return
                
            # Analyze hand shapes modularly
            gestures = self.gesture_recognizer.analyze(landmarks_list, handedness)
            
            # Record telemetry for HUD visualization (saving the standard list)
            self.hands_data.append({
                'label': handedness,
                'landmarks': landmarks_list
            })
            
            # Route parameters based on hand assignment
            if handedness == 'Right':
                # --- 1. Right Hand: Pitch & Filters ---
                # Map height exponentially [100Hz, 1200Hz] for natural musical scale
                freq = 100.0 * (1200.0 / 100.0) ** gestures['height']
                self.synth_controller.set_frequency("synth_inst", freq)
                
                # Lowpass filter: index finger controls cutoff frequency, thumb controls intensity
                lp_cutoff = 50.0 + 19950.0 * gestures['lp_freq']
                lp_intensity = gestures['lp_intensity']
                self.synth_controller.set_filters("synth_inst", lowpass=lp_cutoff, lp_intensity=lp_intensity)
                
                # Highpass filter: middle/ring average controls cutoff frequency, pinky controls intensity
                hp_cutoff = 4500.0 * gestures['hp_freq']
                hp_intensity = gestures['hp_intensity']
                self.synth_controller.set_filters("synth_inst", highpass=hp_cutoff, hp_intensity=hp_intensity)
                
            else:
                # --- 2. Left Hand: Master Volume, Effects & Gating ---
                # Master volume scale is controlled by Left Hand height
                master_vol = gestures['height']
                
                # Reverb controlled by thumb extension applied to synth_inst
                self.synth_controller.set_effects("synth_inst", reverb=gestures['reverb_ext'])
                
                # Distortion drive controlled by index extension applied to synth_inst
                self.synth_controller.set_effects("synth_inst", distortion=gestures['distortion_ext'])
                
                # Gate Channel 1 and 2 (accompaniment songs) volumes scaled by master volume
                ch1_vol = gestures['ch1_gating'] * master_vol
                ch2_vol = gestures['ch2_gating'] * master_vol
                
                self.synth_controller.mix_channel("ch1", volume=ch1_vol)
                self.synth_controller.mix_channel("ch2", volume=ch2_vol)
        except Exception as e:
            import traceback
            print(f"[HAND CONTROLLER EXCEPTION]: {e}")
            traceback.print_exc()

    def change_waveform(self, wave_type):
        self.ch1_wave = wave_type
        self.ch2_wave = wave_type
        self.synth_controller.set_waveform_type("synth_inst", wave_type)

    def draw_visuals(self, image=None):
        # Queries audio data, draws the aesthetic visualizer canvas, and optionally draws
        # debug telemetry skeleton overlays on the mirrored camera frame.
        audio_data = None
        if self.audio_engine is not None:
            audio_data = self.audio_engine.latest_block
            
        # Fetch current active parameters to supply to visualizers
        lp_cutoff = 20000.0
        hp_cutoff = 0.0
        ch1_freq = 440.0
        ch1_vol = 0.0
        ch2_vol = 0.0
        
        if 'ch1' in self.synth_controller.channels:
            ch1 = self.synth_controller.channels['ch1']
            if hasattr(ch1, 'lowpass_cutoff'):
                lp_cutoff = ch1.lowpass_cutoff
            if hasattr(ch1, 'highpass_cutoff'):
                hp_cutoff = ch1.highpass_cutoff
            if hasattr(ch1, 'frequency'):
                ch1_freq = ch1.frequency
            if hasattr(ch1, 'volume'):
                ch1_vol = ch1.volume
                
        if 'ch2' in self.synth_controller.channels:
            ch2 = self.synth_controller.channels['ch2']
            if hasattr(ch2, 'volume'):
                ch2_vol = ch2.volume
                
        # Normalize individual gate percentages relative to master volume
        master_scale = self.master_volume if self.master_volume > 0 else 1.0
        
        synth_state = {
            'hands_data': self.hands_data,
            'lowpass': lp_cutoff,
            'highpass': hp_cutoff,
            'ch1_freq': ch1_freq,
            'master_volume': self.master_volume,
            'ch1_gate': min(1.0, ch1_vol / master_scale),
            'ch2_gate': min(1.0, ch2_vol / master_scale),
            'ch1_wave': self.ch1_wave,
            'ch2_wave': self.ch2_wave,
            'debug': self.debug
        }
        
        # 1. ALWAYS Draw and Show the Separate Aesthetic Visualizer Window
        vis_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for vis in self.canvas_visualizers:
            vis.draw(vis_canvas, audio_data, synth_state)
            
        import cv2 as cv
        cv.imshow('AudioVisualizer', vis_canvas)
        
        # 2. Optionally draw skeletal/HUD metrics on the camera frame if debug/camera is visible
        if image is not None:
            for vis in self.debug_visualizers:
                vis.draw(image, audio_data, synth_state)
