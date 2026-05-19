from abc import ABC, abstractmethod
import numpy as np

class BaseGesture(ABC):
    @abstractmethod
    def recognize(self, landmarks, handedness: str) -> float:
        """
        Processes hand landmarks and returns a normalized parameter value [0.0, 1.0].
        """
        pass

class HandHeightGesture(BaseGesture):
    def __init__(self, landmark_idx=0):
        self.idx = landmark_idx

    def recognize(self, landmarks, handedness: str) -> float:
        if not isinstance(landmarks, list):
            landmarks = [lm for lm in landmarks]
        if len(landmarks) < 21 or self.idx >= len(landmarks):
            return 0.0
            
        y = landmarks[self.idx].y
        val = 1.0 - y
        return float(np.clip(val, 0.0, 1.0))

class MultiLandmarkHeightGesture(BaseGesture):
    # Measures the average height (y-coordinates) of multiple target landmarks.
    def __init__(self, landmark_indices):
        self.indices = landmark_indices if isinstance(landmark_indices, list) else [landmark_indices]

    def recognize(self, landmarks, handedness: str) -> float:
        if not isinstance(landmarks, list):
            landmarks = [lm for lm in landmarks]
        if len(landmarks) < 21:
            return 0.0
            
        y_vals = [landmarks[i].y for i in self.indices if i < len(landmarks)]
        if not y_vals:
            return 0.0
            
        avg_y = np.mean(y_vals)
        val = 1.0 - avg_y
        return float(np.clip(val, 0.0, 1.0))

class PinchGesture(BaseGesture):
    # Measures 3D Euclidean distance between groups of landmarks (e.g., finger tips).
    def __init__(self, finger1_indices, finger2_indices, min_dist=0.03, max_dist=0.18, invert=True):
        self.f1_ind = finger1_indices if isinstance(finger1_indices, list) else [finger1_indices]
        self.f2_ind = finger2_indices if isinstance(finger2_indices, list) else [finger2_indices]
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.invert = invert

    def recognize(self, landmarks, handedness: str) -> float:
        if not isinstance(landmarks, list):
            landmarks = [lm for lm in landmarks]
        if len(landmarks) < 21:
            return 0.0
            
        # Calculate centroids of the two finger groups
        p1 = np.mean([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in self.f1_ind if i < len(landmarks)], axis=0)
        p2 = np.mean([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in self.f2_ind if i < len(landmarks)], axis=0)
        
        dist = float(np.linalg.norm(p1 - p2))
        norm = (dist - self.min_dist) / (self.max_dist - self.min_dist)
        val = np.clip(norm, 0.0, 1.0)
        if self.invert:
            val = 1.0 - val
        return float(val)

class FingerExtensionGesture(BaseGesture):
    # Measures the extension of a target finger by calculating tip-to-base 3D distance
    # and normalizing by overall hand size (wrist to middle finger MCP) to handle hand scaling.
    def __init__(self, tip_idx, base_idx=0, min_val=1.2, max_val=2.5):
        self.tip_idx = tip_idx
        self.base_idx = base_idx
        self.min_val = min_val
        self.max_val = max_val

    def recognize(self, landmarks, handedness: str) -> float:
        if not isinstance(landmarks, list):
            landmarks = [lm for lm in landmarks]
        if len(landmarks) < 21:
            return 0.0
            
        # Hand scale factor: wrist (0) to middle finger MCP (9)
        p_wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        p_mcp = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
        scale = float(np.linalg.norm(p_wrist - p_mcp))
        if scale == 0.0:
            scale = 1.0
            
        p_tip = np.array([landmarks[self.tip_idx].x, landmarks[self.tip_idx].y, landmarks[self.tip_idx].z])
        p_base = np.array([landmarks[self.base_idx].x, landmarks[self.base_idx].y, landmarks[self.base_idx].z])
        
        dist = float(np.linalg.norm(p_tip - p_base)) / scale
        val = (dist - self.min_val) / (self.max_val - self.min_val)
        return float(np.clip(val, 0.0, 1.0))

class GestureRecognizer:
    """
    Aggregates modular gestures and analyzes hand landmarks to output structured synth controls.
    """
    def __init__(self):
        # Base height/pitch and volume
        self.height_gesture = HandHeightGesture(0)
        
        # Right hand lowpass: frequency controlled by index height, intensity by thumb height
        self.lp_freq = HandHeightGesture(8)
        self.lp_intensity = HandHeightGesture(4)
        
        # Right hand highpass: frequency controlled by middle/ring average, intensity by pinky height
        self.hp_freq = MultiLandmarkHeightGesture([12, 16])
        self.hp_intensity = HandHeightGesture(20)
        
        # Left hand extensions (effects and channel gating)
        self.thumb_ext = FingerExtensionGesture(tip_idx=4, base_idx=5, min_val=0.4, max_val=1.1)      # Reverb
        self.index_ext = FingerExtensionGesture(tip_idx=8, base_idx=0, min_val=1.3, max_val=2.4)      # Distortion
        self.middle_ext = FingerExtensionGesture(tip_idx=12, base_idx=0, min_val=1.3, max_val=2.5)    # Channel 1 gate part 1
        self.ring_ext = FingerExtensionGesture(tip_idx=16, base_idx=0, min_val=1.2, max_val=2.4)      # Channel 1 gate part 2
        self.pinky_ext = FingerExtensionGesture(tip_idx=20, base_idx=0, min_val=1.1, max_val=2.1)     # Channel 2 gate

    def analyze(self, landmarks, handedness: str) -> dict:
        """
        Extracts all recognized gesture parameters for a given hand.
        """
        results = {
            'height': self.height_gesture.recognize(landmarks, handedness),
        }
        
        if handedness == 'Right':
            results['lp_freq'] = self.lp_freq.recognize(landmarks, handedness)
            results['lp_intensity'] = self.lp_intensity.recognize(landmarks, handedness)
            results['hp_freq'] = self.hp_freq.recognize(landmarks, handedness)
            results['hp_intensity'] = self.hp_intensity.recognize(landmarks, handedness)
        else: # Left Hand
            results['reverb_ext'] = self.thumb_ext.recognize(landmarks, handedness)
            results['distortion_ext'] = self.index_ext.recognize(landmarks, handedness)
            
            # Channel 1 is active/gated by the average of middle and ring finger extension
            ch1_val = (self.middle_ext.recognize(landmarks, handedness) + 
                       self.ring_ext.recognize(landmarks, handedness)) / 2.0
            results['ch1_gating'] = ch1_val
            
            # Channel 2 is gated by pinky finger extension
            results['ch2_gating'] = self.pinky_ext.recognize(landmarks, handedness)
            
        return results
