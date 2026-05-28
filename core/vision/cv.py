import cv2 as cv
import mediapipe as mp
from .camera import Camera

class HandDetector:
    def __init__(self, controller, max_hands:int=2, debug:bool=False):
        self.debug = debug
        self.mxh = max_hands
        self.d = mp.solutions.drawing_utils
        self.h = mp.solutions.hands
        self.m_h = self.h.Hands(min_detection_confidence=0.7, max_num_hands=self.mxh)
        self.c = controller

    def detect(self, img):
        try:
            results = self.m_h.process(img)
        except Exception as e:
            return False
            
        # Reset current frame's tracked hands accumulator
        self.c.clear_hands_data()
            
        if not results.multi_hand_landmarks:
            return False
            
        for idx, hand in enumerate(results.multi_hand_landmarks):
            # Default to Right if handedness cannot be queried
            handedness = 'Right'
            try:
                if hasattr(results, 'multi_handedness') and results.multi_handedness is not None:
                    for h_idx, h_data in enumerate(results.multi_handedness):
                        if h_idx == idx:
                            if hasattr(h_data, 'classification') and h_data.classification:
                                handedness = h_data.classification[0].label
                            break
            except Exception as e:
                pass
                
            # Controller activates
            self.c.update(image=img, hand=hand, ctx=self, handedness=handedness)
        return True

class VisionProcessor:
    def __init__(self, camera:Camera, controller, debug:bool=False, show_camera:bool=False, input_controller=None):
        self.debug = debug
        self.show_camera = show_camera
        self.cam = camera
        self.hdetector = HandDetector(controller, 2, debug)
        self.input_controller = input_controller

    def process(self):
        frame = self.cam.capture()
        if frame is None:
            return
        img_display = cv.flip(frame, 1)
        image_rgb = cv.cvtColor(img_display, cv.COLOR_BGR2RGB)
        self.hdetector.detect(image_rgb)
        
        # 3. Draw the sci-fi HUD and dynamic music visualizers
        if self.show_camera:
            self.hdetector.c.draw_visuals(img_display)
        else:
            self.hdetector.c.draw_visuals(image=None)
            
        # 4. Display the mirrored camera feed only if show_camera is enabled
        if self.show_camera:
            cv.imshow('HandTracking', img_display)
        
        # 5. Handle Keyboard Inputs (waitKey(1) must always run to refresh OpenCV visualizer windows)
        key = cv.waitKey(1) & 0xFF
        if self.input_controller:
            self.input_controller.process_input(key)
        else:
            if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                wave_map = {ord('1'): 'sine', ord('2'): 'square', ord('3'): 'triangle', ord('4'): 'saw'}
                self.hdetector.c.change_waveform(wave_map[key])
            elif key == 27:  # ESC to exit
                self.cam.work = False

    def close(self):
        cv.destroyAllWindows()