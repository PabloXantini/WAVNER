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
        results = self.m_h.process(img)
        if not results.multi_hand_landmarks:
            return False
        for hand in results.multi_hand_landmarks:
            # Controller activates
            self.c.update(image=img, hand=hand, ctx=self)

class VisionProcessor:
    def __init__(self, camera:Camera, controller, debug:bool=False):
        self.debug = debug
        self.cam = camera
        self.hdetector = HandDetector(controller, 2, debug)

    def process(self):
        frame = self.cam.capture()
        image = frame.copy()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Mediapipe logic
        self.hdetector.detect(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if not self.debug:
            return
        cv.imshow('HandTracking', cv.flip(image, 1))
        cv.waitKey(1)
    def close(self):
        if not self.debug:
            return
        cv.destroyWindow('Handtracking')