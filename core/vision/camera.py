import cv2 as cv

class Camera:
    def __init__(self, w:int, h:int):
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
    def capture(self):
        self.work, frame  = self.cap.read()
        return frame
    def end(self):
        self.cap.release()