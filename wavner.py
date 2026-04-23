from core.vision.camera import Camera
from core.vision.cv import VisionProcessor
from core.interfaces.cvembed import HandSynthetizer
def main():
    cam = Camera(1280, 720)
    vision = VisionProcessor(camera=cam, controller=HandSynthetizer(), debug=True)
    running = True
    while running:
        vision.process()
        if not cam.work:
            running = False
    vision.close()
    cam.end()
if __name__=='__main__':
    main()