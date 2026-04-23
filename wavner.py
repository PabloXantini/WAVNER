from core.vision.camera import Camera
from core.vision.cv import VisionProcessor
from core.interfaces.cvembed import HandSynthetizer
from core.audio.processor import AudioEngine

def main():
    # Initialize components
    cam = Camera(1280, 720)
    vision = VisionProcessor(camera=cam, controller=HandSynthetizer(), debug=True)
    
    # Audio System
    audio = AudioEngine()
    # Load background music or track
    bg_music = audio.load_sound('assets/samples/1.mp3', loop=True)
    bg_music.play()
    
    # Load a secondary sound (optional)
    # sfx = audio.load_sound('assets/samples/2.mp3', loop=False)
    # sfx.play()

    audio.start()
    
    print("Application running. Press Ctrl+C or close window to exit.")
    
    running = True
    try:
        while running:
            vision.process()
            
            # Here you can trigger SFX based on vision logic
            # if vision.some_event:
            #     sfx.play()

            if not cam.work:
                running = False
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        vision.close()
        cam.end()
        audio.close()

if __name__=='__main__':
    main()