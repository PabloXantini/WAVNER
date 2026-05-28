import argparse
from core.vision.camera import Camera
from core.vision.cv import VisionProcessor
from core.interfaces.cvembed import HandSynthetizer
from core.audio.processor import AudioEngine
from core.audio.syntethizer import SyntethizerController
from core.audio.oscillator import OscillatorChannel
from core.interfaces.input import KeyboardController

def main():
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description="WAVNER Interactive Synthesizer")
    parser.add_argument('--show-camera', action='store_true', help="Show webcam feed with hand tracking debug skeleton and metrics.")
    args = parser.parse_args()

    # Audio System Base
    audio = AudioEngine()
    # Synthesizer Controller (The "Glue")
    synth = SyntethizerController()
    
    # Load accompaniment sounds as background channels
    ch1 = audio.load_sound("assets/samples/s1.mp3", loop=True)
    ch2 = audio.load_sound("assets/samples/s2.mp3", loop=True)
    
    # Initialize main real-time modulated synthesizer instrument
    synth_inst = audio.load_sound("assets/samples/main.mp3", loop=True)

    synth.add_channel("ch1", ch1)
    synth.add_channel("ch2", ch2)
    synth.add_channel("synth_inst", synth_inst)
    
    # Start playback on all channels
    ch1.play()
    ch2.play()
    synth_inst.play()
    
    # Initialize dynamic gating volumes (controlled by Left Hand middle/ring and pinky extensions)
    ch1.set_volume(0.0)
    ch2.set_volume(0.0)
    synth_inst.set_volume(0.5)
    
    audio.start()

    # CV System with injected dependencies
    cam = Camera(1280, 720)
    handler = HandSynthetizer(synth_controller=synth, audio_engine=audio, debug=True)
    input_ctrl = KeyboardController(handler=handler, camera=cam, sample_rate=audio.sample_rate)
    vision = VisionProcessor(camera=cam, controller=handler, debug=True, show_camera=args.show_camera, input_controller=input_ctrl)
    
    print("Application running. Starting separate premium 'AudioVisualizer' window.")
    if args.show_camera:
        print("Camera preview window enabled ('HandTracking' window).")
    else:
        print("Camera preview window hidden. Run with '--show-camera' to view the webcam.")
    print("Use keys 1, 2, 3, 4 to switch active waveform types dynamically.")
    
    running = True
    try:
        while running:
            # All modulation happens inside vision.process -> handler.update -> synth.mix_channel
            vision.process()

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

