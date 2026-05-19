# Parts of must consider in the project
## Libraries
- librosa
- pyaudio
- numpy
- mediapipe
- opencv-python

## Classes
- GestureRecognizer Class

## Modules
- **__Modulator__**: Modulation of instrumentsignals (in oscillator.py)
    **__WaveformTypes__**
        - sine
        - square
        - triangle
        - saw
    **__Tone__**
        - tone: RightHand
    **__Volume__**
        - vol: LeftHand
    **__Filter__**
        - lowpass (offset, intensity) RightHand: [thumb, index]
        - highpass (offset, intensity) RightHand: [(middle, ring), pinky]
    **__Effects__**
        - reverb (intensity) LeftHand: [thumb]
        - distortion (intensity) LeftHand: [index]
    **__Channels__**    
        - channel 1 LeftHand [(middle, ring)]
        - channel 2 LeftHand [pinky]
        
### Done mostly (due to refactor)
- **__Syntethizer__**: Mix audio channel and play it. (syntethizer.py)
- **__AudioEngine__**: Audio player (in processor.py)
- **__Camera__**: Camera handler (in camera.py)
- **__HandDetector__**: Hand detector (in cv.py)
- **__VisionProcessor__**: Vision processor (in cv.py) 
- **__Controller__**: Controller for hand gestures (in controller.py)
- **__UIInterfaces__**: Immediate UI controllers for debugging (in cvembed.py & etc)

### ToDo
- **__AudioAestheticVisualizer__**: Audio atractive visualizer.

## Features

### "Theremin Like" instrument