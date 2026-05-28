from core.vision.controller import InputController, WaveformController
from core.audio.oscillator import OscillatorChannel

class KeyboardController(InputController):
    """
    KeyboardController handles user inputs and routes them to the appropriate
    systems: camera exit, staged waveform selection, and instrument swapping.

    Keys 1-4  — stage a waveform on the local oscillator (audible preview).
    Enter      — commit the staged waveform to the current active instrument.
    Space      — toggle synth_inst between sample-based and oscillator-based.
    ESC        — exit the application.
    """
    def __init__(self, handler: WaveformController, camera,
                 synth_controller, sample_inst, audio_engine,
                 sample_rate: int = 44100):
        self.handler = handler
        self.camera = camera
        self.synth = synth_controller
        self.audio = audio_engine

        # Staging oscillator — audible preview of the pending waveform
        self.oscillator = OscillatorChannel(sample_rate=sample_rate, waveform_type='sine')
        self.oscillator.set_volume(0.5)
        self._staged_wave = 'sine'

        # Keep reference to original sample instrument for toggling back
        self._sample_inst = sample_inst
        self._using_oscillator = False

        # Register oscillator with audio engine (starts silent — not playing yet)
        with self.audio._lock:
            self.audio.channels_list.append(self.oscillator)

        self.wave_map = {
            ord('1'): 'sine',
            ord('2'): 'square',
            ord('3'): 'triangle',
            ord('4'): 'saw'
        }

    def process_input(self, key: int):
        if key in self.wave_map:
            # Stage waveform and preview it via the oscillator
            self._staged_wave = self.wave_map[key]
            self.oscillator.set_waveform_type(self._staged_wave)
            if not self.oscillator.is_playing:
                self.oscillator.play()
            print(f"[KEYBOARD] Staged waveform: {self._staged_wave} — Enter to apply, Space to swap instrument")

        elif key == 13:  # Enter — commit staged waveform to current active instrument
            self.handler.change_waveform(self._staged_wave)
            print(f"[KEYBOARD] Applied waveform '{self._staged_wave}' to instrument")

        elif key == ord(' '):  # Space — toggle between sample and oscillator instrument
            current_inst = self.synth.channels.get("synth_inst")
            if self._using_oscillator:
                # Switch back to sample-based instrument
                next_inst = self._sample_inst
                self.oscillator.stop()
                self._using_oscillator = False
                print("[KEYBOARD] Instrument: Sample")
            else:
                # Switch to oscillator-based instrument
                next_inst = self.oscillator
                self.oscillator.set_waveform_type(self._staged_wave)
                self.oscillator.play()
                self._using_oscillator = True
                print(f"[KEYBOARD] Instrument: Oscillator ({self._staged_wave})")

            # Transfer state from the current instrument to the new one
            if current_inst and next_inst and current_inst is not next_inst:
                next_inst.set_volume(current_inst.volume)
                if hasattr(current_inst, 'frequency') and callable(getattr(next_inst, 'set_frequency', None)):
                    next_inst.set_frequency(current_inst.frequency)
                
                # Copy filter state
                if callable(getattr(next_inst, 'set_filters', None)):
                    next_inst.set_filters(
                        lowpass=getattr(current_inst, 'lowpass_cutoff', None),
                        highpass=getattr(current_inst, 'highpass_cutoff', None),
                        lp_intensity=getattr(current_inst, 'lp_intensity', None),
                        hp_intensity=getattr(current_inst, 'hp_intensity', None)
                    )
                
                # Copy effects state
                if callable(getattr(next_inst, 'set_effects', None)):
                    next_inst.set_effects(
                        reverb=getattr(current_inst, 'reverb_intensity', None),
                        distortion=getattr(current_inst, 'distortion_intensity', None)
                    )

            self.synth.add_channel("synth_inst", next_inst)

        elif key == 27:  # ESC to exit
            if self.camera:
                self.camera.work = False
