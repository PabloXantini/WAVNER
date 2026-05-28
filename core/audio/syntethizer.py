class SyntethizerController:
    def __init__(self):
        # Dictionary to store named channels: { "ch1": AudioChannelObject }
        self.channels = {}

    def add_channel(self, name, channel):
        self.channels[name] = channel

    def mix_channel(self, name, volume=None, pitch=None):
        if name in self.channels:
            ch = self.channels[name]
            if volume is not None:
                ch.set_volume(volume)
            if pitch is not None and callable(getattr(ch, 'set_frequency', None)):
                ch.set_frequency(pitch)
        else:
            # Silently ignore or print warning if channel doesn't exist
            pass

    def set_frequency(self, name, frequency):
        if name in self.channels:
            ch = self.channels[name]
            if callable(getattr(ch, 'set_frequency', None)):
                ch.set_frequency(frequency)

    def set_filters(self, name, lowpass=None, highpass=None, lp_intensity=None, hp_intensity=None):
        if name in self.channels:
            ch = self.channels[name]
            if callable(getattr(ch, 'set_filters', None)):
                ch.set_filters(lowpass=lowpass, highpass=highpass, lp_intensity=lp_intensity, hp_intensity=hp_intensity)

    def set_effects(self, name, reverb=None, distortion=None):
        if name in self.channels:
            ch = self.channels[name]
            if callable(getattr(ch, 'set_effects', None)):
                ch.set_effects(reverb=reverb, distortion=distortion)

    def set_waveform_type(self, name, wave_type):
        if name in self.channels:
            ch = self.channels[name]
            if callable(getattr(ch, 'set_waveform_type', None)):
                ch.set_waveform_type(wave_type)