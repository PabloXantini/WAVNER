# WAVNER Operation

WAVNER operates trough real-time hand gesture recognition using MediaPipe, primarly utilizing 3 audio channels, which can be customized to set any type of simple signal

> _**Nota:** This versión of the software is currently in development, and customization is presently applied only by modifying the files in this repository._

## Description of Audio Channels

* **Background Channels:** 
These are channels where background music or sounds are played; they limited modulation driven by hand gestures.

* **Main Channel:** Designed for gesture modulation using modifiers which include:
    * High pass filter.
    * Low pass filter.
    * Reverb/Echo
    * Distortion
    * Volume
    * Pitch

* **Extra Channel:** An additional channel where extra signals can be swapped using keyboard keys.

## Controls

### Left Hand
* **Pinky finger gesture (curl):** Background Channel 1 volume.
* **Ring-Middle finger gesture (curl):** Background Channel 2 volume.
* **Gesto de dedo índice (doblez) Index finger gesture (curl):** Main Channel distortion modifier.
* **Thumb gesture (curl):** Main Channel reverb modifier.

## Right Hand
* **Thumb-Index finger gesture (pinch):** Main Channel low-pass filter modifier.
* **Middle/Ring-Pinky finger gesture (pinch):** Main Channel high-pass modifier.

# Keyboard
<table style="margin: auto;">
    <thead>
        <th>Key</th>
        <th>Action</th>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>Set the <b>sinoidal waveform</b> at Extra Channel.</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Set the <b>square waveform</b> at Extra Channel.</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Set the <b>triangular waveform</b> at Extra Channel.</td>
        </tr>
        <tr>
            <td>4</td>
            <td>Set the <b>sawtoorh waveform</b> at Extra Channel.</td>
        </tr>
        <tr>
            <td>ENTER</td>
            <td>Store Extra-Channel.</td>
        </tr>
        <tr>
            <td>SPACE</td>
            <td>Toggle-off Extra Channel.</td>
        </tr>
    </tbody>
</table>
