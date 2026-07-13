import wave
import struct
import math
import os
from config import AUDIO_INPUT_DIR

def generate_tone(filename, duration=15, freq=440.0, volume=0.5, sample_rate=48000):
    if not os.path.exists(AUDIO_INPUT_DIR):
        os.makedirs(AUDIO_INPUT_DIR)
        
    path = os.path.join(AUDIO_INPUT_DIR, filename)
    
    n_samples = int(duration * sample_rate)
    
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        
        for i in range(n_samples):
            value = int(volume * 32767.0 * math.sin(2.0 * math.pi * freq * i / sample_rate))
            data = struct.pack('<h', value)
            w.writeframesraw(data)
            
    print(f"Generated test audio at {path}")

if __name__ == "__main__":
    generate_tone("test_audio.wav")
