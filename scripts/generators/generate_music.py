import numpy as np
import wave
import struct

# Generate a simple sine wave music
sample_rate = 44100
duration = 5  # seconds
frequency = 440  # A4 note

t = np.linspace(0, duration, int(sample_rate * duration), False)
audio = np.sin(frequency * 2 * np.pi * t)

# Normalize to 16-bit range
audio = (audio * 32767).astype(np.int16)

# Write to WAV file
with wave.open("generated_music.wav", "wb") as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    for sample in audio:
        wav_file.writeframes(struct.pack("<h", sample))

print("Generated simple music as generated_music.wav")
