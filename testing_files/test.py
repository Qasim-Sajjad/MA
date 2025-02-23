import librosa
import numpy as np

# Path to your audio file
audio_path = "audio_files\A Beacon of Hope.mp3"  # Change this to your audio file path

# Load the audio file
# y is the audio signal, sr is the sampling rate
y, sr = librosa.load(audio_path)

# Print basic information about the loaded audio
print(f"Sampling rate: {sr} Hz")
print(f"Audio length: {len(y)} samples")
print(f"Duration: {len(y)/sr:.2f} seconds")

# If you want to specify a different sampling rate, you can use:
# y, sr = librosa.load(audio_path, sr=22050)  # Default sr=22050

# If you want to load only a portion of the audio:
# y, sr = librosa.load(audio_path, offset=30, duration=10)  # Load 10 seconds starting at 30 seconds