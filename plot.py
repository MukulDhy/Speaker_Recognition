import matplotlib.pyplot as plt
import numpy as np
import librosa

# Load audio file
audio_file = "speech_sample.wav"
signal, sample_rate = librosa.load(audio_file, sr=None)

# Create time axis
time = np.arange(0, len(signal)) / sample_rate
print(signal)
print(time)
print(time.shape)

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(time, signal)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()


import librosa.display

# Compute spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)

# Plot spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()




# Compute mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Plot mel spectrogram
plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_spec_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()