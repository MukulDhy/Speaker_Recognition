import wave
import numpy as np
import matplotlib.pyplot as plt
# --- Load audio manually ---
with wave.open("speech_sample.wav", 'rb') as wf:
    sr = wf.getframerate()
    n_samples = wf.getnframes()
    audio = wf.readframes(n_samples)
    signal = np.frombuffer(audio, dtype=np.int16)

# --- Normalize ---
normalized_signal = signal / np.max(np.abs(signal))

# --- Resample (using scipy or linear interpolation) ---
from scipy.signal import resample

target_sr = 16000
num_samples = int(len(normalized_signal) * target_sr / sr)
resampled_signal = resample(normalized_signal, num_samples)
sr = target_sr

# --- Noise Reduction (simple) ---
# Example: Spectral gating or external libraries needed
# Or: apply a moving average filter for very basic smoothing
smoothed_signal = np.convolve(resampled_signal, np.ones(3)/3, mode='same')

# --- Trimming silence (manual thresholding) ---
threshold = 0.02
non_silent = smoothed_signal[np.abs(smoothed_signal) > threshold]

# --- Pre-emphasis Filter ---
pre_emphasized = np.append(non_silent[0], non_silent[1:] - 0.97 * non_silent[:-1])


plt.figure(figsize=(12, 4))
plt.plot(pre_emphasized)
plt.title("Preprocessed Audio Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()