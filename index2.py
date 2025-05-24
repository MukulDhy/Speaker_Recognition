import librosa
import soundfile as sf

# Load an audio file
audio_file = "sample.wav"
signal, sample_rate = librosa.load(audio_file, sr=None)

# Print basic information
print(f"Sample rate: {sample_rate} Hz")
print(f"Audio duration: {len(signal)/sample_rate:.2f} seconds")
print(f"Number of samples: {len(signal)}")

# Alternative using soundfile
data, samplerate = sf.read(audio_file)