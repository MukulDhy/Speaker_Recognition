import matplotlib.pyplot as plt
import wave
import numpy as np

aud = wave.open("testingMic.wav","rb")


sample = aud.readframes(-1)
s = np.frombuffer(sample,dtype=np.int16)

nchannels = aud.getnchannels()         # 1 for mono, 2 for stereo
sampwidth = aud.getsampwidth()         # Sample width in bytes (e.g. 2 bytes = 16-bit)
framerate = aud.getframerate()         # Sample rate (e.g., 44100 Hz)
nframes = aud.getnframes()             # Total number of frames
duration = nframes / framerate              # Duration in seconds
print(f"Channels: {nchannels}")
print(f"Sample Width: {sampwidth * 8} bits")
print(f"Sample Rate: {framerate} Hz")
print(f"Total Frames: {nframes}")
print(f"Duration: {duration:.2f} seconds")

# print(type(s))
print(aud.getnframes())

times = np.linspace(0,duration,num=nframes)

plt.figure(figsize=(15,5))

plt.plot(times,s)
plt.show()