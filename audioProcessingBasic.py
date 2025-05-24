import wave
import matplotlib.pyplot as plt
audio_file = wave.open("sample.wav","r")


# print(audio_file.getnchannels())


# plt.plot(audio_file)
# print(audio_file.getnframes()/ audio_file.getframerate())
frames = audio_file.readframes(1)
print(frames[0:1000:1])