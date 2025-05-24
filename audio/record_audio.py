import pyaudio

import wave

FRAME_PER_BUFFER = 3200

FORMAT = pyaudio.paInt16

CHANNEL = 1

RATE =16000

p = pyaudio.PyAudio()


stream = p.open(
    format=FORMAT,
    rate=RATE,
    channels=CHANNEL,
    input=True,
    frames_per_buffer=FRAME_PER_BUFFER
)

print("Streeeee")
second =5

frames =[]

for fra in range(0,int(RATE/FRAME_PER_BUFFER)*5):
    data = stream.read(FRAME_PER_BUFFER)
    
    frames.append(data)
    
stream.stop_stream()
stream.close()

p.terminate()

import wave

obj = wave.open("testingMic.wav","wb")

obj.setnchannels(CHANNEL)
obj.setframerate(RATE)
obj.setsampwidth(p.get_sample_size(FORMAT))
obj.writeframes(b"".join(frames))

obj.close()