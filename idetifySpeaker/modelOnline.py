import os
import time
import json
import numpy as np
import torch
import librosa
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import queue
import threading
import pyaudio
from collections import deque

app = Flask(__name__)

# Audio Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and Feature Configuration
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class RealTimeProcessor:
    def __init__(self):
        self.audio_buffer = queue.Queue()
        self.is_processing = False
        self.current_speakers = {}
        self.speaker_history = deque(maxlen=10)
        self.p = pyaudio.PyAudio()
        self.model = None
        self.label_encoder = None
        
    def load_model(self, user_id):
        model_path = os.path.join(MODEL_DIR, f"{user_id}_exported_model.pth")
        metadata_path = os.path.join(MODEL_DIR, f"{user_id}_exported_model_metadata.json")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            return False
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.label_encoder = {name: idx for idx, name in enumerate(metadata['family_members'])}
        
        # Load model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        return True
    
    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32, n_fft=1024, hop_length=256)
            features = [
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256), axis=1),
                np.mean(librosa.feature.zero_crossing_rate(y, hop_length=256), axis=1)
            ]
            
            return np.concatenate(features)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def identify_speaker(self, audio_path):
        features = self.extract_features(audio_path)
        if features is None:
            return None
            
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        speaker = list(self.label_encoder.keys())[prediction.item()]
        return speaker, confidence.item()
    
    def start_realtime_processing(self, user_id, callback):
        if not self.load_model(user_id):
            return False
            
        self.is_processing = True
        
        def audio_capture():
            stream = self.p.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               input=True,
                               frames_per_buffer=CHUNK)
            
            while self.is_processing:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_buffer.put(data)
                
            stream.stop_stream()
            stream.close()
        
        def processing_loop():
            audio_segment = bytearray()
            last_process_time = time.time()
            
            while self.is_processing:
                try:
                    data = self.audio_buffer.get_nowait()
                    audio_segment.extend(data)
                    
                    if len(audio_segment) >= RATE * 1.5:  # 1.5 seconds of audio
                        temp_file = os.path.join(tempfile.gettempdir(), f"temp_{time.time()}.wav")
                        with wave.open(temp_file, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(self.p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(audio_segment)
                        
                        result = self.identify_speaker(temp_file)
                        if result:
                            callback(result)
                        
                        audio_segment = bytearray()
                        last_process_time = time.time()
                        os.remove(temp_file)
                        
                except queue.Empty:
                    time.sleep(0.01)
        
        threading.Thread(target=audio_capture, daemon=True).start()
        threading.Thread(target=processing_loop, daemon=True).start()
        return True
    
    def stop_realtime_processing(self):
        self.is_processing = False

rt_processor = RealTimeProcessor()

@app.route('/identify', methods=['POST'])
def identify():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    user_id = request.form.get('user_id', 'default')
    
    if not rt_processor.load_model(user_id):
        return jsonify({"error": "Model not found"}), 404
    
    temp_path = os.path.join(tempfile.gettempdir(), secure_filename(audio_file.filename))
    audio_file.save(temp_path)
    
    result = rt_processor.identify_speaker(temp_path)
    os.remove(temp_path)
    
    if result:
        speaker, confidence = result
        return jsonify({"speaker": speaker, "confidence": confidence})
    return jsonify({"error": "Identification failed"}), 400

@app.route('/realtime/start', methods=['POST'])
def start_realtime():
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    
    def callback(result):
        # In a real implementation, you'd send this via WebSocket
        print(f"Identified speaker: {result[0]} (confidence: {result[1]:.2f})")
    
    success = rt_processor.start_realtime_processing(user_id, callback)
    return jsonify({"success": success})

@app.route('/realtime/stop', methods=['POST'])
def stop_realtime():
    rt_processor.stop_realtime_processing()
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)