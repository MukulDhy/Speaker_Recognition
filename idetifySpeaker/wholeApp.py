import os
import time
import threading
import json
import numpy as np
import pyaudio
import wave
import pickle
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from scipy.io import wavfile
from collections import deque, Counter
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import split_on_silence
import queue
import concurrent.futures
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import uuid

# Constants with optimized values
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
FRAME_SIZE = 512
HOP_SIZE = 256
N_MFCC = 32
N_FFT = 1024
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SILENCE_THRESHOLD = 0.02
DATA_DIR = "family_voice_data"
MODEL_PATH = "family_voice_model.pth"
USERS_PATH = "users.json"
FEATURE_CACHE_DIR = "feature_cache"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

# Thread pool for parallel processing
MAX_WORKERS = max(2, os.cpu_count() - 1) if os.cpu_count() else 4
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Neural Network Model (same as before)
class SpeakerRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_speakers):
        super(SpeakerRecognitionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_speakers)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.skip = nn.Linear(input_size, hidden_size // 2)
        
    def forward(self, x):
        # Main path
        main = self.relu(self.bn1(self.layer1(x)))
        main = self.dropout(main)
        main = self.relu(self.bn2(self.layer2(main)))
        
        # Skip connection path
        skip = self.relu(self.skip(x))
        
        # Combine paths
        combined = main + skip
        combined = self.dropout(combined)
        output = self.layer3(combined)
        
        return output

    def optimize_for_inference(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

# SpectralFeatureExtractor (same as before)
class SpectralFeatureExtractor:
    def __init__(self):
        self.n_mfcc = N_MFCC
        self.n_fft = N_FFT
        self.hop_length = HOP_SIZE
        self.cache = {}
        self.load_cache()
        
    def load_cache(self):
        cache_file = os.path.join(FEATURE_CACHE_DIR, "feature_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached features")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        cache_file = os.path.join(FEATURE_CACHE_DIR, "feature_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                if len(self.cache) > 1000:
                    keys = list(self.cache.keys())
                    for key in keys[:-1000]:
                        del self.cache[key]
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
        
    def extract_features(self, audio_path):
        if audio_path in self.cache:
            return self.cache[audio_path]
        
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, res_type='kaiser_fast')
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None
        
        y = self._remove_silence(y)
        
        if len(y) == 0:
            return None
        
        try:
            features = []
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            features.append(np.mean(mfccs, axis=1))
            features.append(np.std(mfccs, axis=1))
            
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            features.append(np.mean(spectral_centroid, axis=1))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            features.append(np.mean(zero_crossing_rate, axis=1))
            
            combined_features = np.concatenate(features)
            self.cache[audio_path] = combined_features
            
            if len(self.cache) % 10 == 0:
                executor.submit(self.save_cache)
                
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _remove_silence(self, y, threshold=SILENCE_THRESHOLD):
        frame_length = 1024
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        voiced_frames = np.where(rms > threshold)[0]
        
        if len(voiced_frames) == 0:
            return np.array([])
        
        voiced_indexes = librosa.frames_to_samples(voiced_frames, hop_length=hop_length)
        voiced_indexes = np.minimum(voiced_indexes, len(y) - 1)
        
        if len(voiced_indexes) > 10:
            start_idx = voiced_indexes[0]
            end_idx = voiced_indexes[-1]
            return y[start_idx:end_idx+1]
        else:
            return np.array([])

# VoiceActivityDetector (same as before)
class VoiceActivityDetector:
    def __init__(self, threshold=SILENCE_THRESHOLD):
        self.base_threshold = threshold
        self.adaptive_threshold = threshold
        self.window_size = 30
        self.min_speech_frames = 8
        self.energy_history = deque(maxlen=100)
        self.speech_detected = False
        self.speech_prob = 0.0
        
    def is_speech(self, audio_chunk):
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        energy = np.mean(np.abs(audio_array))
        self.energy_history.append(energy)
        
        if len(self.energy_history) > 20:
            sorted_energy = sorted(self.energy_history)
            noise_floor = sorted_energy[int(len(sorted_energy) * 0.1)]
            speech_level = sorted_energy[int(len(sorted_energy) * 0.9)]
            
            if speech_level > noise_floor * 1.5:
                self.adaptive_threshold = noise_floor + (speech_level - noise_floor) * 0.3
            else:
                self.adaptive_threshold = self.base_threshold
                
        if len(self.energy_history) > 0:
            max_energy = max(self.energy_history)
            if max_energy > 0:
                self.speech_prob = min(1.0, energy / max_energy)
            
        if self.speech_detected:
            is_speech_now = energy > self.adaptive_threshold * 0.7
        else:
            is_speech_now = energy > self.adaptive_threshold
            
        self.speech_detected = is_speech_now
        return is_speech_now, self.speech_prob

# SpeakerRecognitionSystem (same as before, with minor adjustments for server)
class SpeakerRecognitionSystem:
    def __init__(self):
        self.feature_extractor = SpectralFeatureExtractor()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.users = {}
        self.active_user = None
        self.family_members = []
        self.feature_size = None
        self.load_users()
        self._init_audio()
        self.vad = VoiceActivityDetector()
        self.audio_buffer = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        self.current_speakers = {}
        self.speaker_history = deque(maxlen=10)
        self.processing_times = deque(maxlen=50)
        self.last_prediction_time = time.time()
        self.confidence_threshold = 0.55
        self.prediction_cache = {}
    
    # All the methods from your original class go here
    # (register_user, select_user, record_audio, train_model, identify_speaker, etc.)
    # I'm omitting them for brevity, but they should be included exactly as in your original code
    
    def save_uploaded_file(self, file):
        """Save uploaded file to uploads folder and return path"""
        if not file:
            return None
            
        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.wav'):
            filename = f"{filename}.wav"
            
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        return filepath

# Initialize the system
system = SpeakerRecognitionSystem()

# Helper function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API Routes
@app.route('/api/register_user', methods=['POST'])
def register_user():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    success = system.register_user(username)
    if success:
        return jsonify({'message': f'User {username} registered successfully'}), 200
    else:
        return jsonify({'error': f'User {username} already exists'}), 400

@app.route('/api/select_user', methods=['POST'])
def select_user():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    success = system.select_user(username)
    if success:
        return jsonify({'message': f'Selected user: {username}'}), 200
    else:
        return jsonify({'error': f'User {username} does not exist'}), 404

@app.route('/api/register_family_member', methods=['POST'])
def register_family_member():
    if not system.active_user:
        return jsonify({'error': 'No active user selected'}), 400
    
    data = request.get_json()
    member_name = data.get('member_name')
    if not member_name:
        return jsonify({'error': 'Member name is required'}), 400
    
    success = system.register_family_member(member_name)
    if success:
        return jsonify({'message': f'Family member {member_name} registered'}), 200
    else:
        return jsonify({'error': f'Family member {member_name} already exists'}), 400

@app.route('/api/record_samples', methods=['POST'])
def record_samples():
    if not system.active_user:
        return jsonify({'error': 'No active user selected'}), 400
    
    data = request.get_json()
    member_name = data.get('member_name')
    num_samples = data.get('num_samples', 3)
    
    if not member_name:
        return jsonify({'error': 'Member name is required'}), 400
    
    success = system.record_family_voice_samples(member_name, num_samples)
    if success:
        return jsonify({'message': f'Recorded {num_samples} samples for {member_name}'}), 200
    else:
        return jsonify({'error': 'Failed to record samples'}), 400

@app.route('/api/train_model', methods=['POST'])
def train_model():
    if not system.active_user:
        return jsonify({'error': 'No active user selected'}), 400
    
    print("Starting model training...")
    success = system.train_model()
    if success:
        return jsonify({'message': 'Model trained successfully'}), 200
    else:
        return jsonify({'error': 'Failed to train model'}), 400

@app.route('/api/identify_speaker', methods=['POST'])
def identify_speaker():
    if not system.active_user:
        return jsonify({'error': 'No active user selected'}), 400
    
    if not system.model:
        return jsonify({'error': 'No trained model available'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            audio_path = system.save_uploaded_file(file)
            if not audio_path:
                return jsonify({'error': 'Failed to save uploaded file'}), 400
            
            # Identify speaker
            result = system.identify_speaker(audio_path)
            
            # Clean up temporary file
            try:
                os.remove(audio_path)
            except:
                pass
            
            if result:
                speaker, confidence = result
                return jsonify({
                    'speaker': speaker,
                    'confidence': float(confidence),
                    'message': 'Speaker identified successfully'
                }), 200
            else:
                return jsonify({'error': 'Could not identify speaker'}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only WAV files are allowed'}), 400

@app.route('/api/start_realtime', methods=['POST'])
def start_realtime():
    if not system.active_user:
        return jsonify({'error': 'No active user selected'}), 400
    
    if not system.model:
        return jsonify({'error': 'No trained model available'}), 400
    
    try:
        # Start real-time recognition in a separate thread
        def recognition_thread():
            system.start_real_time_recognition()
        
        thread = threading.Thread(target=recognition_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Real-time recognition started'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to start real-time recognition: {str(e)}'}), 500

@app.route('/api/stop_realtime', methods=['POST'])
def stop_realtime():
    system.stop_real_time_recognition()
    return jsonify({'message': 'Real-time recognition stopped'}), 200

@app.route('/api/list_users', methods=['GET'])
def list_users():
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, 'r') as f:
            users = json.load(f)
        return jsonify({'users': list(users.keys())}), 200
    else:
        return jsonify({'users': []}), 200

@app.route('/api/list_family_members', methods=['GET'])
def list_family_members():
    if not system.active_user:
        return jsonify({'error': 'No active user selected'}), 400
    
    family_members = [member['name'] for member in system.users[system.active_user]["family_members"]]
    return jsonify({'family_members': family_members}), 200

@app.route('/api/export_model', methods=['POST'])
def export_model():
    if not system.active_user:
        return jsonify({'error': 'No active user selected'}), 400
    
    if not system.model:
        return jsonify({'error': 'No trained model available'}), 400
    
    data = request.get_json()
    export_path = data.get('export_path', None)
    
    success = system.export_model(export_path)
    if success:
        return jsonify({'message': 'Model exported successfully'}), 200
    else:
        return jsonify({'error': 'Failed to export model'}), 400

@app.route('/api/optimize', methods=['POST'])
def optimize():
    if not system.active_user or not system.model:
        return jsonify({'error': 'No model loaded'}), 400
    
    success = system.optimize_for_device()
    if success:
        return jsonify({'message': f'Model optimized for {DEVICE}'}), 200
    else:
        return jsonify({'error': 'Failed to optimize model'}), 400

@app.route('/api/status', methods=['GET'])
def status():
    status = {
        'active_user': system.active_user,
        'model_loaded': system.model is not None,
        'device': str(DEVICE),
        'family_members': [member['name'] for member in system.users.get(system.active_user, {}).get("family_members", [])]
    }
    return jsonify(status), 200

if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)