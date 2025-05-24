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

# Constants with optimized values
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
FRAME_SIZE = 512
HOP_SIZE = 256
N_MFCC = 32  # Reduced from 40 for faster processing
N_FFT = 1024  # Reduced from 2048 for faster processing
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SILENCE_THRESHOLD = 0.02
DATA_DIR = "family_voice_data"
MODEL_PATH = "family_voice_model.pth"
USERS_PATH = "users.json"
FEATURE_CACHE_DIR = "feature_cache"  # Cache directory for features

# Thread pool for parallel processing
MAX_WORKERS = max(2, os.cpu_count() - 1) if os.cpu_count() else 4
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# Optimized Neural Network Model with batch normalization and skip connections
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
        
        # Skip connection
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

    # Method to optimize model for inference
    def optimize_for_inference(self):
        self.eval()  # Set to evaluation mode
        # Disable gradients for faster inference
        for param in self.parameters():
            param.requires_grad = False
        return self

# Optimized feature extraction with caching
class SpectralFeatureExtractor:
    def __init__(self):
        self.n_mfcc = N_MFCC
        self.n_fft = N_FFT
        self.hop_length = HOP_SIZE
        
    def extract_features(self, audio_path):
        """Extract features with caching for improved performance"""
        # Check cache first
        if audio_path in self.cache:
            return self.cache[audio_path]
        
        # Load audio file
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, res_type='kaiser_fast')  # Faster loading
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None
        
        # Remove silent parts
        y = self._remove_silence(y)
        
        if len(y) == 0:  # If no sound was detected
            return None
        
        try:
            # Extract features in parallel
            features = []
            
            # Extract MFCCs (primary feature)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            features.append(np.mean(mfccs, axis=1))
            features.append(np.std(mfccs, axis=1))
            
            # For real-time performance, we'll limit additional features
            # Extract efficient spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            features.append(np.mean(spectral_centroid, axis=1))
            
            # Zero crossing rate (efficient temporal feature)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            features.append(np.mean(zero_crossing_rate, axis=1))
            
            # Flatten and concatenate all features
            combined_features = np.concatenate(features)
            
            # Cache the result
            self.cache[audio_path] = combined_features
            
            # Periodically save cache (not every time to avoid I/O overhead)
            if len(self.cache) % 10 == 0:
                executor.submit(self.save_cache)
                
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _remove_silence(self, y, threshold=SILENCE_THRESHOLD):
        """More efficient silence removal"""
        # Calculate amplitude envelope using RMS
        frame_length = 1024
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find frames where RMS is above threshold
        voiced_frames = np.where(rms > threshold)[0]
        
        # If all is silence, return empty array
        if len(voiced_frames) == 0:
            return np.array([])
        
        # Reconstruct signal from non-silent frames
        # This is more efficient than masking the entire signal
        voiced_indexes = librosa.frames_to_samples(voiced_frames, hop_length=hop_length)
        voiced_indexes = np.minimum(voiced_indexes, len(y) - 1)
        
        # If we have enough voiced frames
        if len(voiced_indexes) > 10:
            # Find start and end of voiced segments
            start_idx = voiced_indexes[0]
            end_idx = voiced_indexes[-1]
            # Return the audio segment
            return y[start_idx:end_idx+1]
        else:
            return np.array([])

# Optimized Voice Activity Detector with adaptive thresholding
class VoiceActivityDetector:
    def __init__(self, threshold=SILENCE_THRESHOLD):
        self.base_threshold = threshold
        self.adaptive_threshold = threshold
        self.window_size = 30  # frames to check
        self.min_speech_frames = 8  # reduced from 10 for faster detection
        self.energy_history = deque(maxlen=100)
        self.speech_detected = False
        self.speech_prob = 0.0
        
    def is_speech(self, audio_chunk):
        """Enhanced speech detection with adaptive thresholding"""
        # Convert audio to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize
        
        # Calculate energy
        energy = np.mean(np.abs(audio_array))
        self.energy_history.append(energy)
        
        # Dynamically adjust threshold based on recent audio history
        if len(self.energy_history) > 20:
            # Sort energy values
            sorted_energy = sorted(self.energy_history)
            
            # Set threshold between noise floor (10th percentile) and speech level (90th percentile)
            noise_floor = sorted_energy[int(len(sorted_energy) * 0.1)]
            speech_level = sorted_energy[int(len(sorted_energy) * 0.9)]
            
            # Adjust threshold if there's a clear difference
            if speech_level > noise_floor * 1.5:
                self.adaptive_threshold = noise_floor + (speech_level - noise_floor) * 0.3
            else:
                # Fallback to base threshold
                self.adaptive_threshold = self.base_threshold
                
        # Calculate speech probability
        if len(self.energy_history) > 0:
            max_energy = max(self.energy_history)
            if max_energy > 0:
                self.speech_prob = min(1.0, energy / max_energy)
            
        # Speech detection with hysteresis (prevents rapid switching)
        if self.speech_detected:
            # If already in speech mode, use lower threshold to maintain
            is_speech_now = energy > self.adaptive_threshold * 0.7
        else:
            # If in silence mode, require higher energy to switch to speech
            is_speech_now = energy > self.adaptive_threshold
            
        self.speech_detected = is_speech_now
        return is_speech_now, self.speech_prob

# Enhanced speaker recognition system
class SpeakerRecognitionSystem:
    def __init__(self):
        self.feature_extractor = SpectralFeatureExtractor()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_size = None
        self.load_users()
        self._init_audio()
        self.vad = VoiceActivityDetector()
        
        # Enhanced real-time processing
        self.audio_buffer = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        self.current_speakers = {}  # Store currently detected speakers
        self.speaker_history = deque(maxlen=10)  # Store recent speaker predictions (increased from 5)
        
        # Performance monitoring
        self.processing_times = deque(maxlen=50)
        self.last_prediction_time = time.time()
        
        # Model confidence threshold (can be adjusted dynamically)
        self.confidence_threshold = 0.55  # Slightly lowered from 0.6 for better detection
        
        # Prediction cache
        self.prediction_cache = {}
        
    def _init_audio(self):
        """Initialize PyAudio"""
        self.p = pyaudio.PyAudio()
            
    def record_audio(self, member_name=None, duration=RECORD_SECONDS, save_path=None):
        """Record audio for the specified duration with enhanced feedback"""
        if not self.active_user and member_name:
            print("No active user. Please select or register a user first.")
            return None
        
        if member_name and member_name not in [member['name'] for member in self.users[self.active_user]["family_members"]]:
            print(f"Family member {member_name} does not exist for {self.active_user}.")
            return None
        
        print(f"Recording {'sample' if member_name else 'audio'} for {duration} seconds...")
        print("Please speak now...")
        
        # Setup recording
        stream = self.p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        
        frames = []
        audio_levels = []
        
        # Record audio with live feedback
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Calculate audio level for visual feedback
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            level = np.mean(np.abs(audio_array))
            audio_levels.append(level)
            
            # Visual indicator of recording progress and volume
            if i % 5 == 0:  # Update more frequently
                bars = min(40, int(level * 400))
                print(f"\rRecording: [{'#' * bars}{' ' * (40-bars)}] {i/(RATE/CHUNK*duration)*100:.0f}%", end="", flush=True)
        
        print("\nFinished recording.")
        
        # Provide feedback on audio quality
        avg_level = np.mean(audio_levels)
        if avg_level < 0.01:
            print("Warning: Audio level very low. Please speak louder in future recordings.")
        elif avg_level > 0.5:
            print("Warning: Audio level very high. Consider speaking more softly in future recordings.")
        else:
            print("Audio quality seems good.")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Generate filename if not provided
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if member_name:
                member_dir = os.path.join(DATA_DIR, self.active_user, member_name)
                os.makedirs(member_dir, exist_ok=True)
                save_path = os.path.join(member_dir, f"{timestamp}.wav")
            else:
                save_path = os.path.join(DATA_DIR, f"temp_{timestamp}.wav")
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the recorded audio
        wf = wave.open(save_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # If recording for a family member, update the user data
        if member_name:
            for member in self.users[self.active_user]["family_members"]:
                if member["name"] == member_name:
                    member["recordings"].append(save_path)
                    self.save_users()
                    break
        
        return save_path
    
    def extract_features_for_training(self):
        """Extract features from all recordings for training with parallel processing"""
        if not self.active_user:
            print("No active user selected.")
            return None, None
            
        features = []
        labels = []
        futures = []
        
        print("Extracting features for training...")
        
        # Function for parallel processing
        def process_recording(recording, member_name):
            if not os.path.exists(recording):
                print(f"Warning: Recording file {recording} not found. Skipping.")
                return None, None
                
            # Extract features
            feature_vector = self.feature_extractor.extract_features(recording)
            
            if feature_vector is not None:
                return feature_vector, member_name
            else:
                print(f"Warning: Could not extract features from {recording}. Skipping.")
                return None, None
        
        # Submit tasks for parallel processing
        for member in self.users[self.active_user]["family_members"]:
            member_name = member["name"]
            recordings = member["recordings"]
            
            if not recordings:
                print(f"No recordings found for {member_name}. Please record voice samples first.")
                continue
                
            print(f"Processing {len(recordings)} recordings for {member_name}...")
            
            for recording in recordings:
                futures.append(executor.submit(process_recording, recording, member_name))
        
        # Collect results
        total_files = len(futures)
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result, label = future.result()
            if result is not None and label is not None:
                features.append(result)
                labels.append(label)
            
            # Progress update
            if (i+1) % 5 == 0 or i+1 == total_files:
                print(f"Processed {i+1}/{total_files} files...")
        
        if not features:
            print("No features extracted. Make sure all family members have valid recordings.")
            return None, None
            
        # Store the feature size for model initialization
        self.feature_size = len(features[0])
        
        # Convert labels to numerical representation
        self.label_encoder.fit(labels)
        numerical_labels = self.label_encoder.transform(labels)
        
        print(f"Extracted features for {len(set(labels))} family members, total samples: {len(features)}")
        return np.array(features), numerical_labels
    
    
    def _save_model(self, path):
        """Save model and label encoder"""
        # Save model
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'feature_size': self.feature_size,
            'num_speakers': len(self.label_encoder.classes_),
            'label_encoder_classes': self.label_encoder.classes_
        }
        torch.save(model_state, path)
    
    def _load_model(self, path):
        """Load model and label encoder"""
        if not os.path.exists(path):
            print(f"Model file {path} not found.")
            return False
            
        # Load model state
        model_state = torch.load(path, map_location=DEVICE)
        
        # Initialize model with correct dimensions
        self.feature_size = model_state['feature_size']
        num_speakers = model_state['num_speakers']
        self.model = SpeakerRecognitionModel(self.feature_size, 128, num_speakers).to(DEVICE)
        
        # Load model weights
        self.model.load_state_dict(model_state['model_state_dict'])
        
        # Optimize model for inference
        self.model.optimize_for_inference()
        
        # Restore label encoder
        self.label_encoder.classes_ = model_state['label_encoder_classes']
        
        # Clear prediction cache
        self.prediction_cache = {}
        
        return True
    def identify_speaker(self, audio_path=None, use_cache=True):
        """Identify the speaker in the recorded audio with caching"""
        MODEL_PATH_FINAL = "mukul_identify.pth"
        if not self.model:
            model_path = MODEL_PATH_FINAL
            if not self._load_model(model_path):
                print("No trained model found. Please train the model first.")
                return None
        
        # If no audio path is provided, record new audio
        if not audio_path:
            print("Recording audio for identification...")
            audio_path = self.record_audio(duration=3)
        
        start_time = time.time()
        
        feature_vector = self.feature_extractor.extract_features(audio_path)
        
        if feature_vector is None:
            print("Could not extract features from the audio.")
            return None
            
        # Convert to tensor and make prediction
        feature_tensor = torch.FloatTensor(feature_vector).to(DEVICE)
        feature_tensor = feature_tensor.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():  # Disable gradient calculation for faster inference
            output = self.model(feature_tensor)
            
            # Get probabilities
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            confidence = confidence.item()
            prediction = prediction.item()
            
        # Get predicted label
        predicted_speaker = self.label_encoder.inverse_transform([prediction])[0]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        # Cache the result for future use
        self.prediction_cache[audio_path] = (predicted_speaker, confidence)
        
        if confidence < self.confidence_threshold:
            print(f"Speaker identification uncertain (confidence: {confidence:.2f})")
            return None
            
        print(f"Identified speaker: {predicted_speaker} (confidence: {confidence:.2f}, processing time: {processing_time:.3f}s, avg: {avg_processing_time:.3f}s)")
        return predicted_speaker, confidence
    
   
# User interface for the speaker recognition system
def main():
    system = SpeakerRecognitionSystem()
    
    print("Family Voice Recognition System")
    print("===============================")
    print("Recording audio for recognition...")
    audio_path = system.record_audio(duration=3)
    result = system.identify_speaker(audio_path)
    
    if result:
        speaker, confidence = result
        print(f"Identified speaker: {speaker} (confidence: {confidence:.2f})")
    else:
        print("Could not identify the speaker.")
                

if __name__ == "__main__":
    main()