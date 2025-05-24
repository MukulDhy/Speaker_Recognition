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
        self.cache = {}
        self.load_cache()
        
    def load_cache(self):
        """Load feature cache from disk"""
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
        """Save feature cache to disk"""
        cache_file = os.path.join(FEATURE_CACHE_DIR, "feature_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                # Limit cache size to 1000 entries to prevent it from growing too large
                if len(self.cache) > 1000:
                    # Keep only the most recent 1000 entries
                    keys = list(self.cache.keys())
                    for key in keys[:-1000]:
                        del self.cache[key]
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
        
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
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
        
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
        self.users = {}
        self.active_user = None
        self.family_members = []
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
        
    def load_users(self):
        """Load users from json file"""
        if os.path.exists(USERS_PATH):
            with open(USERS_PATH, 'r') as f:
                self.users = json.load(f)
                
    def save_users(self):
        """Save users to json file"""
        # Save in a separate thread to avoid blocking
        def _save():
            with open(USERS_PATH, 'w') as f:
                json.dump(self.users, f, indent=2)
        
        executor.submit(_save)
            
    def register_user(self, username):
        """Register a new user (family)"""
        if username in self.users:
            print(f"User {username} already exists.")
            return False
        
        self.users[username] = {"family_members": []}
        self.save_users()
        self.active_user = username
        print(f"User {username} registered successfully.")
        return True
    
    def register_family_member(self, member_name):
        """Register a new family member for the active user"""
        if not self.active_user:
            print("No active user. Please select or register a user first.")
            return False
        
        # Check if member already exists
        if member_name in [member['name'] for member in self.users[self.active_user]["family_members"]]:
            print(f"Family member {member_name} already exists for {self.active_user}.")
            return False
        
        # Add new member
        self.users[self.active_user]["family_members"].append({
            "name": member_name, 
            "recordings": []
        })
        self.save_users()
        print(f"Family member {member_name} registered for {self.active_user}.")
        return True
    
    def select_user(self, username):
        """Select an existing user as active"""
        if username not in self.users:
            print(f"User {username} does not exist.")
            return False
        
        self.active_user = username
        self.family_members = [member['name'] for member in self.users[self.active_user]["family_members"]]
        print(f"Selected user: {username}")
        print(f"Family members: {', '.join(self.family_members)}")
        
        # Load model for this user if it exists
        model_path = f"{username}_model.pth"
        if os.path.exists(model_path):
            self._load_model(model_path)
            print(f"Loaded model for {username}")
        
        return True
    
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
    
    def record_family_voice_samples(self, member_name, num_samples=3):
        """Record multiple voice samples for a family member with improved guidance"""
        if not self.active_user:
            print("No active user. Please select or register a user first.")
            return False
            
        # Check if member exists
        found = False
        for member in self.users[self.active_user]["family_members"]:
            if member["name"] == member_name:
                found = True
                break
                
        if not found:
            print(f"Family member {member_name} not found.")
            return False
        
        print(f"Recording {num_samples} voice samples for {member_name}")
        print("Tips for good voice samples:")
        print("- Speak naturally at a normal pace")
        print("- Include different sentences for each sample")
        print("- Try to maintain a consistent distance from the microphone")
        print("- Minimize background noise")
        
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}")
            input("Press Enter to start recording...")
            
            # Create directory for member if it doesn't exist
            member_dir = os.path.join(DATA_DIR, self.active_user, member_name)
            os.makedirs(member_dir, exist_ok=True)
            
            # Record and save sample
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(member_dir, f"sample_{i+1}_{timestamp}.wav")
            self.record_audio(member_name, duration=5, save_path=file_path)
            
            # Give user time to prepare for next sample
            if i < num_samples - 1:
                print("Great! Let's continue to the next sample.")
                time.sleep(1)
        
        print(f"Completed recording {num_samples} samples for {member_name}")
        return True
    
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
    
    def train_model(self):
        """Train the speaker recognition model with improved techniques"""
        if not self.active_user:
            print("No active user selected.")
            return False
            
        # Extract features for training
        X, y = self.extract_features_for_training()
        if X is None or len(X) == 0:
            return False
            
        # Data augmentation (add slight variations to increase dataset)
        print("Applying data augmentation to improve model robustness...")
        X_augmented = []
        y_augmented = []
        
        # Add original data
        X_augmented.extend(X)
        y_augmented.extend(y)
        
        # Add noise variants (random noise added to features)
        for i in range(len(X)):
            noise = np.random.normal(0, 0.01, X[i].shape)
            X_augmented.append(X[i] + noise)
            y_augmented.append(y[i])
        
        # Convert to numpy arrays
        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_augmented)
        y_tensor = torch.LongTensor(y_augmented)
        
        # Initialize model
        num_speakers = len(self.label_encoder.classes_)
        self.model = SpeakerRecognitionModel(self.feature_size, 128, num_speakers).to(DEVICE)
        
        # Training parameters with learning rate scheduling
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Enhanced training parameters
        num_epochs = 100  # Keep epochs the same
        batch_size = min(16, len(X_augmented))  # Larger batch size if enough data
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Create data indices for training and validation splits
        indices = torch.randperm(X_tensor.size()[0])
        split = int(X_tensor.size()[0] * 0.8)  # 80% training, 20% validation
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        # Training loop with early stopping
        print(f"Starting training with {len(train_indices)} training and {len(val_indices)} validation samples...")
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            # Simple batch creation
            permutation = torch.randperm(train_indices.size()[0])
            
            for i in range(0, train_indices.size()[0], batch_size):
                indices = train_indices[permutation[i:i+batch_size]]
                batch_x, batch_y = X_tensor[indices], y_tensor[indices]
                
                # Move to device
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                val_x = X_tensor[val_indices].to(DEVICE)
                val_y = y_tensor[val_indices].to(DEVICE)
                
                outputs = self.model(val_x)
                val_loss = criterion(outputs, val_y).item()
                
                _, predicted = torch.max(outputs, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()
            
            val_accuracy = 100 * correct / total
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            if (epoch+1) % 10 == 0 or epoch < 5:
                print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_indices)*batch_size:.4f}, '
                      f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                
                # Save the best model
                model_path = f"{self.active_user}_model.pth"
                self._save_model(model_path)
                print(f"New best model saved! (Validation Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load the best model (which we saved during training)
        model_path = f"{self.active_user}_model.pth"
        self._load_model(model_path)
        
        # Optimize model for inference
        self.model.optimize_for_inference()
        print(f"Training completed. Final model accuracy: {val_accuracy:.2f}%")
        return True
    
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
        if not self.active_user:
            print("No active user selected.")
            return None
            
        if not self.model:
            model_path = f"{self.active_user}_model.pth"
            if not self._load_model(model_path):
                print("No trained model found. Please train the model first.")
                return None
        
        # If no audio path is provided, record new audio
        if not audio_path:
            print("Recording audio for identification...")
            audio_path = self.record_audio(duration=3)
        
        # Check cache for faster repeated identification
        if use_cache and audio_path in self.prediction_cache:
            return self.prediction_cache[audio_path]
        
        start_time = time.time()
        
        # Extract features
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
    
    def start_real_time_recognition(self, window_size=1.0, step_size=0.5):
        """Start real-time speaker recognition with optimized performance"""
        if not self.active_user:
            print("No active user selected.")
            return
            
        if not self.model:
            model_path = f"{self.active_user}_model.pth"
            if not self._load_model(model_path):
                print("No trained model found. Please train the model first.")
                return
                
        print(f"Starting real-time recognition for {self.active_user}'s family...")
        print(f"Detected family members: {', '.join(self.label_encoder.classes_)}")
        
        # Initialize variables
        self.is_processing = True
        audio_window = deque(maxlen=int(RATE * window_size))
        last_prediction_time = time.time()
        prediction_interval = step_size
        prediction_frames = []
        is_speech_active = False
        speech_cooldown = 0
        
        # Setup audio stream
        stream = self.p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        
        try:
            print("Listening... (Press Ctrl+C to stop)")
            
            # Buffer for storing audio segments
            audio_buffer = bytearray()
            last_process_time = time.time()
            
            # Start processing thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pred_executor:
                future_prediction = None
                
                while self.is_processing:
                    # Read audio chunk
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    
                    # Append to audio window
                    audio_window.extend(data)
                    
                    # Check for voice activity
                    is_speech, speech_prob = self.vad.is_speech(data)
                    
                    if is_speech:
                        # Reset cooldown when speech is detected
                        speech_cooldown = 15  # ~1.5 seconds at 10 chunks/second
                        
                        # Collect frames for processing
                        prediction_frames.append(data)
                        
                        # Visual feedback of speech detection
                        if not is_speech_active:
                            is_speech_active = True
                            print("\nSpeech detected!", end="", flush=True)
                        else:
                            # Show speech probability as a simple bar
                            bars = int(speech_prob * 10)
                            print(f"\rSpeech level: [{'#' * bars}{' ' * (10-bars)}]", end="", flush=True)
                    else:
                        # Decrease cooldown
                        if speech_cooldown > 0:
                            speech_cooldown -= 1
                            prediction_frames.append(data)
                        elif is_speech_active and speech_cooldown == 0:
                            # Speech ended, reset activity flag
                            is_speech_active = False
                            print("\nSpeech ended.", end="", flush=True)
                    
                    # Only process if we've accumulated enough speech data and it's time for a new prediction
                    current_time = time.time()
                    elapsed = current_time - last_process_time
                    
                    if (len(prediction_frames) >= int(RATE * 1.0 / CHUNK) and  # At least 1 second of audio
                        elapsed >= prediction_interval and                     # Minimum time between predictions
                        (is_speech or speech_cooldown > 0) and                # Speech is active or just ended
                        (future_prediction is None or future_prediction.done())):  # Previous prediction complete
                        
                        # Join frames for processing
                        audio_segment = b''.join(prediction_frames[-int(RATE * 1.5 / CHUNK):])  # Use last 1.5 seconds
                        
                        # Process in separate thread
                        future_prediction = pred_executor.submit(self._process_audio_segment, audio_segment)
                        
                        # Reset for next prediction
                        last_process_time = current_time
                        
                    # Check if prediction is ready
                    if future_prediction and future_prediction.done():
                        result = future_prediction.result()
                        if result:
                            speaker, confidence = result
                            
                            # Update speaker history with confidence-weighted entry
                            self.speaker_history.append((speaker, confidence))
                            
                            # Get most common speaker from recent history with weighted voting
                            speaker_votes = {}
                            for spk, conf in self.speaker_history:
                                if spk in speaker_votes:
                                    speaker_votes[spk] += conf
                                else:
                                    speaker_votes[spk] = conf
                            
                            # Find speaker with most weighted votes
                            if speaker_votes:
                                current_speaker = max(speaker_votes.items(), key=lambda x: x[1])[0]
                                confidence_avg = speaker_votes[current_speaker] / sum(1 for s, _ in self.speaker_history if s == current_speaker)
                                
                                # Update current speakers dictionary with timestamp
                                self.current_speakers[current_speaker] = {
                                    'last_detected': time.time(),
                                    'confidence': confidence_avg
                                }
                            
                            # Display active speakers (those detected in the last 5 seconds)
                            active_speakers = {
                                name: data for name, data in self.current_speakers.items()
                                if time.time() - data['last_detected'] < 5
                            }
                            
                            if active_speakers:
                                # Clear line and display speakers
                                speakers_str = ", ".join([
                                    f"{name} ({data['confidence']:.2f})" 
                                    for name, data in active_speakers.items()
                                ])
                                print(f"\rCurrent speakers: {speakers_str}", end="", flush=True)
                        
                        # Reset future
                        future_prediction = None
                    
                    # Sleep briefly to reduce CPU usage
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("\nStopping real-time recognition...")
        finally:
            # Cleanup
            stream.stop_stream()
            stream.close()
            self.is_processing = False
    def _process_audio_segment(self, audio_segment):
        try:
            # Save audio to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_file = os.path.join(DATA_DIR, f"temp_{timestamp}.wav")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            
            # Save audio
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_segment)
            wf.close()
            
            # Identify speaker
            result = self.identify_speaker(temp_file)
            
            # Remove temporary file
            try:
                os.remove(temp_file)
            except:
                pass  # Ignore errors removing temp file
                
            return result
            
        except Exception as e:
            print(f"Error processing audio segment: {e}")
            return None
    
    def stop_real_time_recognition(self):
        """Stop real-time recognition"""
        self.is_processing = False
        print("Stopping real-time recognition...")
    
    def tune_recognition_parameters(self):
        """Auto-tune parameters for improved recognition performance"""
        if not self.active_user:
            print("No active user selected.")
            return
            
        if not self.model:
            model_path = f"{self.active_user}_model.pth"
            if not self._load_model(model_path):
                print("No trained model found. Please train the model first.")
                return
                
        print("Tuning recognition parameters...")
        
        # Test with different confidence thresholds
        thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        best_threshold = 0.5
        best_score = 0
        
        # Collect some test samples for each family member
        test_samples = {}
        
        for member_name in self.label_encoder.classes_:
            # Find recordings for this member
            for member in self.users[self.active_user]["family_members"]:
                if member["name"] == member_name:
                    if member["recordings"]:
                        # Use a subset of recordings for testing
                        test_recordings = member["recordings"][-min(3, len(member["recordings"])):]
                        test_samples[member_name] = test_recordings
                        break
        
        if not test_samples:
            print("No test samples available. Make sure all family members have recordings.")
            return
            
        print(f"Testing with {sum(len(samples) for samples in test_samples.values())} samples...")
        
        # Test each threshold
        for threshold in thresholds:
            self.confidence_threshold = threshold
            
            correct = 0
            total = 0
            
            for true_speaker, recordings in test_samples.items():
                for recording in recordings:
                    result = self.identify_speaker(recording)
                    
                    if result:
                        predicted_speaker, _ = result
                        if predicted_speaker == true_speaker:
                            correct += 1
                    
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            print(f"Threshold {threshold}: Accuracy {accuracy:.2f} ({correct}/{total})")
            
            if accuracy > best_score:
                best_score = accuracy
                best_threshold = threshold
        
        # Set optimal threshold
        self.confidence_threshold = best_threshold
        print(f"Optimal confidence threshold: {best_threshold} (accuracy: {best_score:.2f})")
        
        # Also tune VAD threshold
        print("\nTuning Voice Activity Detection threshold...")
        
        vad_thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]
        best_vad_threshold = 0.02
        best_vad_score = 0
        
        for vad_threshold in vad_thresholds:
            # Create a temporary VAD with this threshold
            temp_vad = VoiceActivityDetector(threshold=vad_threshold)
            
            # Test on each recording
            correct_frames = 0
            total_frames = 0
            
            for recordings in test_samples.values():
                for recording in recordings[:1]:  # Test with just one recording per person for speed
                    try:
                        # Load audio
                        wf = wave.open(recording, 'rb')
                        audio_data = wf.readframes(wf.getnframes())
                        wf.close()
                        
                        # Process in chunks
                        for i in range(0, len(audio_data), CHUNK):
                            chunk = audio_data[i:i+CHUNK]
                            if len(chunk) < CHUNK:
                                break
                                
                            # Test VAD
                            is_speech, _ = temp_vad.is_speech(chunk)
                            
                            # We assume most frames should be speech since these are speaking samples
                            if is_speech:
                                correct_frames += 1
                            
                            total_frames += 1
                    except Exception as e:
                        print(f"Error processing file {recording}: {e}")
            
            speech_ratio = correct_frames / total_frames if total_frames > 0 else 0
            print(f"VAD Threshold {vad_threshold}: Speech detection rate {speech_ratio:.2f}")
            
            # We want a balance, not too sensitive but not missing speech
            score = speech_ratio if speech_ratio > 0.5 else 0
            
            if score > best_vad_score:
                best_vad_score = score
                best_vad_threshold = vad_threshold
        
        # Create a new VAD with optimal threshold
        self.vad = VoiceActivityDetector(threshold=best_vad_threshold)
        print(f"Optimal VAD threshold: {best_vad_threshold} (speech detection rate: {best_vad_score:.2f})")
        
        # Save settings
        self.users[self.active_user]["settings"] = {
            "confidence_threshold": best_threshold,
            "vad_threshold": best_vad_threshold
        }
        self.save_users()
        print("Parameters tuned and saved.")
    
    def export_model(self, export_path=None):
        """Export the trained model for deployment"""
        if not self.active_user:
            print("No active user selected.")
            return False
            
        if not self.model:
            model_path = f"{self.active_user}_model.pth"
            if not self._load_model(model_path):
                print("No trained model found. Please train the model first.")
                return False
        
        if not export_path:
            export_path = f"{self.active_user}_exported_model.pth"
            
        # Convert model to TorchScript for faster inference
        try:
            # Create example input for tracing
            example_input = torch.zeros((1, self.feature_size), dtype=torch.float32).to(DEVICE)
            
            # Trace the model
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Save the traced model
            traced_model.save(export_path)
            
            # Save metadata separately
            metadata_path = export_path.replace('.pth', '_metadata.json')
            metadata = {
                "family_members": list(self.label_encoder.classes_),
                "feature_size": self.feature_size,
                "confidence_threshold": self.confidence_threshold,
                "vad_threshold": self.vad.base_threshold
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Model exported to {export_path}")
            print(f"Metadata saved to {metadata_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting model: {e}")
            return False
    
    def optimize_for_device(self):
        """Optimize model for the current device (CPU/GPU)"""
        if not self.model:
            print("No model loaded.")
            return False
            
        print(f"Optimizing model for {DEVICE}...")
        
        if DEVICE.type == 'cuda':
            # GPU optimizations
            torch.backends.cudnn.benchmark = True
            
            # Convert to half precision if available and not already
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and next(self.model.parameters()).dtype != torch.float16:
                try:
                    self.model = self.model.half()
                    print("Model converted to half precision for faster GPU inference.")
                except Exception as e:
                    print(f"Could not convert to half precision: {e}")
        else:
            # CPU optimizations
            try:
                # Try to use Intel MKL optimizations if available
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
                print("Model optimized with Intel extensions for PyTorch.")
            except ImportError:
                # Fallback to standard optimizations
                if hasattr(torch, 'quantization'):
                    # Quantize model to int8 for faster CPU inference
                    try:
                        torch.backends.quantized.engine = 'qnnpack'  # For ARM, use 'qnnpack'
                        model_fp32 = self.model
                        model_fp32.eval()
                        model_int8 = torch.quantization.quantize_dynamic(
                            model_fp32, {nn.Linear}, dtype=torch.qint8
                        )
                        self.model = model_int8
                        print("Model quantized to int8 for faster CPU inference.")
                    except Exception as e:
                        print(f"Could not quantize model: {e}")
        
        # Optimize parameters
        self.model.optimize_for_inference()
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        self.is_processing = False
        try:
            self.p.terminate()
        except:
            pass
        # Save cache before exiting
        self.feature_extractor.save_cache()
        print("Resources cleaned up.")

# User interface for the speaker recognition system
def main():
    system = SpeakerRecognitionSystem()
    
    print("Family Voice Recognition System")
    print("===============================")
    
    while True:
        print("\nMenu:")
        print("1. Register new user (family)")
        print("2. Select existing user")
        print("3. Add family member")
        print("4. Record voice samples for family member")
        print("5. Train recognition model")
        print("6. Test recognition on recorded audio")
        print("7. Start real-time recognition")
        print("8. Tune recognition parameters")
        print("9. Export model")
        print("10. Optimize for device")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            username = input("Enter new user (family) name: ")
            system.register_user(username)
            
        elif choice == '2':
            if not os.path.exists(USERS_PATH):
                print("No users registered yet.")
                continue
                
            with open(USERS_PATH, 'r') as f:
                users = json.load(f)
                
            if not users:
                print("No users registered yet.")
                continue
                
            print("\nAvailable users:")
            for i, user in enumerate(users.keys()):
                print(f"{i+1}. {user}")
                
            user_idx = input("Select user (number): ")
            try:
                user_idx = int(user_idx) - 1
                username = list(users.keys())[user_idx]
                system.select_user(username)
            except (ValueError, IndexError):
                print("Invalid selection.")
                
        elif choice == '3':
            if not system.active_user:
                print("No user selected. Please select a user first.")
                continue
                
            member_name = input("Enter family member name: ")
            system.register_family_member(member_name)
            
        elif choice == '4':
            if not system.active_user:
                print("No user selected. Please select a user first.")
                continue
                
            # Show available family members
            family_members = [member['name'] for member in system.users[system.active_user]["family_members"]]
            
            if not family_members:
                print("No family members registered yet. Please add family members first.")
                continue
                
            print("\nFamily members:")
            for i, member in enumerate(family_members):
                print(f"{i+1}. {member}")
                
            member_idx = input("Select family member (number): ")
            
            try:
                member_idx = int(member_idx) - 1
                member_name = family_members[member_idx]
                num_samples = input("Number of voice samples to record (default: 3): ")
                num_samples = int(num_samples) if num_samples.isdigit() else 3
                system.record_family_voice_samples(member_name, num_samples)
            except (ValueError, IndexError):
                print("Invalid selection.")
                
        elif choice == '5':
            if not system.active_user:
                print("No user selected. Please select a user first.")
                continue
                
            print("Training model... This may take a few minutes.")
            success = system.train_model()
            
            if success:
                print("Model trained successfully.")
                
                # Optimize for the current device
                system.optimize_for_device()
            
        elif choice == '6':
            if not system.active_user:
                print("No user selected. Please select a user first.")
                continue
                
            print("Recording audio for recognition...")
            audio_path = system.record_audio(duration=3)
            result = system.identify_speaker(audio_path)
            
            if result:
                speaker, confidence = result
                print(f"Identified speaker: {speaker} (confidence: {confidence:.2f})")
            else:
                print("Could not identify the speaker.")
                
        elif choice == '7':
            if not system.active_user:
                print("No user selected. Please select a user first.")
                continue
                
            try:
                system.start_real_time_recognition()
            except KeyboardInterrupt:
                system.stop_real_time_recognition()
                
        elif choice == '8':
            if not system.active_user:
                print("No user selected. Please select a user first.")
                continue
                
            system.tune_recognition_parameters()
            
        elif choice == '9':
            if not system.active_user:
                print("No user selected. Please select a user first.")
                continue
                
            export_path = input("Enter export path (or press Enter for default): ")
            if not export_path:
                export_path = None
                
            system.export_model(export_path)
            
        elif choice == '10':
            if not system.active_user or not system.model:
                print("No model loaded. Please select a user and train or load a model first.")
                continue
                
            system.optimize_for_device()
            
        elif choice == '0':
            system.cleanup()
            print("Exiting. Thank you for using the Family Voice Recognition System!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()