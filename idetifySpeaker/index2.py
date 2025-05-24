import os
import time
import threading
import json
import numpy as np
import pyaudio
import wave
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from collections import deque, Counter
from datetime import datetime
import queue
from typing import Optional, Dict, List, Tuple, Union
import noisereduce as nr
import onnxruntime as ort
from multiprocessing import Pool, cpu_count

# Constants
CHUNK = 1024  # Reduced from 1024 to 512 for lower latency
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
FRAME_SIZE = 512
HOP_SIZE = 256
N_MFCC = 40
N_FFT = 2048
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SILENCE_THRESHOLD = 0.03
DATA_DIR = "family_voice_data"
MODEL_PATH = "family_voice_model.pth"
USERS_PATH = "users.json"
MIN_CONFIDENCE = 0.6  # Minimum confidence threshold for speaker identification
REALTIME_BUFFER_SECONDS = 1.5  # Buffer size for real-time processing (in seconds)
MIN_SPEECH_FRAMES = 15  # Minimum consecutive frames to consider as speech

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

class SpeakerRecognitionModel(nn.Module):
    """Optimized neural network model for speaker recognition."""
    def __init__(self, input_size: int, hidden_size: int, num_speakers: int):
        super(SpeakerRecognitionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_speakers)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class SpectralFeatureExtractor:
    """Optimized feature extraction with noise reduction and parallel processing."""
    def __init__(self):
        self.n_mfcc = N_MFCC
        self.n_fft = N_FFT
        self.hop_length = HOP_SIZE
        self.pool = Pool(processes=max(1, cpu_count() - 1))

    def extract_features(self, audio_path: Optional[str] = None, audio_data: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Extract features from audio file or numpy array."""
        try:
            if audio_path:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            elif audio_data is not None:
                y = audio_data
                sr = SAMPLE_RATE
            else:
                return None
                
            # Parallel processing of audio chunks
            chunks = self._split_audio(y)
            features = self.pool.map(self._process_chunk, chunks)
            features = [f for f in features if f is not None]
            
            if not features:
                return None
                
            return np.mean(features, axis=0)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def _split_audio(self, y: np.ndarray) -> List[np.ndarray]:
        """Split audio into chunks for parallel processing."""
        chunk_size = len(y) // 4
        return [y[i*chunk_size:(i+1)*chunk_size] for i in range(4)] if len(y) > 10000 else [y]

    def _process_chunk(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Process a single audio chunk."""
        try:
            # Noise reduction
            y = nr.reduce_noise(y=y, sr=SAMPLE_RATE)
            
            # Remove silence
            y = self._remove_silence(y)
            if len(y) == 0:
                return None
                
            # Extract features in parallel
            mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=self.n_mfcc, 
                                        n_fft=self.n_fft, hop_length=self.hop_length)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, 
                                                                n_fft=self.n_fft, hop_length=self.hop_length)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            
            # Compute statistics
            features = []
            for feature in [mfccs, spectral_centroid, zero_crossing_rate]:
                features.append(np.mean(feature, axis=1))
                features.append(np.std(feature, axis=1))
                
            return np.concatenate(features)
        except:
            return None

    def _remove_silence(self, y: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> np.ndarray:
        """Remove silence from audio using more advanced VAD."""
        intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
        if not intervals.size:
            return np.array([])
        return np.concatenate([y[start:end] for start, end in intervals])

class VoiceActivityDetector:
    """Improved voice activity detection with adaptive thresholding."""
    def __init__(self, threshold: float = SILENCE_THRESHOLD):
        self.threshold = threshold
        self.energy_history = deque(maxlen=100)
        self.speech_prob = 0
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Detect if audio chunk contains speech with adaptive threshold."""
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Calculate energy
        energy = np.mean(audio_array ** 2)
        self.energy_history.append(energy)
        
        # Adaptive threshold based on recent history
        if len(self.energy_history) > 10:
            avg_energy = np.mean(self.energy_history)
            adaptive_threshold = max(self.threshold, avg_energy * 1.5)
            
            # Simple speech probability
            if energy > adaptive_threshold:
                self.speech_prob = min(1.0, self.speech_prob + 0.1)
                return self.speech_prob > 0.5
            else:
                self.speech_prob = max(0.0, self.speech_prob - 0.05)
                return self.speech_prob > 0.5
                
        return energy > self.threshold

class SpeakerRecognitionSystem:
    """Optimized speaker recognition system with ONNX support and better real-time processing."""
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
        
        # Real-time processing
        self.audio_buffer = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        self.current_speakers = {}
        self.speaker_history = deque(maxlen=5)
        self.ort_session = None  # ONNX runtime session
        
    def _init_audio(self) -> None:
        """Initialize PyAudio with optimized settings."""
        self.p = pyaudio.PyAudio()
        
    def load_users(self) -> None:
        """Load users from json file with error handling."""
        try:
            if os.path.exists(USERS_PATH):
                with open(USERS_PATH, 'r') as f:
                    self.users = json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
            self.users = {}
            
    def save_users(self) -> None:
        """Save users to json file with atomic write."""
        try:
            temp_path = USERS_PATH + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(self.users, f)
            os.replace(temp_path, USERS_PATH)
        except Exception as e:
            print(f"Error saving users: {e}")
            
    def register_user(self, username: str) -> bool:
        """Register a new user with validation."""
        if not username.strip():
            print("Username cannot be empty.")
            return False
            
        if username in self.users:
            print(f"User {username} already exists.")
            return False
            
        self.users[username] = {"family_members": []}
        self.save_users()
        self.active_user = username
        print(f"User {username} registered successfully.")
        return True
    
    def register_family_member(self, member_name: str) -> bool:
        """Register a new family member with validation."""
        if not self.active_user:
            print("No active user. Please select or register a user first.")
            return False
            
        if not member_name.strip():
            print("Member name cannot be empty.")
            return False
            
        existing_members = [m['name'] for m in self.users[self.active_user]["family_members"]]
        if member_name in existing_members:
            print(f"Family member {member_name} already exists for {self.active_user}.")
            return False
            
        self.users[self.active_user]["family_members"].append({
            "name": member_name, 
            "recordings": []
        })
        self.save_users()
        print(f"Family member {member_name} registered for {self.active_user}.")
        return True
    
    def select_user(self, username: str) -> bool:
        """Select an existing user with validation."""
        if username not in self.users:
            print(f"User {username} does not exist.")
            return False
            
        self.active_user = username
        self.family_members = [m['name'] for m in self.users[self.active_user]["family_members"]]
        print(f"Selected user: {username}")
        print(f"Family members: {', '.join(self.family_members)}")
        
        # Try to load ONNX model first for better performance
        onnx_path = f"{username}_model.onnx"
        if os.path.exists(onnx_path):
            self._load_onnx_model(onnx_path)
            print(f"Loaded optimized ONNX model for {username}")
        else:
            model_path = f"{username}_model.pth"
            if os.path.exists(model_path):
                self._load_model(model_path)
                print(f"Loaded PyTorch model for {username}")
        
        return True
    
    def record_audio(self, member_name: Optional[str] = None, duration: int = RECORD_SECONDS, 
                    save_path: Optional[str] = None) -> Optional[str]:
        """Record audio with error handling."""
        if not self.active_user and member_name:
            print("No active user. Please select or register a user first.")
            return None
            
        if member_name and member_name not in [m['name'] for m in self.users[self.active_user]["family_members"]]:
            print(f"Family member {member_name} does not exist for {self.active_user}.")
            return None
            
        print(f"Recording {'sample' if member_name else 'audio'} for {duration} seconds...")
        
        try:
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                               input=True, frames_per_buffer=CHUNK)
            frames = []
            
            for _ in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if member_name:
                    member_dir = os.path.join(DATA_DIR, self.active_user, member_name)
                    os.makedirs(member_dir, exist_ok=True)
                    save_path = os.path.join(member_dir, f"{timestamp}.wav")
                else:
                    save_path = os.path.join(DATA_DIR, f"temp_{timestamp}.wav")
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with wave.open(save_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            
            if member_name:
                for member in self.users[self.active_user]["family_members"]:
                    if member["name"] == member_name:
                        member["recordings"].append(save_path)
                        self.save_users()
                        break
            
            return save_path
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    def record_family_voice_samples(self, member_name: str, num_samples: int = 3) -> bool:
        """Record multiple voice samples with progress tracking."""
        if not self.active_user:
            print("No active user. Please select or register a user first.")
            return False
            
        if not any(m['name'] == member_name for m in self.users[self.active_user]["family_members"]):
            print(f"Family member {member_name} not found.")
            return False
            
        print(f"Recording {num_samples} voice samples for {member_name}")
        member_dir = os.path.join(DATA_DIR, self.active_user, member_name)
        os.makedirs(member_dir, exist_ok=True)
        
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}")
            input("Press Enter to start recording...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(member_dir, f"sample_{i+1}_{timestamp}.wav")
            
            if not self.record_audio(member_name, duration=5, save_path=file_path):
                print(f"Failed to record sample {i+1}")
                continue
                
            if i < num_samples - 1:
                time.sleep(1)  # Short pause between samples
        
        print(f"\nCompleted recording {num_samples} samples for {member_name}")
        return True
    
    def extract_features_for_training(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract features with parallel processing and progress tracking."""
        if not self.active_user:
            print("No active user selected.")
            return None, None
            
        features = []
        labels = []
        total_recordings = sum(len(m['recordings']) for m in self.users[self.active_user]["family_members"])
        processed = 0
        
        print(f"Extracting features from {total_recordings} recordings...")
        
        for member in self.users[self.active_user]["family_members"]:
            member_name = member["name"]
            recordings = member["recordings"]
            
            if not recordings:
                print(f"No recordings found for {member_name}.")
                continue
                
            print(f"Processing {len(recordings)} recordings for {member_name}...")
            
            # Process recordings in parallel
            results = []
            for recording in recordings:
                if os.path.exists(recording):
                    results.append(self.feature_extractor.extract_features(recording))
                    processed += 1
                    print(f"\rProgress: {processed}/{total_recordings} ({processed/total_recordings*100:.1f}%)", end="")
                else:
                    print(f"\nWarning: Recording file {recording} not found.")
            
            # Filter valid results
            valid_features = [f for f in results if f is not None]
            features.extend(valid_features)
            labels.extend([member_name] * len(valid_features))
            
        if not features:
            print("\nNo valid features extracted.")
            return None, None
            
        self.feature_size = len(features[0])
        self.label_encoder.fit(labels)
        numerical_labels = self.label_encoder.transform(labels)
        
        print(f"\nExtracted features for {len(set(labels))} family members.")
        return np.array(features), numerical_labels
    
    def train_model(self) -> bool:
        """Train model with early stopping and learning rate scheduling."""
        if not self.active_user:
            print("No active user selected.")
            return False
            
        X, y = self.extract_features_for_training()
        if X is None or len(X) == 0:
            return False
            
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        y_tensor = torch.LongTensor(y).to(DEVICE)
        
        # Initialize model
        num_speakers = len(self.label_encoder.classes_)
        self.model = SpeakerRecognitionModel(self.feature_size, 256, num_speakers).to(DEVICE)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        batch_size = 16
        num_epochs = 100
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop with early stopping
        print("Training model...")
        for epoch in range(num_epochs):
            self.model.train()
            permutation = torch.randperm(X_tensor.size()[0])
            epoch_loss = 0
            
            for i in range(0, X_tensor.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_tensor[indices], y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (X_tensor.size()[0] / batch_size)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                self._save_model(f"{self.active_user}_model.pth")
                # Also export to ONNX for faster inference
                self._export_to_onnx(f"{self.active_user}_model.onnx")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        print("Training completed.")
        return True
    
    def _export_to_onnx(self, path: str) -> None:
        """Export model to ONNX format for faster inference."""
        if not self.model:
            return
            
        dummy_input = torch.randn(1, self.feature_size).to(DEVICE)
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
    
    def _load_onnx_model(self, path: str) -> bool:
        """Load ONNX model for faster inference."""
        try:
            self.ort_session = ort.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            
            # Get model metadata
            meta = self.ort_session.get_modelmeta()
            self.feature_size = self.ort_session.get_inputs()[0].shape[1]
            self.label_encoder.classes_ = json.loads(meta.custom_metadata_map['classes'])
            return True
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return False
    
    def _save_model(self, path: str) -> None:
        """Save model with metadata."""
        if not self.model:
            return
            
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'feature_size': self.feature_size,
            'num_speakers': len(self.label_encoder.classes_),
            'label_encoder_classes': list(self.label_encoder.classes_)
        }
        torch.save(model_state, path)
    
    def _load_model(self, path: str) -> bool:
        """Load PyTorch model with error handling."""
        try:
            model_state = torch.load(path, map_location=DEVICE)
            self.feature_size = model_state['feature_size']
            num_speakers = model_state['num_speakers']
            
            self.model = SpeakerRecognitionModel(self.feature_size, 256, num_speakers).to(DEVICE)
            self.model.load_state_dict(model_state['model_state_dict'])
            self.model.eval()
            
            self.label_encoder.classes_ = np.array(model_state['label_encoder_classes'])
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def identify_speaker(self, audio_path: Optional[str] = None, audio_data: Optional[np.ndarray] = None) -> Optional[Dict[str, Union[str, float]]]:
        """Identify speaker with optimized inference."""
        if not self.active_user:
            print("No active user selected.")
            return None
            
        if not self.model and not self.ort_session:
            model_path = f"{self.active_user}_model.pth"
            if not self._load_model(model_path):
                print("No trained model found.")
                return None
        
        # Record audio if no path/data provided
        if not audio_path and audio_data is None:
            print("Recording audio for identification...")
            audio_path = self.record_audio(duration=3)
            if not audio_path:
                return None
                
        # Extract features
        feature_vector = self.feature_extractor.extract_features(audio_path=audio_path, audio_data=audio_data)
        if feature_vector is None:
            print("Feature extraction failed.")
            return None
            
        # Use ONNX if available for faster inference
        if self.ort_session:
            input_name = self.ort_session.get_inputs()[0].name
            output_name = self.ort_session.get_outputs()[0].name
            ort_inputs = {input_name: feature_vector.astype(np.float32).reshape(1, -1)}
            ort_outs = self.ort_session.run([output_name], ort_inputs)[0]
            output = torch.from_numpy(ort_outs)
        else:
            feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = self.model(feature_tensor)
                
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        speaker_name = self.label_encoder.inverse_transform([predicted.item()])[0]
        confidence_val = confidence.item()
        
        return {
            "name": speaker_name if confidence_val > MIN_CONFIDENCE else "Unknown",
            "confidence": confidence_val
        }
    
    def start_real_time_identification(self) -> bool:
        """Start real-time identification with optimized processing."""
        if not self.active_user:
            print("No active user selected.")
            return False
            
        if not self.model and not self.ort_session:
            model_path = f"{self.active_user}_model.pth"
            if not self._load_model(model_path):
                print("No trained model found.")
                return False
                
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_audio_stream, daemon=True)
        self.processing_thread.start()
        self._start_audio_stream()
        return True
    
    def _start_audio_stream(self) -> None:
        """Start audio stream with optimized callback."""
        def callback(in_data, frame_count, time_info, status):
            self.audio_buffer.put(in_data)
            return (in_data, pyaudio.paContinue)
            
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                input=True, frames_per_buffer=CHUNK,
                                stream_callback=callback)
                                
        print("Real-time speaker identification started. Press Ctrl+C to stop.")
        
        try:
            while self.is_processing:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.is_processing = False
            self.stream.stop_stream()
            self.stream.close()
            
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
    
    def _process_audio_stream(self) -> None:
        """Optimized real-time audio processing."""
        buffer = []
        speech_counter = 0
        last_speaker = None
        last_change_time = time.time()
        
        while self.is_processing:
            if self.audio_buffer.empty():
                time.sleep(0.01)
                continue
                
            data = self.audio_buffer.get()
            buffer.append(data)
            
            # Keep buffer size reasonable (about 2 seconds)
            if len(buffer) > int(2 * RATE / CHUNK):
                buffer = buffer[-int(1 * RATE / CHUNK):]  # Keep last 1 second
                
            # Voice activity detection
            if self.vad.is_speech(data):
                speech_counter += 1
            else:
                speech_counter = max(0, speech_counter - 1)
            
            # Process when we have enough speech
            if speech_counter >= MIN_SPEECH_FRAMES and len(buffer) >= int(REALTIME_BUFFER_SECONDS * RATE / CHUNK):
                # Convert buffer to numpy array
                audio_data = np.frombuffer(b''.join(buffer[-int(REALTIME_BUFFER_SECONDS * RATE / CHUNK):]), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Identify speaker
                result = self.identify_speaker(audio_data=audio_data)
                
                if result and result["name"] != "Unknown":
                    self.speaker_history.append(result["name"])
                    
                    # Check for consistent predictions
                    if len(self.speaker_history) >= 3:
                        counts = Counter(self.speaker_history)
                        common = counts.most_common(1)[0]
                        
                        if common[1] >= len(self.speaker_history) / 2:
                            if last_speaker != common[0]:
                                last_speaker = common[0]
                                last_change_time = time.time()
                                print(f"\nSpeaker identified: {last_speaker} (confidence: {result['confidence']:.2f})")
                            else:
                                # Show we're still tracking
                                print(".", end="", flush=True)
                
                # Reset buffer but keep some context
                buffer = buffer[-int(0.5 * RATE / CHUNK):]
                speech_counter = 0
    
    def stop_real_time_identification(self) -> None:
        """Stop real-time processing."""
        self.is_processing = False
        print("Real-time identification stopped.")
    
    def list_users(self) -> None:
        """List registered users."""
        if not self.users:
            print("No users registered.")
            return
            
        print("Registered users:")
        for username, data in self.users.items():
            print(f"- {username} ({len(data['family_members'])} family members)")
    
    def list_family_members(self) -> None:
        """List family members with details."""
        if not self.active_user:
            print("No active user selected.")
            return
            
        members = self.users[self.active_user]["family_members"]
        if not members:
            print(f"No family members for {self.active_user}.")
            return
            
        print(f"Family members for {self.active_user}:")
        for member in members:
            print(f"- {member['name']} ({len(member['recordings'])} samples)")
    
    def close(self) -> None:
        """Clean up resources."""
        self.feature_extractor.pool.close()
        self.p.terminate()
        if self.is_processing:
            self.stop_real_time_identification()

def main():
    """Main application loop."""
    print("=" * 50)
    print("Enhanced Family Voice Recognition System")
    print("=" * 50)
    
    system = SpeakerRecognitionSystem()
    
    menu_options = {
        '1': ("Register new user", lambda: system.register_user(input("Enter username: "))),
        '2': ("Select existing user", lambda: system.select_user(input("Enter username: "))),
        '3': ("Register family member", lambda: system.register_family_member(input("Enter member name: "))),
        '4': ("Record voice samples", lambda: (
            system.list_family_members(),
            system.record_family_voice_samples(
                input("Enter member name: "),
                int(input("Number of samples (3-5 recommended): "))
            ) if system.active_user else print("No active user."))),
        '5': ("Train model", lambda: system.train_model() if system.active_user else print("No active user.")),
        '6': ("Identify speaker", lambda: (
            print(f"Result: {system.identify_speaker()}") if system.active_user else print("No active user."))),
        '7': ("Real-time identification", lambda: (
            system.start_real_time_identification() if system.active_user else print("No active user."))),
        '8': ("List users", system.list_users),
        '9': ("List family members", system.list_family_members),
        '0': ("Exit", system.close)
    }
    
    while True:
        print("\nMenu:")
        for key, (desc, _) in menu_options.items():
            print(f"{key}. {desc}")
        
        choice = input("\nEnter choice: ")
        if choice in menu_options:
            if choice == '0':
                menu_options[choice][1]()
                break
            menu_options[choice][1]()
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()