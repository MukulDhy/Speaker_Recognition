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
from collections import deque
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import split_on_silence
import queue

# Constants
CHUNK = 1024
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

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Neural Network Model for Speaker Recognition
class SpeakerRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_speakers):
        super(SpeakerRecognitionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_speakers)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class SpectralFeatureExtractor:
    def __init__(self):
        self.n_mfcc = N_MFCC
        self.n_fft = N_FFT
        self.hop_length = HOP_SIZE
        
    def extract_features(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Remove silent parts
        y = self._remove_silence(y)
        
        if len(y) == 0:  # If no sound was detected
            return None
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Extract temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        
        # Compute statistics of features
        features = []
        for feature in [mfccs, spectral_centroid, spectral_contrast, spectral_bandwidth, zero_crossing_rate]:
            features.append(np.mean(feature, axis=1))
            features.append(np.std(feature, axis=1))
        
        # Flatten and concatenate all features
        combined_features = np.concatenate(features)
        
        return combined_features
    
    def _remove_silence(self, y, threshold=SILENCE_THRESHOLD):
        # Calculate amplitude envelope
        amplitude_envelope = np.abs(y)
        
        # Find indices where amplitude is above threshold
        mask = amplitude_envelope > threshold
        
        # If all is silence, return empty array
        if not np.any(mask):
            return np.array([])
        
        # Keep only non-silent parts
        y_filtered = y[mask]
        
        return y_filtered

class VoiceActivityDetector:
    def __init__(self, threshold=SILENCE_THRESHOLD):
        self.threshold = threshold
        self.window_size = 30  # frames to check
        self.min_speech_frames = 10  # minimum consecutive frames to consider as speech
        
    def is_speech(self, audio_chunk):
        """Detect if audio chunk contains speech"""
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize
        
        # Check if audio level is above threshold
        if np.mean(np.abs(audio_array)) > self.threshold:
            return True
        return False

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
        
        # For real-time processing
        self.audio_buffer = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        self.current_speakers = {}  # Store currently detected speakers
        self.speaker_history = deque(maxlen=5)  # Store recent speaker predictions
        
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
        with open(USERS_PATH, 'w') as f:
            json.dump(self.users, f)
            
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
        """Record audio for the specified duration"""
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
        
        # Record audio
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Visual indicator of recording progress
            if i % 10 == 0:
                print(".", end="", flush=True)
        
        print("\nFinished recording.")
        
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
        """Record multiple voice samples for a family member"""
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
        
        for i in range(num_samples):
            print(f"Sample {i+1}/{num_samples}")
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
                time.sleep(1)
        
        print(f"Completed recording {num_samples} samples for {member_name}")
        return True
    
    def extract_features_for_training(self):
        """Extract features from all recordings for training"""
        if not self.active_user:
            print("No active user selected.")
            return None, None
            
        features = []
        labels = []
        
        print("Extracting features for training...")
        
        for member in self.users[self.active_user]["family_members"]:
            member_name = member["name"]
            recordings = member["recordings"]
            
            if not recordings:
                print(f"No recordings found for {member_name}. Please record voice samples first.")
                continue
                
            print(f"Processing {len(recordings)} recordings for {member_name}...")
            
            for recording in recordings:
                # Check if file exists
                if not os.path.exists(recording):
                    print(f"Warning: Recording file {recording} not found. Skipping.")
                    continue
                    
                # Extract features
                feature_vector = self.feature_extractor.extract_features(recording)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(member_name)
                else:
                    print(f"Warning: Could not extract features from {recording}. Skipping.")
        
        if not features:
            print("No features extracted. Make sure all family members have valid recordings.")
            return None, None
            
        # Store the feature size for model initialization
        self.feature_size = len(features[0])
        
        # Convert labels to numerical representation
        self.label_encoder.fit(labels)
        numerical_labels = self.label_encoder.transform(labels)
        
        print(f"Extracted features for {len(set(labels))} family members.")
        return np.array(features), numerical_labels
    
    def train_model(self):
        """Train the speaker recognition model"""
        if not self.active_user:
            print("No active user selected.")
            return False
            
        # Extract features for training
        X, y = self.extract_features_for_training()
        if X is None or len(X) == 0:
            return False
            
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Initialize model
        num_speakers = len(self.label_encoder.classes_)
        self.model = SpeakerRecognitionModel(self.feature_size, 128, num_speakers).to(DEVICE)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        num_epochs = 100
        batch_size = 8
        
        # Training loop
        print("Training model...")
        for epoch in range(num_epochs):
            # Simple batch creation
            permutation = torch.randperm(X_tensor.size()[0])
            
            for i in range(0, X_tensor.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
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
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Save the model
        model_path = f"{self.active_user}_model.pth"
        self._save_model(model_path)
        print(f"Training completed. Model saved to {model_path}")
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
        self.model.eval()
        
        # Restore label encoder
        self.label_encoder.classes_ = model_state['label_encoder_classes']
        
        return True
    
    def identify_speaker(self, audio_path=None):
        """Identify the speaker in the recorded audio"""
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
        
        # Extract features
        feature_vector = self.feature_extractor.extract_features(audio_path)
        
        if feature_vector is None:
            print("Could not extract features. Please try again with clearer audio.")
            return None
            
        # Convert to tensor
        feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(feature_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        speaker_idx = predicted.item()
        confidence_val = confidence.item()
        speaker_name = self.label_encoder.inverse_transform([speaker_idx])[0]
        
        # Only return prediction if confidence is high enough
        if confidence_val > 0.6:
            return {
                "name": speaker_name,
                "confidence": confidence_val
            }
        else:
            return {
                "name": "Unknown",
                "confidence": confidence_val
            }
    
    def start_real_time_identification(self):
        """Start real-time speaker identification"""
        if not self.active_user:
            print("No active user selected.")
            return False
            
        if not self.model:
            model_path = f"{self.active_user}_model.pth"
            if not self._load_model(model_path):
                print("No trained model found. Please train the model first.")
                return False
                
        # Start processing thread
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start audio stream
        self._start_audio_stream()
        
        return True
    
    def _start_audio_stream(self):
        """Start audio stream for real-time processing"""
        def callback(in_data, frame_count, time_info, status):
            self.audio_buffer.put(in_data)
            return (in_data, pyaudio.paContinue)
            
        # Open audio stream
        self.stream = self.p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK,
                                stream_callback=callback)
                                
        print("Real-time speaker identification started. Press Ctrl+C to stop.")
        
        try:
            while self.is_processing:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping real-time identification...")
            
        # Stop and close stream
        self.is_processing = False
        self.stream.stop_stream()
        self.stream.close()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
    
    def _process_audio_stream(self):
        """Process audio stream and identify speakers"""
        audio_buffer = []
        silent_frames = 0
        speaking_frames = 0
        current_speaker = None
        
        while self.is_processing:
            # Get audio data from buffer
            if self.audio_buffer.empty():
                time.sleep(0.01)
                continue
                
            data = self.audio_buffer.get()
            audio_buffer.append(data)
            
            # Check if audio contains speech
            is_speech = self.vad.is_speech(data)
            
            if is_speech:
                speaking_frames += 1
                silent_frames = 0
            else:
                silent_frames += 1
                speaking_frames = 0
            
            # If enough consecutive speech frames, process for identification
            if speaking_frames > 10:
                # Check if we have enough audio data (about 1.5 seconds)
                if len(audio_buffer) >= int(1.5 * RATE / CHUNK):
                    # Save buffer to temp file
                    temp_file = os.path.join(DATA_DIR, f"temp_rt_{int(time.time())}.wav")
                    wf = wave.open(temp_file, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(audio_buffer[-int(1.5 * RATE / CHUNK):]))
                    wf.close()
                    
                    # Identify speaker
                    result = self.identify_speaker(temp_file)
                    
                    # Remove temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    
                    # Update current speaker if valid result
                    if result and result["name"] != "Unknown":
                        # Add to speaker history
                        self.speaker_history.append(result["name"])
                        
                        # Only update current speaker if consistent predictions
                        if len(self.speaker_history) >= 3:
                            # Get most common prediction in history
                            from collections import Counter
                            counter = Counter(self.speaker_history)
                            most_common = counter.most_common(1)[0]
                            
                            # Update if prediction is consistent (appears in more than half of recent frames)
                            if most_common[1] >= len(self.speaker_history) / 2:
                                if current_speaker != most_common[0]:
                                    current_speaker = most_common[0]
                                    print(f"Speaker identified: {current_speaker}")
                    
                    # Reset buffer but keep some recent frames
                    audio_buffer = audio_buffer[-int(0.5 * RATE / CHUNK):]
                    speaking_frames = 0
            
            # If silent for some time, clear current speaker
            if silent_frames > 20 and current_speaker:
                print("No speech detected.")
                current_speaker = None
                self.speaker_history.clear()
                audio_buffer = []
    
    def stop_real_time_identification(self):
        """Stop real-time speaker identification"""
        self.is_processing = False
        print("Real-time speaker identification stopped.")
    
    def list_users(self):
        """List registered users"""
        if not self.users:
            print("No users registered.")
            return
            
        print("Registered users:")
        for username in self.users:
            family_count = len(self.users[username]["family_members"])
            print(f"- {username} ({family_count} family members)")
    
    def list_family_members(self):
        """List family members for active user"""
        if not self.active_user:
            print("No active user selected.")
            return
            
        family_members = self.users[self.active_user]["family_members"]
        if not family_members:
            print(f"No family members registered for {self.active_user}.")
            return
            
        print(f"Family members for {self.active_user}:")
        for member in family_members:
            recording_count = len(member["recordings"])
            print(f"- {member['name']} ({recording_count} recordings)")
    
    def close(self):
        """Clean up resources"""
        self.p.terminate()
        

def main():
    """Main function to run the speaker recognition system"""
    print("=" * 50)
    print("Family Voice Recognition System")
    print("=" * 50)
    
    system = SpeakerRecognitionSystem()
    
    while True:
        print("\nOptions:")
        print("1. Register new user (family)")
        print("2. Select existing user")
        print("3. Register family member")
        print("4. Record voice samples for family member")
        print("5. Train voice recognition model")
        print("6. Identify speaker (one-time)")
        print("7. Start real-time speaker identification")
        print("8. List registered users")
        print("9. List family members for current user")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ")
        
        if choice == '1':
            username = input("Enter username for new family: ")
            system.register_user(username)
            
        elif choice == '2':
            system.list_users()
            username = input("Enter username to select: ")
            system.select_user(username)
            
        elif choice == '3':
            if not system.active_user:
                print("No active user selected. Please select a user first.")
                continue
                
            member_name = input("Enter name of family member: ")
            system.register_family_member(member_name)
            
        elif choice == '4':
            if not system.active_user:
                print("No active user selected. Please select a user first.")
                continue
                
            system.list_family_members()
            member_name = input("Enter name of family member to record: ")
            num_samples = int(input("Enter number of voice samples to record (recommended: 3-5): "))
            system.record_family_voice_samples(member_name, num_samples)
            
        elif choice == '5':
            if not system.active_user:
                print("No active user selected. Please select a user first.")
                continue
                
            system.train_model()
            
        elif choice == '6':
            if not system.active_user:
                print("No active user selected. Please select a user first.")
                continue
                
            result = system.identify_speaker()
            if result:
                print(f"Identified speaker: {result['name']} (confidence: {result['confidence']:.2f})")
            else:
                print("Could not identify speaker.")
                
        elif choice == '7':
            if not system.active_user:
                print("No active user selected. Please select a user first.")
                continue
                
            try:
                system.start_real_time_identification()
            except KeyboardInterrupt:
                system.stop_real_time_identification()
                
        elif choice == '8':
            system.list_users()
            
        elif choice == '9':
            system.list_family_members()
            
        elif choice == '0':
            print("Exiting program.")
            system.close()
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()