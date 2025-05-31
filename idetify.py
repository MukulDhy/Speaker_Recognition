import os
import time
import numpy as np
import pyaudio
import wave
import pickle
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3
N_MFCC = 32
N_FFT = 1024
HOP_SIZE = 256
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SILENCE_THRESHOLD = 0.02
MODEL_PATH = "mukul2_model.pth"

# Neural Network Model (same as training)
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

    def optimize_for_inference(self):
        """Optimize model for inference"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

# Feature Extractor
class FeatureExtractor:
    def __init__(self):
        self.n_mfcc = N_MFCC
        self.n_fft = N_FFT
        self.hop_length = HOP_SIZE
        
    def extract_features(self, audio_path):
        """Extract features from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, res_type='kaiser_fast')
            
            # Remove silence
            y = self._remove_silence(y)
            
            if len(y) == 0:
                print("No speech detected in audio")
                return None
            
            # Extract features
            features = []
            
            # MFCCs (primary feature)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                       n_fft=self.n_fft, hop_length=self.hop_length)
            features.append(np.mean(mfccs, axis=1))
            features.append(np.std(mfccs, axis=1))
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                                n_fft=self.n_fft, 
                                                                hop_length=self.hop_length)
            features.append(np.mean(spectral_centroid, axis=1))
            
            # Zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            features.append(np.mean(zero_crossing_rate, axis=1))
            
            # Combine all features
            combined_features = np.concatenate(features)
            
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _remove_silence(self, y, threshold=SILENCE_THRESHOLD):
        """Remove silence from audio"""
        try:
            # Calculate RMS energy
            frame_length = 1024
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Find voiced frames
            voiced_frames = np.where(rms > threshold)[0]
            
            if len(voiced_frames) == 0:
                return np.array([])
            
            # Get voiced segments
            voiced_indexes = librosa.frames_to_samples(voiced_frames, hop_length=hop_length)
            voiced_indexes = np.minimum(voiced_indexes, len(y) - 1)
            
            if len(voiced_indexes) > 10:
                start_idx = voiced_indexes[0]
                end_idx = voiced_indexes[-1]
                return y[start_idx:end_idx+1]
            else:
                return np.array([])
                
        except Exception as e:
            print(f"Error removing silence: {e}")
            return y

# Speaker Identifier
class SpeakerIdentifier:
    def __init__(self, model_path=MODEL_PATH, confidence_threshold=0.6):
        self.model = None
        self.label_encoder = None
        self.feature_extractor = FeatureExtractor()
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Load the trained model
        self.load_model()
        
    def load_model(self):
        """Load the pre-trained model"""
        if not os.path.exists(self.model_path):
            print(f"Model file {self.model_path} not found!")
            print("Please ensure you have a trained model file.")
            return False
            
        try:
            # Load model state
            model_state = torch.load(self.model_path, map_location=DEVICE)
            
            # Extract model parameters
            feature_size = model_state['feature_size']
            num_speakers = model_state['num_speakers']
            label_encoder_classes = model_state['label_encoder_classes']
            
            # Initialize model
            self.model = SpeakerRecognitionModel(feature_size, 128, num_speakers).to(DEVICE)
            self.model.load_state_dict(model_state['model_state_dict'])
            self.model.optimize_for_inference()
            
            # Initialize label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = label_encoder_classes
            
            print(f"Model loaded successfully!")
            print(f"Recognized speakers: {', '.join(label_encoder_classes)}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def record_audio(self, duration=RECORD_SECONDS):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds... Please speak now!")
        
        # Setup recording
        stream = self.p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        
        frames = []
        
        # Record with visual feedback
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Simple progress indicator
            if i % 5 == 0:
                progress = int((i / (RATE / CHUNK * duration)) * 20)
                print(f"\rRecording: [{'#' * progress}{'-' * (20-progress)}]", end="", flush=True)
        
        print("\nRecording finished!")
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        
        # Save temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_recording_{timestamp}.wav"
        
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return temp_filename
    
    def identify_speaker(self, audio_path=None):
        """Identify speaker from audio file or record new audio"""
        if not self.model:
            print("No model loaded. Cannot identify speaker.")
            return None
        
        # Record audio if no path provided
        if not audio_path:
            audio_path = self.record_audio()
        
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} not found.")
            return None
        
        print("Processing audio...")
        start_time = time.time()
        
        # Extract features
        features = self.feature_extractor.extract_features(audio_path)
        
        if features is None:
            print("Could not extract features from audio.")
            # Clean up temporary file
            if audio_path.startswith("temp_recording_"):
                try:
                    os.remove(audio_path)
                except:
                    pass
            return None
        
        # Make prediction
        feature_tensor = torch.FloatTensor(features).to(DEVICE)
        feature_tensor = feature_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = self.model(feature_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            prediction = prediction.item()
        
        # Get speaker name
        predicted_speaker = self.label_encoder.inverse_transform([prediction])[0]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up temporary file
        if audio_path.startswith("temp_recording_"):
            try:
                os.remove(audio_path)
            except:
                pass
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            print(f"Low confidence identification: {predicted_speaker} ({confidence:.2f})")
            print("Speaker identification uncertain. Try speaking more clearly or closer to the microphone.")
            return None
        
        print(f"Speaker identified: {predicted_speaker}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        return {
            'speaker': predicted_speaker,
            'confidence': confidence,
            'processing_time': processing_time
        }
    
    def identify_from_file(self, audio_file_path):
        """Identify speaker from existing audio file"""
        return self.identify_speaker(audio_file_path)
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'p'):
            self.p.terminate()

def main():
    """Main function to run speaker identification"""
    print("Speaker Identification System")
    print("=" * 30)
    
    # Initialize identifier
    identifier = SpeakerIdentifier()
    
    if not identifier.model:
        print("Failed to load model. Exiting.")
        return
    
    try:
        while True:
            print("\nOptions:")
            print("1. Record and identify speaker")
            print("2. Identify from audio file")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                # Record and identify
                result = identifier.identify_speaker()
                if result:
                    print(f"\n✓ Identified: {result['speaker']} (confidence: {result['confidence']:.2f})")
                else:
                    print("\n✗ Could not identify speaker")
                    
            elif choice == '2':
                # Identify from file
                file_path = input("Enter audio file path: ").strip()
                if os.path.exists(file_path):
                    result = identifier.identify_from_file(file_path)
                    if result:
                        print(f"\n✓ Identified: {result['speaker']} (confidence: {result['confidence']:.2f})")
                    else:
                        print("\n✗ Could not identify speaker")
                else:
                    print("File not found!")
                    
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        identifier.cleanup()

if __name__ == "__main__":
    main()