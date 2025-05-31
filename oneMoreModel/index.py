import os
import torch
import numpy as np
import torchaudio
import webrtcvad
import librosa
from speechbrain.pretrained import EncoderClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import cosine

# ========== CONFIG ==========
SAMPLE_RATE = 16000
VAD_FRAME_DURATION = 30  # ms
VAD_MODE = 3  # 0-3 (aggressive)

# Load model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# =======================
# VAD: WebRTC-based
# =======================
def apply_vad(audio, sr):
    vad = webrtcvad.Vad(VAD_MODE)
    frame_len = int(sr * VAD_FRAME_DURATION / 1000)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = (audio * 32768).astype(np.int16).tobytes()

    voiced_frames = []
    for i in range(0, len(audio), frame_len * 2):
        frame = audio[i:i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        if vad.is_speech(frame, SAMPLE_RATE):
            voiced_frames.extend(frame)
    return np.frombuffer(bytes(voiced_frames), dtype=np.int16).astype(np.float32) / 32768.0

# =======================
# Extract embeddings
# =======================
def get_embedding(audio_path):
    signal, sr = torchaudio.load(audio_path)
    signal = signal.mean(dim=0)  # mono
    voiced = apply_vad(signal.numpy(), sr)
    if len(voiced) == 0:
        return None
    tensor = torch.from_numpy(voiced).unsqueeze(0)
    embedding = classifier.encode_batch(tensor).squeeze().detach().numpy()
    return embedding

# =======================
# Prepare Training Data
# =======================
def prepare_known_speakers(folder):
    embeddings = []
    labels = []
    for idx, filename in enumerate(sorted(os.listdir(folder))):
        if filename.endswith(".wav"):
            path = os.path.join(folder, filename)
            emb = get_embedding(path)
            if emb is not None:
                embeddings.append(emb)
                labels.append(f"Speaker {idx+1}")
    return np.array(embeddings), np.array(labels)

# =======================
# Classifier + Inference
# =======================
def classify(test_path, clf, known_embeddings, known_labels, threshold=0.65):
    emb = get_embedding(test_path)
    if emb is None:
        return "Silence"
    pred = clf.predict([emb])[0]
    probs = clf.decision_function([emb])
    confidence = np.max(probs)

    # Optional threshold on cosine distance
    similarities = [1 - cosine(emb, ke) for ke in known_embeddings]
    best_sim = max(similarities)
    if best_sim < threshold:
        return "Unknown Speaker"
    return pred

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    # Train
    print("[+] Extracting known speaker embeddings...")
    known_embeddings, known_labels = prepare_known_speakers("known_speakers")
    print("[+] Training classifier...")
    clf = make_pipeline(StandardScaler(), SVC(probability=True, kernel='linear'))
    clf.fit(known_embeddings, known_labels)

    # Test
    print("\n[+] Testing...")
    test_folder = "test_audio"
    for file in os.listdir(test_folder):
        if file.endswith(".wav"):
            result = classify(os.path.join(test_folder, file), clf, known_embeddings, known_labels)
            print(f"{file} → {result}")
