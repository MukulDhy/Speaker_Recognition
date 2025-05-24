import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment, silence
from pydub.utils import make_chunks
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from sklearn.preprocessing import StandardScaler

def advanced_preprocess_audio(input_path, output_folder, chunk_length=5000, 
                            min_silence_len=300, silence_thresh=-45, keep_silence=50,
                            noise_reduce_level=0.8, spectral_gate=0.1, 
                            equalize_freq=True, sample_rate=16000):
    """
    Enhanced audio preprocessing pipeline for speaker recognition:
    
    1. Load and resample audio
    2. Convert to mono and normalize
    3. Advanced noise reduction
    4. Spectral gating
    5. Dynamic range compression
    6. Equalization
    7. Silence removal
    8. Wavelet denoising
    9. Split into fixed-length chunks
    10. Save processed chunks
    
    Parameters:
        input_path (str): Path to input audio file
        output_folder (str): Folder to save processed chunks
        chunk_length (int): Length of chunks in milliseconds (default 5000 = 5s)
        min_silence_len (int): Minimum silence length to consider for trimming (ms)
        silence_thresh (int): Silence threshold in dB
        keep_silence (int): Amount of silence to keep at edges (ms)
        noise_reduce_level (float): Aggressiveness of noise reduction (0-1)
        spectral_gate (float): Threshold for spectral gating (0-1)
        equalize_freq (bool): Whether to apply frequency equalization
        sample_rate (int): Target sample rate for output
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Load with librosa for advanced processing
        y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        
        # 1. Pre-emphasis to enhance high frequencies
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # 2. Advanced noise reduction
        reduced_noise = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=True,
            prop_decrease=noise_reduce_level,
            n_fft=1024,
            win_length=512
        )
        
        # 3. Spectral gating
        D = librosa.stft(reduced_noise)
        magnitude = np.abs(D)
        threshold = np.percentile(magnitude, 100 * spectral_gate)
        mask = magnitude > threshold
        D_clean = D * mask
        y_clean = librosa.istft(D_clean)
        
        # 4. Wavelet denoising
        coeffs = pywt.wavedec(y_clean, 'db8', level=5)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(y_clean)))
        coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
        y_denoised = pywt.waverec(coeffs, 'db8')
        
        # Handle potential length mismatch
        if len(y_denoised) > len(y_clean):
            y_denoised = y_denoised[:len(y_clean)]
        else:
            y_denoised = np.pad(y_denoised, (0, max(0, len(y_clean) - len(y_denoised))))
        
        # 5. Frequency equalization
        if equalize_freq:
            D = librosa.stft(y_denoised)
            magnitude = np.abs(D)
            db_magnitude = librosa.amplitude_to_db(magnitude)
            equalized_db = StandardScaler().fit_transform(db_magnitude.T).T
            equalized_magnitude = librosa.db_to_amplitude(equalized_db)
            D_equalized = equalized_magnitude * np.exp(1j * np.angle(D))
            y_denoised = librosa.istft(D_equalized)
        
        # 6. Convert to AudioSegment for silence removal and chunking
        y_denoised = (y_denoised * (2**15 - 1)).astype(np.int16)
        audio = AudioSegment(
            y_denoised.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        
        # 7. Dynamic range compression
        audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
        
        # 8. Silence removal
        non_silent_audio = silence.split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        
        processed_audio = AudioSegment.empty()
        for segment in non_silent_audio:
            processed_audio += segment
        
        # 9. Split into chunks
        chunks = make_chunks(processed_audio, chunk_length)
        
        # 10. Save valid chunks (exactly 5 seconds)
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        valid_chunks = 0
        
        for i, chunk in enumerate(chunks):
            if len(chunk) == chunk_length:
                output_path = os.path.join(output_folder, f"{base_filename}_chunk{i}.wav")
                
                # Apply final normalization
                chunk = chunk.normalize(headroom=0.1)
                
                chunk.export(output_path, format="wav", parameters=["-ac", "1", "-ar", str(sample_rate)])
                valid_chunks += 1
        
        print(f"Processed {input_path}: {valid_chunks} valid chunks saved to {output_folder}")
        return valid_chunks
    
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return 0

def visualize_processing(input_path, output_path):
    """Compare original and processed audio"""
    plt.figure(figsize=(14, 10))
    
    # Original audio
    y_orig, sr = librosa.load(input_path, sr=None)
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_orig, sr=sr)
    plt.title("Original Audio")
    
    # Processed audio
    y_proc, _ = librosa.load(output_path, sr=sr)
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_proc, sr=sr)
    plt.title("Processed Audio")
    
    plt.tight_layout()
    plt.show()
    
    print("Original Audio:")
    display(ipd.Audio(input_path))
    print("Processed Audio:")
    display(ipd.Audio(output_path))

def batch_process_with_progress(input_folder, output_base_folder, **kwargs):
    """
    Process all audio files in a folder with progress tracking
    """
    from tqdm import tqdm
    
    audio_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
    
    if not audio_files:
        print("No audio files found in input folder")
        return
    
    total_chunks = 0
    with tqdm(audio_files, desc="Processing audio files") as pbar:
        for filename in pbar:
            input_path = os.path.join(input_folder, filename)
            speaker_id = os.path.splitext(filename)[0].split('_')[0]
            output_folder = os.path.join(output_base_folder, speaker_id)
            
            chunks_created = advanced_preprocess_audio(input_path, output_folder, **kwargs)
            if chunks_created:
                total_chunks += chunks_created
                pbar.set_postfix({"chunks": total_chunks})
    
    print(f"\nProcessing complete. Total chunks created: {total_chunks}")

if __name__ == "__main__":
    # Example usage
    input_file = "path/to/your/audio.wav"
    output_folder = "processed_chunks_enhanced"
    
    # Process single file with enhanced preprocessing
    chunks = advanced_preprocess_audio(
        input_file,
        output_folder,
        noise_reduce_level=0.9,
        spectral_gate=0.15,
        equalize_freq=True
    )
    
    # Visualize results
    if chunks > 0:
        sample_output = os.path.join(output_folder, os.listdir(output_folder)[0])
        visualize_processing(input_file, sample_output)
    
    # For batch processing:
    # input_folder = "path/to/audio_files"
    # output_base_folder = "enhanced_speaker_chunks"
    # batch_process_with_progress(input_folder, output_base_folder)