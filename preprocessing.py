import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment, silence
from pydub.utils import make_chunks
import matplotlib.pyplot as plt
import IPython.display as ipd

def preprocess_audio(input_path, output_folder, chunk_length=5000, min_silence_len=500, silence_thresh=-40, keep_silence=100):
    """
    Preprocess audio files for speaker recognition:
    1. Load audio file
    2. Remove long silences
    3. Reduce noise
    4. Split into 5-second chunks
    5. Save valid chunks (exactly 5 seconds)
    
    Parameters:
        input_path (str): Path to input audio file
        output_folder (str): Folder to save processed chunks
        chunk_length (int): Length of chunks in milliseconds (default 5000 = 5 seconds)
        min_silence_len (int): Minimum silence length to consider for trimming (ms)
        silence_thresh (int): Silence threshold in dB
        keep_silence (int): Amount of silence to keep at edges (ms)
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load audio file
    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return
    
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Normalize audio
    audio = audio.normalize()
    
    # Remove long silences in the middle
    non_silent_audio = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    
    # Combine non-silent segments
    processed_audio = AudioSegment.empty()
    for segment in non_silent_audio:
        processed_audio += segment
    
    # Noise reduction using noisereduce
    try:
        # Convert to numpy array for noise reduction
        samples = np.array(processed_audio.get_array_of_samples())
        sample_rate = processed_audio.frame_rate
        
        # Perform noise reduction
        reduced_noise = nr.reduce_noise(
            y=samples.astype('float32'),
            sr=sample_rate,
            stationary=True
        )
        
        # Convert back to AudioSegment
        processed_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=sample_rate,
            sample_width=processed_audio.sample_width,
            channels=1
        )
    except Exception as e:
        print(f"Noise reduction failed for {input_path}: {e}")
        # Continue with original audio if noise reduction fails
    
    # Split into chunks
    chunks = make_chunks(processed_audio, chunk_length)
    
    # Save valid chunks (exactly 5 seconds)
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    valid_chunks = 0
    
    for i, chunk in enumerate(chunks):
        if len(chunk) == chunk_length:
            output_path = os.path.join(output_folder, f"{base_filename}_chunk{i}.wav")
            chunk.export(output_path, format="wav")
            valid_chunks += 1
    
    print(f"Processed {input_path}: {valid_chunks} valid chunks saved to {output_folder}")
    return valid_chunks

def visualize_audio(file_path):
    """Visualize audio waveform and play it"""
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(os.path.basename(file_path))
    plt.show()
    display(ipd.Audio(file_path))

def batch_process_audio(input_folder, output_base_folder):
    """
    Process all audio files in a folder
    """
    total_chunks = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            input_path = os.path.join(input_folder, filename)
            speaker_id = os.path.splitext(filename)[0].split('_')[0]  # Assuming filename starts with speaker ID
            output_folder = os.path.join(output_base_folder, speaker_id)
            
            chunks_created = preprocess_audio(input_path, output_folder)
            if chunks_created:
                total_chunks += chunks_created
    
    print(f"\nProcessing complete. Total chunks created: {total_chunks}")

if __name__ == "__main__":
    # Example usage:
    input_file = "path/to/your/audio_file.wav"  # Replace with your audio file path
    output_folder = "processed_chunks"
    
    # Process single file
    preprocess_audio(input_file, output_folder)
    
    # To visualize a processed file (optional)
    if os.path.exists(output_folder) and os.listdir(output_folder):
        sample_output = os.path.join(output_folder, os.listdir(output_folder)[0])
        visualize_audio(sample_output)
    
    # For batch processing a folder of audio files:
    # input_folder = "path/to/audio_files"
    # output_base_folder = "processed_speaker_chunks"
    # batch_process_audio(input_folder, output_base_folder)