import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def split_into_chunks(input_path, output_folder, chunk_length=5000):
    """
    Simply split audio files into 5-second chunks
    
    Parameters:
        input_path (str): Path to input audio file
        output_folder (str): Folder to save processed chunks
        chunk_length (int): Length of chunks in milliseconds (default 5000 = 5 seconds)
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load audio file
    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return 0
    
    # Split into chunks
    chunks = make_chunks(audio, chunk_length)
    
    # Save all chunks (even if they're slightly shorter than 5 seconds)
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    valid_chunks = 0
    
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_folder, f"{base_filename}_chunk{41 + i}.wav")
        chunk.export(output_path, format="wav")
        valid_chunks += 1
    
    print(f"Processed {input_path}: {valid_chunks} chunks saved to {output_folder}")
    return valid_chunks

if __name__ == "__main__":
    # Example usage:
    input_file = "audio_data/mukul/mukul_sample2.opus"  # Replace with your audio file path
    output_folder = "mukul_chunks"
    
    # Process single file
    split_into_chunks(input_file, output_folder)