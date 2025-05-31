import os
import json
from datetime import datetime

def generate_recordings_array(base_path):
    """
    Generates an array of recording file paths in the specified format.
    
    Args:
        base_path (str): The base directory path (e.g., "family_voice_data\\mukul2\\mukul\\")
    
    Returns:
        list: Array of file paths in the required format
    """
    recordings = []
    
    # Ensure the base path ends with a separator
    if not base_path.endswith(os.path.sep):
        base_path += os.path.sep
    
    # Check if directory exists
    if not os.path.isdir(base_path):
        print(f"Error: Directory not found - {base_path}")
        return recordings
    
    # Get all WAV files in the directory
    for filename in os.listdir(base_path):
        if filename.lower().endswith('.wav'):
            full_path = os.path.join(base_path, filename).replace('/', '\\')
            recordings.append(full_path)
    
    # Sort recordings by filename (optional)
    recordings.sort()
    
    return recordings

# Example usage
if __name__ == "__main__":
    # Replace this with your actual base path
    base_directory = "family_voice_data\\smriti\\pooja\\"
    
    # Generate the recordings array
    recordings_list = generate_recordings_array(base_directory)
    
    # Create the JSON structure
    result = {
        "recordings": recordings_list
    }
    
    # Print as JSON with pretty formatting
    print(json.dumps(result, indent=2))
    
    # Optionally save to a file
    # with open('recordings_list.json', 'w') as f:
    #     json.dump(result, f, indent=2)