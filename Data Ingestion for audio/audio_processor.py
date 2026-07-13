import os
import librosa
import numpy as np
from mutagen import File
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1
from typing import Tuple, List, Dict, Any

from config import SAMPLE_RATE, SAMPLES_PER_CHUNK, CHUNK_LENGTH_SEC

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extracts ID3 metadata from an audio file."""
    metadata = {
        "track_name": os.path.basename(file_path),
        "artist": "Unknown Artist",
        "file_path": file_path
    }
    
    try:
        audio = File(file_path)
        if audio is not None and hasattr(audio, 'tags') and audio.tags:
            tags = audio.tags
            # For MP3/ID3
            if 'TIT2' in tags:
                metadata["track_name"] = str(tags['TIT2'])
            elif 'title' in tags:
                metadata["track_name"] = str(tags['title'][0])
                
            if 'TPE1' in tags:
                metadata["artist"] = str(tags['TPE1'])
            elif 'artist' in tags:
                metadata["artist"] = str(tags['artist'][0])
    except Exception as e:
        print(f"Warning: Could not read metadata for {file_path}. Error: {e}")
        
    return metadata

def load_and_resample(file_path: str) -> np.ndarray:
    """Loads an audio file and resamples it to the target sample rate."""
    # librosa.load will automatically resample to sr and convert to mono
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return waveform

def chunk_waveform(waveform: np.ndarray) -> List[Dict[str, Any]]:
    """Chunks a 1D waveform array into fixed size pieces with timestamps."""
    chunks = []
    total_samples = len(waveform)
    
    for i, start_idx in enumerate(range(0, total_samples, SAMPLES_PER_CHUNK)):
        end_idx = start_idx + SAMPLES_PER_CHUNK
        chunk = waveform[start_idx:end_idx]
        
        start_time_sec = start_idx / SAMPLE_RATE
        end_time_sec = min((end_idx / SAMPLE_RATE), (total_samples / SAMPLE_RATE))
        
        chunks.append({
            "chunk_id": i,
            "waveform": chunk,
            "start_time_sec": start_time_sec,
            "end_time_sec": end_time_sec
        })
        
    return chunks

def process_audio_file(file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Processes a single audio file: extracts metadata, loads, and chunks."""
    metadata = extract_metadata(file_path)
    waveform = load_and_resample(file_path)
    chunks = chunk_waveform(waveform)
    return metadata, chunks
