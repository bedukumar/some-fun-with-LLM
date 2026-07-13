import os
import traceback
from config import AUDIO_INPUT_DIR
from audio_processor import process_audio_file
from embedder import AudioEmbedder
from db_client import QdrantDBClient

def main():
    if not os.path.exists(AUDIO_INPUT_DIR):
        os.makedirs(AUDIO_INPUT_DIR)
        print(f"Created input directory: {AUDIO_INPUT_DIR}. Please add audio files to it.")
        return

    # Initialize components
    embedder = AudioEmbedder()
    db_client = QdrantDBClient()

    # Find all audio files
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a'}
    audio_files = []
    for root, _, files in os.walk(AUDIO_INPUT_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"No audio files found in {AUDIO_INPUT_DIR}.")
        return

    print(f"Found {len(audio_files)} audio files to process.")

    for idx, file_path in enumerate(audio_files):
        print(f"\nProcessing [{idx+1}/{len(audio_files)}]: {file_path}")
        try:
            metadata, chunks = process_audio_file(file_path)
            
            if not chunks:
                print(f"No chunks generated for {file_path}. Skipping.")
                continue

            # We process all chunks of a file in smaller batches to avoid OOM
            BATCH_SIZE = 32
            
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i+BATCH_SIZE]
                
                waveforms = [c["waveform"] for c in batch_chunks]
                embeddings = embedder.generate_embeddings(waveforms)
                
                payloads = []
                for c in batch_chunks:
                    payload = metadata.copy()
                    payload["chunk_id"] = c["chunk_id"]
                    payload["start_time_sec"] = c["start_time_sec"]
                    payload["end_time_sec"] = c["end_time_sec"]
                    payloads.append(payload)
                
                db_client.insert_batch(embeddings, payloads)
                
        except Exception as e:
            print(f"Failed to process {file_path}. Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
