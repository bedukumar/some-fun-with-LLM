import os

# Audio Processing
SAMPLE_RATE = 48000
CHUNK_LENGTH_SEC = 10.0
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_LENGTH_SEC)

# Model
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"

# Vector Database (Qdrant)
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "audio_embeddings"
VECTOR_SIZE = 512

# Input Directory for raw audio files
AUDIO_INPUT_DIR = "./audio_input"
