import torch
from transformers import ClapModel, ClapProcessor
import numpy as np
from typing import List

from config import CLAP_MODEL_NAME, SAMPLE_RATE

class AudioEmbedder:
    def __init__(self):
        print(f"Loading CLAP model ({CLAP_MODEL_NAME})...")
        self.processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
        self.model = ClapModel.from_pretrained(CLAP_MODEL_NAME)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def generate_embeddings(self, waveforms: List[np.ndarray]) -> List[List[float]]:
        """
        Generates embeddings for a list of audio waveforms.
        """
        if not waveforms:
            return []

        # Process the audio
        inputs = self.processor(
            audio=waveforms, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get audio features
            outputs = self.model.get_audio_features(**inputs)
            
            # Handle different return types across transformers versions
            if hasattr(outputs, "pooler_output"):
                audio_features = outputs.pooler_output
                # Apply projection if the model hasn't yet (expected dim is 512)
                if hasattr(self.model, "audio_projection") and audio_features.shape[-1] != self.model.config.projection_dim:
                    audio_features = self.model.audio_projection(audio_features)
            elif hasattr(outputs, "audio_embeds"):
                audio_features = outputs.audio_embeds
            else:
                audio_features = outputs
            
        return audio_features.cpu().tolist()
