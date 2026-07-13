import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any

from config import QDRANT_PATH, COLLECTION_NAME, VECTOR_SIZE

class QdrantDBClient:
    def __init__(self):
        print(f"Initializing Qdrant client at {QDRANT_PATH}...")
        self.client = QdrantClient(path=QDRANT_PATH)
        self.setup_collection()

    def setup_collection(self):
        """Creates the collection if it does not exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            print(f"Creating collection '{COLLECTION_NAME}'...")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, 
                    distance=models.Distance.COSINE
                ),
            )
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")

    def insert_batch(self, embeddings: List[List[float]], payloads: List[Dict[str, Any]]):
        """Inserts a batch of vectors with their associated payloads."""
        if not embeddings or not payloads:
            return

        points = []
        for emb, payload in zip(embeddings, payloads):
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload
                )
            )
            
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Inserted {len(points)} points into Qdrant.")
