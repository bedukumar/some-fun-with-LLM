from qdrant_client import QdrantClient
from config import QDRANT_PATH, COLLECTION_NAME

client = QdrantClient(path=QDRANT_PATH)

try:
    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Total embeddings in DB: {count}")
    
    if count > 0:
        # Retrieve the first one
        res = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=True,
            with_vectors=True
        )
        print(f"Sample payload: {res[0][0].payload}")
except Exception as e:
    print(f"Error querying DB: {e}")
