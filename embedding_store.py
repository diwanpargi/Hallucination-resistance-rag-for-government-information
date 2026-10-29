 
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)

import config


def load_embed_model():
    
    return SentenceTransformer(config.EMBED_MODEL)


def get_qdrant_client():
    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise ValueError("QDRANT_URL aur QDRANT_API_KEY set karo (environment variable).")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    print("Connected to Qdrant Cloud")
    return client


def create_collection(client):
    """make collection if not exist"""
    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.VECTOR_SIZE, distance=Distance.COSINE),
        )
    
    client.create_payload_index(
        collection_name=config.COLLECTION_NAME,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("Collection ready")


def pdf_already_stored(client, filename):
    """if file is already processed then skip that file"""
    result = client.count(
        collection_name=config.COLLECTION_NAME,
        count_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=filename))]),
        exact=True,
    )
    return result.count > 0


def embed_chunks(model, chunks):
    texts = [c["text"] for c in chunks]
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def store_chunks(client, chunks, embeddings):
    points = []
    for chunk, emb in zip(chunks, embeddings):
         
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"]))
        points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=chunk))

    client.upsert(
        collection_name=config.COLLECTION_NAME,
        points=points,
        wait=True,
        timeout=120,
    )
    return len(points)
