 
import os

import config
from ingestion import setup_tesseract, ingest_pdf
from embedding_store import (
    load_embed_model, get_qdrant_client, create_collection,
    pdf_already_stored, embed_chunks, store_chunks,
)


def ingest_folder(folder_path=None):
    folder_path = folder_path or config.PDF_FOLDER
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    setup_tesseract()
    model = load_embed_model()
    client = get_qdrant_client()
    create_collection(client)

    pdf_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))
    print("PDF files found:", len(pdf_files))

    for name in pdf_files:
        print("\n" + "=" * 60)
        print("Processing:", name)

        try:
            if pdf_already_stored(client, name):
                print("Already stored, skip.")
                continue

            chunks = ingest_pdf(os.path.join(folder_path, name))
            if not chunks:
                print("text not founda, skip.")
                continue

            embeddings = embed_chunks(model, chunks)
            count = store_chunks(client, chunks, embeddings)
            print(f"Stored {count} vectors for: {name}")

        except Exception as e:
            
            print(f"Error in {name}: {e}")

    total = client.count(collection_name=config.COLLECTION_NAME, exact=True).count
    print("\nDone. Total vectors in collection:", total)
    return model, client
