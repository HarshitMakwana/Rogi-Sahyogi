import os
import json
import numpy as np
import faiss

def store_in_faiss(faiss_index_path: str, clinical_bert_embeddings, texts):
    print("Processing texts and storing embeddings in FAISS index...")

    # Convert text chunks into embeddings
    embeddings = np.vstack([clinical_bert_embeddings(text) for text in texts]).astype(np.float32)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index
    os.makedirs("vector_db", exist_ok=True)
    faiss.write_index(index, f"vector_db/{faiss_index_path}.index")

    # Store texts in a JSON file
    text_store_path = f"vector_db/{faiss_index_path}.json"
    with open(text_store_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=4)

    print(f"Stored {len(texts)} documents in FAISS and saved texts in {text_store_path}")
