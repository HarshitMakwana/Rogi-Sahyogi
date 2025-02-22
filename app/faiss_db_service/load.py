import faiss
import numpy as np

def load_faiss_index(faiss_index_path: str):
    return faiss.read_index(f"vector_db/{faiss_index_path}")
