import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MedicalEmbeddingStore:
    def __init__(self, model_name='pritamdeka/SciBERT-SQuAD-pt', storage_folder='medical_embeddings'):
        """
        Initialize the embedding model and FAISS storage.
        
        Args:
            model_name (str): The name of the embedding model.
            storage_folder (str): Folder where embeddings and FAISS index are stored.
        """
        self.model = SentenceTransformer(model_name)
        self.storage_folder = f"/vector_db/{storage_folder}"

        # Ensure storage directory exists
        os.makedirs(self.storage_folder, exist_ok=True)

        self.index_path = os.path.join(self.storage_folder, "faiss_index.index")
        self.chunks_path = os.path.join(self.storage_folder, "text_chunks.npy")

        self.faiss_index = None
        self.text_chunks = []

    def create_embeddings(self, text_chunks):
        """
        Generate and store embeddings for large text chunks (500 tokens).
        
        Args:
            text_chunks (list): List of large text chunks to embed and store.
        """
        self.text_chunks = np.array(text_chunks, dtype=object)
        
        # Generate embeddings for chunks
        embeddings = np.array(self.model.encode(text_chunks, convert_to_numpy=True))
        
        # Initialize FAISS index
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(embeddings)
        
        # Save embeddings & chunks
        self.save_embeddings()

        print(f"✅ Stored {len(text_chunks)} text chunks in FAISS at '{self.storage_folder}'.")

    def save_embeddings(self):
        """Save FAISS index and text chunks to disk."""
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, self.index_path)
            np.save(self.chunks_path, self.text_chunks)
            print(f"✅ Embeddings saved in '{self.storage_folder}'.")
        else:
            print("⚠️ FAISS index is empty. Cannot save.")



# Example Usage
if __name__ == "__main__":
    # Initialize the embedding store
    storage = MedicalEmbeddingStore(storage_folder="medical_embeddings")

    # Step 1: Generate embeddings for 500-token chunks
    text_chunks = [
        "This is the first large medical text chunk containing information about diagnosis.",
        "This is another chunk discussing treatment methods in medical science.",
        "A detailed explanation of drug interactions and side effects.",
        "MRI and CT scan analysis techniques explained in this chunk.",
        "Patient history and symptoms related to cardiovascular diseases."
    ]
    storage.create_embeddings(text_chunks)
