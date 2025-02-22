import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.word_embeddings.bert_med_embedding import get_clinical_bert_embeddings
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class EnhancedSearchEngine:
    def __init__(self, faiss_index_path: str):
        """
        Initialize the enhanced search engine with FAISS, BM25, and TF-IDF.
        """
        logging.info("Initializing Enhanced Search Engine...")
        nltk.download('punkt', quiet=True)
        
        # Load FAISS index
        faiss_index_file = f"vector_db/{faiss_index_path}.index"
        self.faiss_index = faiss.read_index(faiss_index_file)
        logging.info(f"Loaded FAISS index from {faiss_index_file}")

        # Load stored documents (fixed incorrect file extension)
        text_store_path = f"vector_db/{faiss_index_path}.json"
        with open(text_store_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        logging.info(f"Loaded {len(self.documents)} documents from {text_store_path}")

        # Tokenize documents for BM25
        tokenized_docs = [word_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Fit TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, norm='l2')
        self.tfidf_matrix = self.tfidf.fit_transform(self.documents)

        # Cache for repeated queries
        self.doc_cache = {}
        logging.info("Search engine initialized successfully.")

    def hybrid_search(self, query: str, top_k: int = 5,
                      weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        Perform a hybrid search using FAISS, BM25, and TF-IDF to retrieve relevant documents.
        Returns top_k retrieved document texts along with their scores.
        """
        logging.debug(f"Performing hybrid search for query: '{query}'")
        
        if weights is None:
            weights = {'bm25': 0.3, 'semantic': 0.4, 'tfidf': 0.3}
        
        cache_key = f"{query}_{top_k}"
        if cache_key in self.doc_cache:
            logging.debug("Returning cached search results.")
            return self.doc_cache[cache_key]

        # ------------------- 1. BM25 Search -------------------
        tokenized_query = word_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        bm25_scores = self.normalize_scores(bm25_scores)
        logging.debug(f"BM25 scores (normalized): {bm25_scores[:5]}")

        # ------------------- 2. Semantic Search (FAISS) -------------------
        query_embedding = get_clinical_bert_embeddings(query).reshape(1, -1)
        D, I = self.faiss_index.search(query_embedding, top_k)
        logging.debug(f"FAISS search results - Distances: {D}, Indices: {I}")

        # Normalize FAISS distances into scores
        semantic_scores = np.zeros(len(self.documents))
        if len(I) > 0 and I.shape[1] > 0:
            max_D = np.max(D[0]) + 1e-10  # Avoid division by zero
            for i, idx in enumerate(I[0]):
                if 0 <= idx < len(semantic_scores):
                    semantic_scores[idx] = 1 - (D[0][i] / max_D)
        logging.debug(f"Semantic scores (normalized): {semantic_scores[:5]}")

        # ------------------- 3. TF-IDF Search -------------------
        query_tfidf = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        logging.debug(f"TF-IDF scores (normalized): {tfidf_scores[:5]}")

        # ------------------- 4. Combine Scores -------------------
        final_scores = (
            weights['bm25'] * bm25_scores +
            weights['semantic'] * semantic_scores +
            weights['tfidf'] * tfidf_scores
        )

        # Get top-k results
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        results = [(self.documents[idx], float(final_scores[idx])) for idx in top_indices]
        logging.debug(f"Final retrieved document texts and scores: {results}")

        # Cache results for faster future queries
        self.doc_cache[cache_key] = results
        return results

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores between 0 and 1.
        """
        if np.all(scores == 0):
            return scores
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        logging.debug(f"Normalized scores: {normalized_scores[:5]}")
        return normalized_scores
