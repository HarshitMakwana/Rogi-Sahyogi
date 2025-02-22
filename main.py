import logging
import time
from app.data_loader.pdf_loader import PdfParser, OcrEngine
from app.pre_processor.markdown_preprocess import text_splitter,text_preprocessor
from app.rag.enhanced_search_engine import EnhancedSearchEngine
from app.word_embeddings.bert_med_embedding import get_clinical_bert_embeddings
from app.faiss_db_service.store import store_in_faiss

class PDFProcessingPipeline:
    def __init__(self, pdf_path: str, output_dir: str = "parsed_pdfs", faiss_index_path: str = "vector_db"):
        """
        Initializes the PDF processing pipeline with given file path and configurations.
        """
        logging.info("Initializing PDFProcessingPipeline...")
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.faiss_index_path = faiss_index_path
        self.parser = self._initialize_parser()
        logging.info("Initialization complete.")
        
    def _initialize_parser(self) -> PdfParser:
        """
        Creates and configures a PdfParser instance.
        """
        logging.info("Setting up PDF parser...")
        return PdfParser(
            ocr_engine=OcrEngine.NONE,  # Disables OCR since text extraction is direct
            languages=["en"],  # Specifies English as the primary language
            use_gpu=True,  # Enables GPU acceleration for performance improvement
            num_threads=4,  # Sets the number of CPU threads for parallel processing
            do_table_structure=True,  # Enables table structure recognition
            cell_matching=True,  # Improves accuracy in structured documents
            output_dir=self.output_dir  # Specifies directory to store parsed files
        )
    
    def parse_pdf(self):
        """
        Parses the PDF and extracts content as markdown.
        """
        logging.info("Starting PDF parsing...")
        result = self.parser.parse_pdf(input_path=self.pdf_path, export_formats=["md"])
        logging.info(f"Parsing completed in {result['processing_time']:.2f} seconds")
        logging.info(f"Exported files: {result['export_paths']}")
        logging.debug(f"Parsed content preview: {result['content'][:500]}")  # Logs preview of parsed text
        return result["content"]
    
    def process_text(self, text: str):
        """
        Splits extracted text into manageable chunks for embedding processing.
        """
        logging.info("Text process before splitting...")
        text = text_preprocessor(text)
        logging.info("Splitting text into chunks...")
        text_chunks = text_splitter.split_text(text)
        logging.info(f"Total chunks created: {len(text_chunks)}")
        logging.debug(f"First chunk preview: {text_chunks[0] if text_chunks else 'No chunks generated'}")
        return text_chunks
    
    def store_embeddings(self, texts_chunks):
        """
        Converts text chunks into embeddings and stores them in FAISS vector database.
        """
        logging.info("Generating embeddings and storing in FAISS...")
        store_in_faiss(
            faiss_index_path=self.faiss_index_path,
            clinical_bert_embeddings=get_clinical_bert_embeddings,
            texts=texts_chunks
        )
        logging.info("FAISS storage completed.")
    
    def run_pipeline(self):
        """
        Executes the full PDF processing pipeline: parsing, text processing, and embedding storage.
        """
        start_time = time.time()
        logging.info("Pipeline execution started.")
        
        parsed_text = self.parse_pdf()
        text_chunks = self.process_text(parsed_text)
        self.store_embeddings(text_chunks)
        
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Pipeline execution completed in {total_time:.2f} seconds.")
        logging.info("Pipeline finished successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)  # Sets logging level to debug for detailed logs
    logging.info("Starting main execution...")
    pdf_processor = PDFProcessingPipeline(
        pdf_path=r"C:\Users\harsh\OneDrive\Desktop\Rize\Rogi-Sahyogi\app\test_pdf\PEREZ_PEDRO_DA_RECORDS.pdf",faiss_index_path="PEREZ_PEDRO_DA_RECORDS"
    )
    pdf_processor.run_pipeline()
    logging.info("Main execution completed.")
    
    search_engine = EnhancedSearchEngine(faiss_index_path="PEREZ_PEDRO_DA_RECORDS")

    query = "patient name"
    top_results = search_engine.hybrid_search(query, top_k=5)

    for i, (text, score) in enumerate(top_results):
        print(f"Rank {i+1}: Score: {score:.4f}\n{text}\n")


















# from app.data_loader.pdf_loader import PdfParser,OcrEngine
# from app.pre_processor.markdown_preprocess import text_splitter
# from app.rag.enhanced_search_engine import EnhancedSearchEngine
# from app.word_embeddings.bert_med_embedding import get_clinical_bert_embeddings
# from app.faiss_db_service.store import store_in_faiss

# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Initialize parser with custom settings
# parser = PdfParser(
#     ocr_engine=OcrEngine.NONE,
#     languages=["en"],
#     use_gpu=True,
#     num_threads=4,
#     do_table_structure=True,
#     cell_matching=True,
#     output_dir="parsed_pdfs"
# )

# # Parse PDF file
# result = parser.parse_pdf(
#     input_path=r"C:\Users\harsh\OneDrive\Desktop\Rize\Rogi-Sahyogi\app\test_pdf\PEREZ_PEDRO_DA_RECORDS.pdf",
#     export_formats=["md"]
# )

# print("Parsing completed!")
# print(f"Processing time: {result['processing_time']:.2f} seconds")
# print("Exported files:")

# for format_name, path in result['export_paths'].items():
#     print(f"- {format_name}: {path}")
    
# texts_chunks = text_splitter.split_text(result["content"])

# store_in_faiss(faiss_index_path="vector_db" ,clinical_bert_embeddings= get_clinical_bert_embeddings,texts=texts_chunks)


# # Step 4: Print the Chunks
# for idx, chunk in enumerate(texts_chunks):
#     print(f"Chunk {idx+1}:\n{chunk}\n{'-'*50}")




# # Initialize the engine
# faiss_index_path = "faiss_index.bin"  # Path to your FAISS index file
# documents = [
#     "Patient diagnosed with pneumonia, treated with antibiotics.",
#     "MRI scan shows no abnormalities in the brain.",
#     "Blood test indicates high cholesterol levels.",
#     "Patient admitted for severe chest pain.",
#     "Doctor prescribed painkillers for post-surgery recovery."
# ]
# search_engine = EnhancedSearchEngine(faiss_index_path, documents)

# query = "Patient experiencing chest pain after surgery."
# top_k_results = search_engine.hybrid_search(query, top_k=5)
# print("Search Results:", top_k_results)

# for idx, score in top_k_results:
#     print(f"Document: {documents[idx]}, Score: {score}")


    



# # File: main.py
# from pathlib import Path
# from config.config import Config
# from src.document_processor import DocumentProcessor
# from src.embeddings import EmbeddingEngine
# from src.search_engine import SearchEngine
# from src.utils import save_metadata

# def main():
#     # Initialize configuration
#     Config.ensure_directories()
    
#     # Initialize components
#     doc_processor = DocumentProcessor(
#         chunk_size=Config.CHUNK_SIZE,
#         chunk_overlap=Config.CHUNK_OVERLAP
#     )
#     embedding_engine = EmbeddingEngine(
#         model_name=Config.MODEL_NAME,
#         vector_store_dir=Config.VECTOR_STORE_DIR
#     )
#     search_engine = SearchEngine()
    
#     # Process documents
#     documents_dir = Config.DATA_DIR / "documents"
#     processed_docs = doc_processor.process_directory(documents_dir)
    
#     # Create vector store for each document
#     for file_name, chunks in processed_docs.items():
#         # Compute embeddings
#         embeddings = embedding_engine.compute_embeddings(chunks)
        
#         # Create and save FAISS index
#         index = embedding_engine.create_index(embeddings, file_name)
        
#         # Save metadata
#         save_metadata(file_name, chunks, Config.VECTOR_STORE_DIR)
        
#         print(f"Processed and indexed {file_name}")
    
#     # Example search
#     query = "flu symptoms and fever treatment"
#     file_name = "example_file"  # Replace with actual file name
    
#     # Load index and perform search
#     index, embeddings = embedding_engine.load_index(file_name)
#     chunks = processed_docs[file_name]
    
#     results = search_engine.hybrid_search(
#         query=query,
#         documents=chunks,
#         embeddings=embeddings,
#         faiss_index=index,
#         embedding_engine=embedding_engine,
#         top_k=Config.TOP_K_RESULTS
#     )
    
#     # Print results
#     print("\nSearch Results:")
#     for idx in results:
#         print(f"- {chunks[idx]}\n")

# if __name__ == "__main__":
#     main()