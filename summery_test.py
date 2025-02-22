import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

class MedicalSummarizer:
    def __init__(self, model_name="Falconsai/medical_summarization", max_input_tokens=2048):
        """Initialize the model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.max_input_tokens = max_input_tokens
        
        if self.device == "cuda":
            self.model.half()  # Optimize for GPU
    
    def summarize_batch(self, chunks):
        """Summarizes text in batches for efficiency."""
        inputs = self.tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=5,  # Increased beams for better summaries
                length_penalty=1.5,
                repetition_penalty=1.2,
                early_stopping=True
            )
        
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    
    def hierarchical_summarization(self, text):
        """Performs hierarchical summarization using LLM-enhanced final summarization."""
        print(f"Processing {len(text)} chunks...")  
        sub_summaries = self.summarize_batch(text)
        
        final_prompt = (
            "Summarize the following medical and work status reports in a professional and concise manner, "
            "ensuring clarity while retaining key details such as patient information, diagnosis, treatment, "
            "work restrictions, medical evaluations, legal aspects, and overall status. Keep it clear, structured, "
            "and easily understandable:\n" + "\n".join(sub_summaries)
        )
        print(final_prompt)
        # final_summary = self.summarize_batch([final_prompt])[0]
        # return final_summary

# Example usage
if __name__ == "__main__":
    summarizer = MedicalSummarizer()
    with open("vector_db/PEREZ_PEDRO_DA_RECORDS.json", "r") as f:
        text_data = json.load(f)
    
    summary = summarizer.hierarchical_summarization(text_data)
    print("\nFinal Summary:\n", summary)



# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from torch.utils.data import DataLoader

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM




# # ------------------------------- CONFIG ------------------------------- #
# MODEL_NAME = "google/long-t5-tglobal-small"  # Lighter model
# MAX_INPUT_TOKENS = 2048  # Reduce token size
# MAX_SUMMARY_TOKENS = 512  # Output token limit
# BATCH_SIZE = 4  # Process in batches

# # Load model & tokenizer
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = AutoTokenizer.from_pretrained("Falconsai/medical_summarization")
# model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/medical_summarization")

# # Enable FP16 for faster execution (if GPU)
# if DEVICE == "cuda":
#     model.half()  # Reduce model size & speed up inference

# # ------------------------------- FUNCTIONS ------------------------------- #
# def summarize_batch(chunks):
#     """Summarizes text in batches for speed."""
#     inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKENS)
#     inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
    
#     with torch.no_grad():  # Disable gradients for faster inference
#         summary_ids = model.generate(inputs["input_ids"], max_length=MAX_SUMMARY_TOKENS, num_beams=5)
    
#     return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

# def generate_summary(chunks):
#     """Process large document in parallel using DataLoader."""
#     dataloader = DataLoader(chunks, batch_size=BATCH_SIZE, shuffle=False)
#     summaries = []

#     for batch in dataloader:
#         summaries.extend(summarize_batch(batch))

#     return summarize_batch(" ".join(summaries))  # Final summary

# # ------------------------------- EXECUTION ------------------------------- #
# if __name__ == "__main__":
#     # Load data from JSON (replace with your file)
#     import json
#     with open("vector_db/PEREZ_PEDRO_DA_RECORDS.json", "r") as f:
#         chunks = json.load(f)

#     print(f"Processing {len(chunks)} chunks...")
#     summary = generate_summary(chunks)

#     print("\nFinal Summary:\n", summary)



# import json
# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from concurrent.futures import ThreadPoolExecutor

# # ------------------------------- CONFIGURATION ------------------------------- #

# # Choose a summarization model from Hugging Face
# MODEL_NAME = "google/long-t5-tglobal-base"  # Alternative: "google/bigbird-pegasus-large-arxiv"
# MAX_INPUT_TOKENS = 4096  # Max tokens per chunk
# MAX_SUMMARY_TOKENS = 512  # Max tokens in the summary

# # Load the tokenizer and model
# print(f"Loading model: {MODEL_NAME}...")  # Debug log
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# # Enable CUDA if available
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(DEVICE)
# print(f"Model loaded on {DEVICE}")  # Debug log


# # ------------------------------- HELPER FUNCTIONS ------------------------------- #

# def summarize_text(text, max_length=MAX_SUMMARY_TOKENS):
#     """
#     Summarizes a single chunk of text using a transformer model.
    
#     :param text: Input text chunk.
#     :param max_length: Maximum length of the summary.
#     :return: Summarized text.
#     """
#     try:
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
#         inputs = {key: val.to(DEVICE) for key, val in inputs.items()}  # Move to GPU if available

#         summary_ids = model.generate(
#             inputs["input_ids"],
#             max_length=max_length,
#             num_beams=5,
#             length_penalty=2.0,
#             early_stopping=True
#         )

#         return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     except Exception as e:
#         print(f"Error during summarization: {e}")
#         return ""  # Return empty summary in case of error


# def read_markdown(file_path):
#     """
#     Reads a markdown medical report and returns its content as a string.

#     :param file_path: Path to the Markdown file.
#     :return: String content of the file.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Markdown file not found: {file_path}")

#     with open(file_path, "r", encoding="utf-8") as f:
#         return f.read()


# def read_json_chunks(file_path):
#     """
#     Reads text chunks from a JSON file.

#     :param file_path: Path to the JSON file containing text chunks.
#     :return: List of text chunks.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"JSON file not found: {file_path}")

#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)


# # ------------------------------- MAIN SUMMARIZATION PIPELINE ------------------------------- #

# def generate_summary(file_identifier, input_type="json"):
#     """
#     Generates a detailed and accurate summary from a large text dataset.

#     :param file_identifier: The base filename (without extension).
#     :param input_type: "json" for pre-chunked text, "md" for markdown files.
#     :return: Final summarized text.
#     """
#     # Determine file path based on input type
#     if input_type == "json":
#         file_path = f"vector_db/{file_identifier}.json"
#         chunks = read_json_chunks(file_path)
#     elif input_type == "md":
#         file_path = f"{file_identifier}.md"
#         raw_text = read_markdown(file_path)
#         chunks = [raw_text]  # Single large document as one chunk
#     else:
#         raise ValueError("Invalid input_type. Use 'json' for chunked text or 'md' for Markdown.")

#     print(f"Processing {len(chunks)} chunks...")  # Debug log

#     # Summarize chunks in parallel for speed
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         summaries = list(executor.map(summarize_text, chunks))

#     # Merge all summaries into a final summary
#     print("Generating final summary...")  # Debug log
#     final_summary = summarize_text(" ".join(summaries))

#     return final_summary


# # ------------------------------- EXECUTION ------------------------------- #

# if __name__ == "__main__":
#     FILE_NAME = "PEREZ_PEDRO_DA_RECORDS"  # Change this to the actual filename (without extension)
#     INPUT_TYPE = "json"  # Use "md" if the input is a Markdown file

#     print("Starting summarization process...")  # Debug log
#     summary = generate_summary(FILE_NAME, INPUT_TYPE)
#     print("\nFinal Summary:\n", summary)
