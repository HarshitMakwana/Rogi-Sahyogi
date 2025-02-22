import torch
from transformers import AutoTokenizer, AutoModel

# Load the Clinical BERT model and tokenizer
model_name = "medicalai/ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings from Clinical BERT
def get_clinical_bert_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Get the model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Extract meaningful embeddings
    print("Embeddings shape:", embeddings.shape)
    return embeddings