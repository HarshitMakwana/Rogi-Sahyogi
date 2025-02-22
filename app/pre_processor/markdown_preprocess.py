from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def text_preprocessor(text):
    text = re.sub(r"<!-- missing-text -->", "", text)
    text = re.sub(r"<!-- image -->", "", text)
    text = re.sub(r'([a-z]+)([A-Z])', r'\1 \2', text)
    text = re.sub(r'\b(\w+)\1+\b', r'\1', text)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    text = text.replace("/n/n","/n")
    return text

# Step 2: Use RecursiveCharacterTextSplitter with Markdown-friendly Separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n# ", "\n## ", "\n### ", "\n- ", "\n\n"]  # Preserve Markdown structure
)
