# File: src/utils.py
from typing import List, Dict
import json
from pathlib import Path

def save_metadata(file_name: str, chunks: List[str], save_dir: Path):
    """Save document chunks metadata."""
    metadata = {
        "num_chunks": len(chunks),
        "chunks": chunks
    }
    
    metadata_path = save_dir / f"{file_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)