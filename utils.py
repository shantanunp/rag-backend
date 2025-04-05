import faiss
import json

def load_index(index_path: str):
    return faiss.read_index(index_path)

def load_metadata(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)
