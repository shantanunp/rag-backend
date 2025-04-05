from fastapi import FastAPI, Request
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import load_index, load_metadata

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load stored embeddings + FAISS index + metadata
index = load_index("index_store.faiss")
metadata = load_metadata("jira_issues.json")  # Each item contains issueId, title, description

class BugQuery(BaseModel):
    summary: str
    description: str

@app.post("/search")
async def search_similar_bug(query: BugQuery):
    full_query = f"{query.summary} {query.description}"
    query_vector = model.encode([full_query])
    D, I = index.search(np.array(query_vector).astype("float32"), k=3)

    results = []
    for i in I[0]:
        if i < len(metadata):
            results.append(metadata[i])

    return {"matches": results}
