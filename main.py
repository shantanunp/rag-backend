from fastapi import FastAPI, Request
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import load_index, load_metadata
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 👇 Add this CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with the specific extension origin for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# pip install -r requirements.txt
# python ingest.py
# uvicorn main:app --reload
# curl -X POST http://localhost:8000/search \
#                               -H "Content-Type: application/json" \
#                                  -d '{"summary": "app login crash", "description": "app crashes when logging in"}'
