from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Enable CORS for frontend extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BugReport(BaseModel):
    summary: str
    description: str

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("jira_issues.json", "r") as f:
    jira_issues = json.load(f)

embeddings = np.load("embeddings.npy")
index = faiss.read_index("index_store.faiss")

# --- Chunking function for query ---
def chunk_text(text, max_tokens=200):
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    token_count = 0

    for sentence in sentences:
        token_estimate = len(sentence.split())
        if token_count + token_estimate > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            token_count = 0
        current_chunk.append(sentence)
        token_count += token_estimate

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --- Query embedding with pooling ---
def get_pooled_embedding(text: str):
    chunks = chunk_text(text)
    chunk_embeddings = model.encode(chunks)
    return np.mean(chunk_embeddings, axis=0).astype("float32")

@app.post("/search")
def search_bug(report: BugReport):
    query = f"{report.summary} {report.description}"
    query_vector = get_pooled_embedding(query).reshape(1, -1)

    print("Query vector shape:", query_vector.shape)
    print("FAISS index dimension:", index.d)

    # Top 5 results
    similarity, indices = index.search(query_vector, k=5)

    results = []
    for score, idx in zip(similarity[0], indices[0]):
        if idx < len(jira_issues) and score < 1.0:
            results.append({
                "issue": jira_issues[idx],
                "similarity": f"{round(float(score) * 100, 2)}%"
            })

    return {"matches": results}

@app.get("/")
def health():
    return {"status": "ok"}


# pip install -r requirements.txt
# python ingest.py
# uvicorn main:app --reload
# curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"summary": "Search not working", "description": "If i enter @ in search its not giving any result."}'