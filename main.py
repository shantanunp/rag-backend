from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Enable CORS for the Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["chrome-extension://your-extension-id"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming request
class BugReport(BaseModel):
    summary: str
    description: str

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Jira issues and FAISS index
with open("jira_issues.json", "r") as f:
    jira_issues = json.load(f)

documents = [f"{issue['summary']} {issue['description']}" for issue in jira_issues]
embeddings = np.load("embeddings.npy")
index = faiss.read_index("index_store.faiss")

@app.post("/search")
def search_bug(report: BugReport):
    query = f"{report.summary} {report.description}"
    query_vector = model.encode([query]).astype("float32")

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


# pip install -r requirements.txt
# python ingest.py
# uvicorn main:app --reload
# curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"summary": "Search not working", "description": "If i enter @ in search its not giving any result."}'