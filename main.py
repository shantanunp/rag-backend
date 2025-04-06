from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI()

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

model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

with open("jira_issues.json", "r") as f:
    jira_issues = json.load(f)

documents = [f"{issue['summary']} {issue['description']}" for issue in jira_issues]
embeddings = np.load("embeddings.npy")
index = faiss.read_index("index_store.faiss")

@app.post("/search")
def search_bug(report: BugReport):
    query = f"{report.summary} {report.description}"
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")

    distances, indices = index.search(query_vector, k=5)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(jira_issues) and dist > 0.5:
            results.append({
                "issue": jira_issues[idx],
                "distance": float(1 - dist)
            })

    return {"matches": results}
