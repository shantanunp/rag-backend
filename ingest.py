import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

# Simulated Jira data (replace with real data from API later)
jira_issues = [
    {"id": "JIRA-1001", "summary": "App crashes on login", "description": "When the user logs in, the app crashes"},
    {"id": "JIRA-1002", "summary": "Slow page load", "description": "Home page takes too long to load"},
    {"id": "JIRA-1003", "summary": "Incorrect calculation", "description": "Total amount is calculated wrong in cart"},
    # Add more
]

model = SentenceTransformer("all-MiniLM-L6-v2")
documents = [f"{issue['summary']} {issue['description']}" for issue in jira_issues]
embeddings = model.encode(documents)

# Save embeddings
np.save("embeddings.npy", embeddings)

# Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, "index_store.faiss")

# Save metadata
with open("jira_issues.json", "w") as f:
    json.dump(jira_issues, f, indent=2)
