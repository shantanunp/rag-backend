import numpy as np
import faiss
import json
import hashlib
import os
from sentence_transformers import SentenceTransformer

# Simulated Jira data (replace with real data from API later)
jira_issues = [
    {
        "id": "JIRA-101",
        "summary": "Login failure on invalid credentials",
        "description": """
            Given I am on the login page
            When I enter invalid username and password
            Then I should see an error message
        """
    },
    {
        "id": "JIRA-102",
        "summary": "Password reset should be redirected",
        "description": """
            Given I forgot my password
            When I click on reset link in email
            Then I should be redirected to a reset form
        """
    },
    {
        "id": "JIRA-103",
        "summary": "Search fails with special characters",
        "description": """
            Given I am logged in
            When I search using special characters like #, @, %
            Then no results are shown or app crashes
        """
    },
    {
        "id": "JIRA-104",
        "summary": "User profile update success message",
        "description": """
            Given I have updated my profile details
            When I save changes
            Then I should see a confirmation message
        """
    },
    {
        "id": "JIRA-105",
        "summary": "Update cart after adding item",
        "description": """
            Given I add a product to cart
            When I check the cart
            Then the item count should increase by one
        """
    },
    {
        "id": "JIRA-106",
        "summary": "Checkout should work for guest users",
        "description": """
            Given I am not logged in
            When I go to checkout
            Then I should be prompted to login or continue as guest
        """
    },
    {
        "id": "JIRA-107",
        "summary": "Mobile view should be responsive",
        "description": """
            Given I open site on mobile browser
            When I navigate to any page
            Then the layout should remain responsive and clean
        """
    },
    {
        "id": "JIRA-108",
        "summary": "Notification banner for new message",
        "description": """
            Given I receive a new message
            When I am on dashboard
            Then a notification banner should appear
        """
    },
    {
        "id": "JIRA-109",
        "summary": "Apply Dark mode theme",
        "description": """
            Given I toggle dark mode
            When I reload the page
            Then the dark theme should persist
        """
    },
    {
        "id": "JIRA-110",
        "summary": "ATM Cash Withdrawal OTP Expiry Issue",
        "description": """
            "Users are experiencing issues with the OTP-based cash withdrawal feature.
             The OTP is supposed to be valid for 15 minutes, but multiple reports indicate that the OTP is expiring 
             within 2–3 minutes.
             This leads to poor user experience as customers are often forced to regenerate a new OTP while standing at 
             the ATM. In addition, the regenerated OTP is sometimes invalid due to backend caching issues.
             Logs from ATM servers show inconsistencies in time synchronization between the mobile app servers and 
             ATM endpoints, causing token validation mismatches.
             This has been observed in multiple cities including Mumbai, Delhi, and Bangalore.
             A deeper investigation revealed that the Redis cache TTL was incorrectly configured to 120 seconds in 
             some regions. Furthermore, if the user tries more than 3 times, the system blocks the account for 24 hours 
             as part of fraud detection — even when the error is on our side. We need to introduce more granular logging,
             ensure consistent configuration across regions, and possibly extend OTP validity to 20 minutes. 
             Also consider enabling real-time monitoring of Redis TTL values and syncing server clocks using NTP on a
             tighter interval. Customer complaints have been piling up in the app store reviews, and 
             this could impact user trust if not addressed urgently.
             Suggested next steps include a temporary rollback to the previous OTP mechanism until a fix is verified, 
             and setting up a war room with backend, mobile, SRE, and QA teams for root cause analysis and resolution.
        """
    },
]

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Chunking ---
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

# --- Mean pooled embedding for all chunks ---
def get_pooled_embedding(text: str):
    chunks = chunk_text(text)
    chunk_embeddings = model.encode(chunks)
    return np.mean(chunk_embeddings, axis=0)

# --- Caching based on content hash ---
def hash_issue(issue):
    combined = issue["summary"] + issue["description"]
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

# --- Load cache ---
cache_path = "issue_embeddings.json"
cached_embeddings = {}
if os.path.exists(cache_path):
    with open(cache_path, "r") as f:
        cached_embeddings = json.load(f)

# --- Generate embeddings ---
new_embeddings = []
updated_cache = {}

for issue in jira_issues:
    issue_id = issue["id"]
    doc_text = f"{issue['summary']} {issue['description']}"
    issue_hash = hash_issue(issue)

    if issue_id in cached_embeddings and cached_embeddings[issue_id]["hash"] == issue_hash:
        embedding = cached_embeddings[issue_id]["embedding"]
        print(f"Using cached embedding for {issue_id}")
    else:
        print(f"Generating embedding for {issue_id}")
        embedding = get_pooled_embedding(doc_text).tolist()

    updated_cache[issue_id] = {
        "embedding": embedding,
        "hash": issue_hash
    }
    new_embeddings.append(embedding)

# --- Save updated cache ---
with open(cache_path, "w") as f:
    json.dump(updated_cache, f, indent=2)

# --- Save embedding matrix ---
embedding_matrix = np.array(new_embeddings).astype("float32")
np.save("embeddings.npy", embedding_matrix)

# --- FAISS Index ---
dimension = embedding_matrix.shape[1]  # Should be 384
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)
faiss.write_index(index, "index_store.faiss")

# --- Save metadata ---
with open("jira_issues.json", "w") as f:
    json.dump(jira_issues, f, indent=2)

print("✅ Index and embeddings saved successfully.")