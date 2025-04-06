import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

# Simulated Jira data (replace with real data from API later)
jira_issues = [
     [
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
        "summary": "Password reset link not working",
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
        "summary": "User profile update success message missing",
        "description": """
            Given I have updated my profile details
            When I save changes
            Then I should see a confirmation message
        """
    },
    {
        "id": "JIRA-105",
        "summary": "Cart does not update after adding item",
        "description": """
            Given I add a product to cart
            When I check the cart
            Then the item count should increase by one
        """
    },
    {
        "id": "JIRA-106",
        "summary": "Checkout page breaks for guest users",
        "description": """
            Given I am not logged in
            When I go to checkout
            Then I should be prompted to login or continue as guest
        """
    },
    {
        "id": "JIRA-107",
        "summary": "Mobile view layout breaks",
        "description": """
            Given I open site on mobile browser
            When I navigate to any page
            Then the layout should remain responsive and clean
        """
    },
    {
        "id": "JIRA-108",
        "summary": "Notification not shown for new message",
        "description": """
            Given I receive a new message
            When I am on dashboard
            Then a notification banner should appear
        """
    },
    {
        "id": "JIRA-109",
        "summary": "Dark mode theme not applied",
        "description": """
            Given I toggle dark mode
            When I reload the page
            Then the dark theme should persist
        """
    },
    {
        "id": "JIRA-110",
        "summary": "Search result links broken",
        "description": """
            Given I perform a search
            When I click on result links
            Then they should open correct detail pages
        """
    },
]

model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
documents = [f"{issue['summary']} {issue['description']}" for issue in jira_issues]
embeddings = model.encode(documents, normalize_embeddings=True)

# Save embeddings
np.save("embeddings.npy", embeddings)

# Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, "index_store.faiss")

# Save metadata
with open("jira_issues.json", "w") as f:
    json.dump(jira_issues, f, indent=2)