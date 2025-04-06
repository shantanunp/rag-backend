import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

# Simulated Jira data (replace with real data from API later)
jira_issues = [
    {
        "id": "JIRA-1001",
        "summary": "App crashes on login",
        "description": "When the user attempts to log in using valid credentials, the application crashes immediately. "
                       "This issue occurs on both Android and iOS platforms. Logs indicate a null pointer exception "
                       "in the authentication module. Steps to reproduce: 1. Open the app. 2. Enter valid credentials. "
                       "3. Tap the login button. Expected behavior: The user should be logged in successfully."
    },
    {
        "id": "JIRA-1002",
        "summary": "Slow page load",
        "description": "The home page takes an unusually long time to load, especially on slower network connections. "
                       "This issue is affecting user experience significantly. The problem seems to be related to the "
                       "large number of API calls made during the initial page load. Steps to reproduce: 1. Open the app. "
                       "2. Navigate to the home page. Expected behavior: The page should load within 2-3 seconds."
    },
    {
        "id": "JIRA-1003",
        "summary": "Incorrect calculation in cart",
        "description": "The total amount displayed in the shopping cart is incorrect when multiple items are added. "
                       "The issue seems to be related to rounding errors in the pricing calculation. Steps to reproduce: "
                       "1. Add multiple items to the cart. 2. Check the total amount displayed. Expected behavior: The total "
                       "amount should match the sum of the individual item prices."
    },
    {
        "id": "JIRA-1004",
        "summary": "Search functionality not working",
        "description": "The search bar on the website does not return any results, even for valid queries. This issue "
                       "is affecting user navigation and discoverability of products. Steps to reproduce: 1. Enter a valid "
                       "search term in the search bar. 2. Press Enter. Expected behavior: Relevant search results should be displayed."
    },
    {
        "id": "JIRA-1005",
        "summary": "Profile picture upload fails",
        "description": "Users are unable to upload profile pictures. The upload button triggers an error message stating "
                       "'File format not supported,' even for supported formats like JPEG and PNG. Steps to reproduce: 1. Go to the "
                       "profile settings page. 2. Attempt to upload a profile picture. Expected behavior: The profile picture should be uploaded successfully."
    },
    {
        "id": "JIRA-1006",
        "summary": "Notifications not appearing",
        "description": "Push notifications are not being delivered to users. This issue is affecting both Android and iOS platforms. "
                       "Logs indicate a failure in the notification service API. Steps to reproduce: 1. Perform an action that triggers a notification. "
                       "2. Check the device for the notification. Expected behavior: The notification should appear on the user's device."
    },
    {
        "id": "JIRA-1007",
        "summary": "Payment gateway timeout",
        "description": "The payment gateway frequently times out during transactions, causing failed payments. This issue is leading to user frustration "
                       "and loss of revenue. Steps to reproduce: 1. Add items to the cart. 2. Proceed to checkout. 3. Attempt to make a payment. "
                       "Expected behavior: The payment should be processed successfully without any timeouts."
    },
    {
        "id": "JIRA-1008",
        "summary": "Broken links on FAQ page",
        "description": "Several links on the FAQ page are broken, leading to 404 errors. This issue is affecting user access to important information. "
                       "Steps to reproduce: 1. Navigate to the FAQ page. 2. Click on any link. Expected behavior: The link should redirect to the correct page."
    },
    {
        "id": "JIRA-1009",
        "summary": "Dark mode not saving preference",
        "description": "The dark mode setting does not persist after the app is closed and reopened. Users have to manually enable dark mode every time "
                       "they launch the app. Steps to reproduce: 1. Enable dark mode in the settings. 2. Close and reopen the app. Expected behavior: "
                       "The dark mode setting should persist across app sessions."
    },
    {
        "id": "JIRA-1010",
        "summary": "Error in exporting reports",
        "description": "Users are unable to export reports in PDF format. The export button triggers an error message stating 'Export failed due to server error.' "
                       "Steps to reproduce: 1. Navigate to the reports section. 2. Attempt to export a report in PDF format. Expected behavior: The report should be exported successfully."
    }
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
