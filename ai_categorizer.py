# ai_categorizer.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def categorize_transaction(description):
    prompt = f"""You are a financial categorization engine. Given the following transaction description, return just one word that best represents the category it belongs to (e.g., "Groceries", "Dining", "Travel", "Gas", "Subscriptions", "Utilities", "Income").

Transaction: "{description}"
Category:"""

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip().split()[0]  # clean output
    except Exception as e:
        print(f"❌ AI categorization failed: {e}")
        return "Uncategorized"
