import pandas as pd
import os
import requests
from pathlib import Path
import json

INCOMING_DIR = Path("./incoming")
PROCESSED_DIR = Path("./processed")
NORMALIZED_DIR = Path("./normalized")
TAG_FEEDBACK_FILE = Path("./config/feedback_tags.json")

# Load filter keywords
def load_filter_keywords(path="./config/filter_keywords.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [line.strip().lower() for line in f.readlines() if line.strip()]

FILTER_KEYWORDS = load_filter_keywords()

# Load previous tag feedback
def load_feedback_tags():
    if TAG_FEEDBACK_FILE.exists():
        with open(TAG_FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {}

# Save updated feedback tags
def save_feedback_tags(tag_dict):
    with open(TAG_FEEDBACK_FILE, "w") as f:
        json.dump(tag_dict, f, indent=2)

FEEDBACK_TAGS = load_feedback_tags()

# Enrich metadata from external API (dummy OpenCorporates-style logic)
def enrich_merchant(description):
    try:
        response = requests.get(f"https://autocomplete.clearbit.com/v1/companies/suggest?query={description}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data[0]['name'] if data else description
    except:
        pass
    return description

# AI category fallback (Ollama)
def ai_categorize(description):
    if description in FEEDBACK_TAGS:
        return FEEDBACK_TAGS[description]
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"Suggest a personal finance category for this transaction: '{description}'",
                "stream": False
            }, timeout=10
        )
        return resp.json().get("response", "Uncategorized").strip()
    except:
        return "Uncategorized"

# Auto-tag known vendors
VENDOR_MAP = {
    "netflix": "Subscription",
    "spotify": "Subscription",
    "uber": "Transport",
    "amazon": "Shopping",
    "walmart": "Groceries"
}

def tag_vendor(description):
    desc_lower = description.lower()
    for vendor, tag in VENDOR_MAP.items():
        if vendor in desc_lower:
            return tag
    return None

# Exclude based on keywords
def should_exclude(description):
    desc = str(description).lower()
    return any(keyword in desc for keyword in FILTER_KEYWORDS)

# Category logic

def get_category(description):
    return tag_vendor(description) or ai_categorize(description)

# Format detection
def detect_format(df):
    cols = set(df.columns)
    if "Appears On Your Statement As" in cols:
        return "amex"
    elif {"Transaction Date", "Post Date", "Description", "Category", "Type", "Amount", "Memo"}.issubset(cols):
        return "chase"
    elif {"Transaction Date", "Description", "Points", "Category"}.issubset(cols) and "Card No." not in cols:
        return "bilt"
    elif {"Posted Date", "Card No.", "Debit", "Credit"}.issubset(cols):
        return "capone"
    else:
        return "unknown"

# Normalize logic
def normalize(df, source):
    df = df[~df["Description"].apply(should_exclude)]

    if source == "amex":
        return pd.DataFrame({
            "date": pd.to_datetime(df["Date"]),
            "description": df["Description"].apply(enrich_merchant),
            "amount": df["Amount"].astype(float),
            "category": df["Description"].apply(get_category),
            "source": "Amex"
        })

    elif source == "chase":
        df["Amount"] = df["Amount"].astype(str).str.replace(",", "").astype(float).abs()
        return pd.DataFrame({
            "date": pd.to_datetime(df["Transaction Date"]),
            "description": df["Description"].apply(enrich_merchant),
            "amount": df["Amount"],
            "category": df["Description"].apply(get_category),
            "source": "Chase"
        })

    elif source == "bilt":
        return pd.DataFrame({
            "date": pd.to_datetime(df["Transaction Date"]),
            "description": df["Description"].apply(enrich_merchant),
            "amount": df["Amount"].astype(float),
            "category": df["Description"].apply(get_category),
            "source": "Bilt"
        })

    elif source == "capone":
        df["amount"] = df["Debit"].fillna(0).astype(float) - df["Credit"].fillna(0).astype(float)
        return pd.DataFrame({
            "date": pd.to_datetime(df["Transaction Date"]),
            "description": df["Description"].apply(enrich_merchant),
            "amount": df["amount"],
            "category": df["Description"].apply(get_category),
            "source": "CapitalOne"
        })

    else:
        raise ValueError(f"Unsupported CSV format: {source}")

# Pipeline core
def process_csv(file_path):
    df = pd.read_csv(file_path)
    source = detect_format(df)
    if source == "unknown":
        print(f"⚠️ Unknown format: {file_path.name}")
        return
    norm_df = normalize(df, source)
    norm_file = NORMALIZED_DIR / f"{file_path.stem}_normalized.csv"
    norm_df.to_csv(norm_file, index=False)
    file_path.rename(PROCESSED_DIR / file_path.name)
    print(f"✅ Processed: {file_path.name} → {norm_file.name}")

# Feedback API for manual tag corrections (to be called from UI)
def update_tag_feedback(description, corrected_tag):
    FEEDBACK_TAGS[description] = corrected_tag
    save_feedback_tags(FEEDBACK_TAGS)

# Main runner
def run_pipeline():
    for file in INCOMING_DIR.glob("*.csv"):
        try:
            process_csv(file)
        except Exception as e:
            print(f"❌ Failed: {file.name} - {e}")

if __name__ == "__main__":
    run_pipeline()
