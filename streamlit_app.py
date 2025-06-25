import streamlit as st
import pandas as pd
import psycopg2
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from prophet import Prophet
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama


# Load environment variables
load_dotenv()

# Load budget config
def load_budget_map():
    try:
        with open("config/category_mapping.json", "r") as f:
            return json.load(f).get("budgets", {})
    except Exception as e:
        st.warning(f"⚠️ Could not load budget file: {e}")
        return {}

# AI categorization fallback (Ollama)
def ai_categorize(description):
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
    except Exception as e:
        return "Uncategorized"

# Auto-tag frequent vendors or subscriptions
def tag_vendor(description):
    vendor_map = {
        "netflix": "Subscription",
        "spotify": "Subscription",
        "uber": "Transport",
        "amazon": "Shopping",
        "walmart": "Groceries"
    }
    desc_lower = description.lower()
    for vendor, tag in vendor_map.items():
        if vendor in desc_lower:
            return tag
    return None

# DB connection
@st.cache_resource
def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="db",
        port="5432"
    )

# Load data
@st.cache_data
def load_data():
    conn = get_conn()
    query = "SELECT * FROM transactions;"
    df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    df["category"] = df.apply(lambda row: row["category"] or tag_vendor(row["description"]) or ai_categorize(row["description"]), axis=1)
    return df.sort_values("date", ascending=False)

# Summary helpers
def summarize(df, budgets):
    grouped = df.groupby("category")["amount"].sum().reset_index()
    grouped["budget"] = grouped["category"].map(budgets).fillna(0)
    grouped["over_under"] = grouped["amount"] - grouped["budget"]
    return grouped.sort_values("amount", ascending=False)

def month_summary(df):
    df["month"] = df["date"].dt.to_period("M")
    return df.groupby(["month", "category"])["amount"].sum().unstack(fill_value=0)

# Forecasting
def forecast_category(df, category):
    cat_df = df[df["category"] == category].copy()
    cat_df["date_period"] = cat_df["date"].dt.to_period("M")
    cat_df = (
        cat_df
        .groupby("date_period")[["amount"]]  # specify only numeric column(s)
        .sum()
        .reset_index()
    )
    cat_df["ds"] = cat_df["date"].dt.to_timestamp()
    cat_df["y"] = cat_df["amount"]
    m = Prophet()
    m.fit(cat_df[["ds", "y"]])
    future = m.make_future_dataframe(periods=2, freq="M")
    forecast = m.predict(future)
    return forecast[["ds", "yhat"]].tail(3)

# Budget Rebalancer
def ai_budget_rebalance(df, target_total):
    category_totals = df.groupby("category")["amount"].sum().to_dict()
    prompt = f"Given these past monthly expenses: {category_totals}, distribute a ${target_total} budget across these categories. Return JSON."
    try:
        resp = requests.post("http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}, timeout=10)
        return json.loads(resp.json().get("response", "{}"))
    except:
        return {}

# LangChain-powered chat on finances
def init_chat(df):
    loader = DataFrameLoader(df)
    docs = loader.load()
    db = FAISS.from_documents(docs, OllamaEmbeddings())
    qa = RetrievalQA.from_chain_type(llm=Ollama(model="mistral"), retriever=db.as_retriever())
    return qa

# --- UI ---
st.set_page_config(page_title="FinSight-AI", layout="wide")
page = option_menu("FinSight-AI", ["Overview", "Trends", "Raw Data", "💬 Ask AI"], orientation="horizontal")

df = load_data()
budgets = load_budget_map()

if page == "Overview":
    st.title("📊 Overview")
    summary = summarize(df, budgets)
    st.dataframe(summary)
    st.bar_chart(summary.set_index("category")["amount"])
    st.bar_chart(summary.set_index("category")[["budget", "amount"]])

    st.subheader("💰 Income vs Expenses")
    income = df[df["amount"] < 0]["amount"].sum()
    expenses = df[df["amount"] > 0]["amount"].sum()
    col1, col2 = st.columns(2)
    col1.metric("💸 Expenses", f"${expenses:,.2f}")
    col2.metric("💰 Income", f"${abs(income):,.2f}")

elif page == "Trends":
    st.title("📈 Monthly Trends & Forecasts")
    monthly = month_summary(df)
    st.line_chart(monthly)

    category = st.selectbox("📂 Forecast Category", monthly.columns.tolist())
    forecast = forecast_category(df, category)
    st.line_chart(forecast.set_index("ds"))

    st.subheader("🧮 Budget Rebalancer")
    target = st.number_input("Target Monthly Budget ($)", value=3000)
    if st.button("Suggest Budget"):
        new_budget = ai_budget_rebalance(df, target)
        st.json(new_budget)

elif page == "Raw Data":
    st.title("📄 All Transactions")
    st.dataframe(df, use_container_width=True)

elif page == "💬 Ask AI":
    st.title("💬 Ask About Your Finances")
    qa = init_chat(df)
    query = st.text_input("What would you like to know?", placeholder="e.g. How much did I spend on groceries last month?")
    if query:
        answer = qa.run(query)
        st.write(answer)
