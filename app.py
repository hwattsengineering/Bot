import os
import glob
import pandas as pd
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# ——————————————————————————————
# 1) Load & concatenate all CSVs in data/
# ——————————————————————————————
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
csv_paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))

dfs = []
for path in csv_paths:
    try:
        dfs.append(pd.read_csv(path))
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")

all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ——————————————————————————————
# 2) Normalize & ensure expected columns
# ——————————————————————————————
all_df.rename(columns={
    "report_id":         "id",
    "equipment_id":      "equipment",
    "fault_description": "issue",
    "corrective_action": "fix",
    "inspection_date":   "date"
}, inplace=True)

for col in ("id","equipment","issue","fix","date"):
    if col not in all_df.columns:
        all_df[col] = ""
    all_df[col] = all_df[col].fillna("").astype(str)

# ——————————————————————————————
# 3) Initialize ChromaDB & OpenAI
# ——————————————————————————————
# Uses local DuckDB+Parquet persistence in .chromadb/
client     = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb"
))
collection = client.get_or_create_collection("service_reports")
openai_api = OpenAI()  # reads OPENAI_API_KEY from env

# ——————————————————————————————
# 4) Index every record once at startup
# ——————————————————————————————
for idx, row in all_df.iterrows():
    # Create a simple text fingerprint from issue+fix
    text = f"{row['issue']} {row['fix']}"
    emb  = openai_api.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )["data"][0]["embedding"]
    collection.add(
        ids=[str(row["id"])],
        embeddings=[emb],
        metadatas=[{
            "id":        row["id"],
            "equipment": row["equipment"],
            "issue":     row["issue"],
            "fix":       row["fix"],
            "date":      row["date"]
        }],
        documents=[text]
    )

# ——————————————————————————————
# 5) Flask & Twilio setup
# ——————————————————————————————
app = Flask(__name__)

def generate_prompt(question: str, top_k: int = 5) -> str:
    # Embed the question
    q_emb = openai_api.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )["data"][0]["embedding"]

    # Retrieve top_k matching records
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    hits = results["metadatas"][0]  # list of record dicts

    # Build concise prompt
    prompt = (
        "You are a CryoFERM AI assistant. Use these past service records to answer:\n\n"
    )
    for rpt in hits:
        prompt += (
            f"- Report {rpt['id']} | Equipment: {rpt['equipment']} | Date: {rpt['date']}\n"
            f"  Issue: {rpt['issue']}\n"
            f"  Fix: {rpt['fix']}\n"
        )
    prompt += f"\nTechnician's question: {question}\nAnswer:"
    return prompt

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    incoming = request.values.get("Body", "").strip()
    if not incoming:
        return "OK"

    prompt = generate_prompt(incoming, top_k=5)
    try:
        resp = openai_api.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error: {e}"

    twiml = MessagingResponse()
    twiml.message(answer)
    return str(twiml)

if __name__ == "__main__":
    # Local debug server
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
