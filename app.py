import os
import glob
import threading
import pandas as pd
import chromadb
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
    if col in all_df.columns:
        all_df[col] = all_df[col].fillna("").astype(str)
    else:
        all_df[col] = ""

# Drop any records without an ID
all_df = all_df[all_df["id"].str.strip() != ""]

# ——————————————————————————————
# 3) Prepare ChromaDB & OpenAI client
# ——————————————————————————————
client     = chromadb.Client()  
collection = client.get_or_create_collection("service_reports")
openai_api = OpenAI()            # reads OPENAI_API_KEY

# ——————————————————————————————
# 4) Lazy indexing setup
# ——————————————————————————————
_index_lock   = threading.Lock()
_indexed_flag = False

def ensure_index():
    global _indexed_flag
    if _indexed_flag:
        return
    with _index_lock:
        if _indexed_flag:
            return
        for _, row in all_df.iterrows():
            text = f"{row['issue']} {row['fix']}"
            resp = openai_api.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            emb = resp.data[0].embedding
            collection.add(
                ids=[row["id"]],
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
        _indexed_flag = True
        print("🔍 Indexed all service records into ChromaDB")

# ——————————————————————————————
# 5) Flask & Twilio setup
# ——————————————————————————————
app = Flask(__name__)

# Health check for GET /
@app.route("/", methods=["GET"])
def health_check():
    return "OK", 200

def generate_prompt(question: str, top_k: int = 5) -> str:
    resp  = openai_api.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    q_emb = resp.data[0].embedding

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    hits = results["metadatas"][0]

    prompt = "You are a CryoFERM AI assistant. Use these past service records to answer:\n\n"
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
    # Build index on first request
    ensure_index()

    incoming = request.values.get("Body", "").strip()
    if not incoming:
        return "OK"

    prompt = generate_prompt(incoming, top_k=5)
    try:
        chat = openai_api.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = chat.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error: {e}"

    twiml = MessagingResponse()
    twiml.message(answer)
    return str(twiml)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
