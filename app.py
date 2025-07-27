import os
import glob
import threading
import re
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
        print(f"Warning reading {path}: {e}")

all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ——————————————————————————————
# 2) Normalize to expected columns
# ——————————————————————————————
desired_cols = [
    "id",
    "date",
    "equipment_code",
    "equipment_name",
    "issue",
    "solution",
    "technician",
    "tags"
]
all_df = all_df.reindex(columns=desired_cols).fillna("").astype(str)

# Drop records missing an ID or date
all_df = all_df[all_df["id"].str.strip() != ""]
all_df = all_df[all_df["date"].str.strip() != ""]

# ——————————————————————————————
# 3) Set up ChromaDB & OpenAI
# ——————————————————————————————
client     = chromadb.Client()
collection = client.get_or_create_collection("service_records")
openai_api = OpenAI()  # uses OPENAI_API_KEY from env

# ——————————————————————————————
# 4) Lazy‐indexing on first request
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
            # Combine issue + solution + tags for embedding
            text = f"Issue: {row['issue']}  Solution: {row['solution']}  Tags: {row['tags']}"
            resp = openai_api.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            emb = resp.data[0].embedding
            collection.add(
                ids=[row["id"]],
                embeddings=[emb],
                metadatas=[{
                    "id":             row["id"],
                    "date":           row["date"],
                    "equipment_code": row["equipment_code"],
                    "equipment_name": row["equipment_name"],
                    "issue":          row["issue"],
                    "solution":       row["solution"],
                    "technician":     row["technician"],
                    "tags":           row["tags"],
                }],
                documents=[text]
            )
        _indexed_flag = True
        print("🔍 Indexed all service records")

# ——————————————————————————————
# 5) Flask + Twilio setup
# ——————————————————————————————
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return "OK", 200

def handle_last_job_query(question: str):
    m = re.search(
        r"what did (\w+) do on (?:his|her|their) last job",
        question, re.IGNORECASE
    )
    if not m:
        return None
    name = m.group(1)
    df = all_df[all_df["technician"].str.contains(name, case=False)]
    if df.empty:
        return f"Sorry, I don’t see any jobs by {name}."
    df = df.assign(_dt=pd.to_datetime(df["date"], errors="coerce"))
    last = df.sort_values("_dt", ascending=False).iloc[0]
    return (
        f"{name}'s last job (Report {last['id']} on {last['date']}):\n"
        f"Issue: {last['issue']}\n"
        f"Solution: {last['solution']}"
    )

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

    prompt = (
        "You are a CryoFERM AI assistant. You have these service records:\n"
        "ONLY answer based on them. Do NOT invent details.\n"
        "If no record answers the question, say so and ask if it's OK to search externally.\n\n"
    )
    for rpt in hits:
        prompt += (
            f"- ID {rpt['id']} | {rpt['equipment_code']} {rpt['equipment_name']} | Date {rpt['date']} | Tech {rpt['technician']}\n"
            f"  Issue: {rpt['issue']}\n"
            f"  Solution: {rpt['solution']}\n\n"
        )
    prompt += f"Technician question: {question}\nAnswer:"
    return prompt

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    ensure_index()

    incoming = request.values.get("Body", "").strip()
    if not incoming:
        return "OK"

    # 1) Direct last‐job lookup
    direct = handle_last_job_query(incoming)
    if direct:
        reply = direct
    else:
        # 2) Retrieval + LLM
        prompt = generate_prompt(incoming, top_k=5)
        resp   = openai_api.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3
        )
        answer = resp.choices[0].message.content.strip()
        if re.search(r"don’t have|don't have|no record", answer, re.IGNORECASE):
            reply = f"{answer}\n\nWould you like me to search externally?"
        else:
            reply = answer

    twiml = MessagingResponse()
    twiml.message(reply)
    return str(twiml)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
