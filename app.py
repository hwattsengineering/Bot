import os
import glob
import threading
import datetime
import csv
import logging

import pandas as pd
import chromadb
from openai import OpenAI
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse

# ——————————————————————————————
# 0) Setup logging
# ——————————————————————————————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cryobot")

# ——————————————————————————————
# 1) Load & concatenate CSVs
# ——————————————————————————————
BASE     = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
csvs     = glob.glob(os.path.join(DATA_DIR, "*.csv"))

dfs = []
for path in csvs:
    try:
        dfs.append(pd.read_csv(path, dtype=str))
    except Exception as e:
        logger.warning(f"Couldn’t read {path}: {e}")
all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ——————————————————————————————
# 2) Normalize columns
# ——————————————————————————————
all_df.rename(columns={
    "report_id":        "id",
    "equipment_id":     "equipment",
    "inspection_date":  "date",
    "fault_description":"issue",
    "corrective_action":"fix",
    "author":           "technician",
    "Prepared by":      "technician",
}, inplace=True)

wanted = ["id","equipment","date","issue","fix","technician"]
all_df   = all_df.reindex(columns=wanted).fillna("").astype(str)
all_df   = all_df[all_df["id"].str.strip() != ""]

# ——————————————————————————————
# 3) ChromaDB & OpenAI
# ——————————————————————————————
client     = chromadb.Client()
collection = client.get_or_create_collection("service_reports")
openai_api = OpenAI()

# ——————————————————————————————
# 4) Lazy indexing
# ——————————————————————————————
_index_lock   = threading.Lock()
_indexed_flag = False

def ensure_indexed():
    global _indexed_flag
    if _indexed_flag:
        return
    with _index_lock:
        if _indexed_flag:
            return
        logger.info("🔍 Indexing records…")
        for _, row in all_df.iterrows():
            text = f"{row['issue']} {row['fix']}"
            emb  = openai_api.embeddings.create(
                input=text, model="text-embedding-3-small"
            ).data[0].embedding
            collection.add(
                ids=[row["id"]],
                embeddings=[emb],
                metadatas=[row.to_dict()],
                documents=[text]
            )
        _indexed_flag = True
        logger.info("✅ Indexed all service records")

# ——————————————————————————————
# 5) Learning machinery
# ——————————————————————————————
LEARNED_CSV     = os.path.join(DATA_DIR, "learned.csv")
_pending_learns = {}

def learn_record(orig_q: str, answer: str):
    ts     = int(datetime.datetime.now().timestamp())
    new_id = f"LEARNED-{ts}"
    today  = datetime.date.today().isoformat()
    row    = {
        "id":         new_id,
        "equipment":  "",
        "date":       today,
        "issue":      orig_q,
        "fix":        answer,
        "technician": "",
    }
    exists = os.path.isfile(LEARNED_CSV)
    with open(LEARNED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    text = f"{row['issue']} {row['fix']}"
    emb  = openai_api.embeddings.create(
        input=text, model="text-embedding-3-small"
    ).data[0].embedding
    collection.add(
        ids=[row["id"]],
        embeddings=[emb],
        metadatas=[row],
        documents=[text]
    )
    logger.info(f"📝 Learned new record {new_id}")

# ——————————————————————————————
# 6) Last‐job helper
# ——————————————————————————————
def find_last_job(name: str):
    df = all_df[all_df["technician"].str.contains(name, case=False, na=False)]
    if df.empty:
        return None
    df2  = df.assign(_dt=pd.to_datetime(df["date"], errors="coerce"))
    last = df2.sort_values("_dt", ascending=False).iloc[0]
    return f"{name}'s last job (Report {last['id']} on {last['date']}):\nIssue: {last['issue']}\nFix: {last['fix']}"

# ——————————————————————————————
# 7) Build prompt
# ——————————————————————————————
def create_prompt(q: str, k=5):
    qemb    = openai_api.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
    results = collection.query(query_embeddings=[qemb], n_results=k)
    hits    = results["metadatas"][0]
    logger.info(f"👀 Hits: {[h['id'] for h in hits]}")
    prompt  = "You are a bot. Only use these records. If none apply, ask to learn.\n\n"
    for r in hits:
        prompt += f"- {r['id']} | {r['equipment']} | {r['date']}\n  Issue: {r['issue']}\n  Fix: {r['fix']}\n\n"
    prompt += f"Question: {q}\nAnswer:"
    return prompt

# ——————————————————————————————
# 8) Flask routes
# ——————————————————————————————
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/debug/hits", methods=["GET"])
def debug_hits():
    # show last query hits (if any)
    return jsonify(last_hits=[h['id'] for h in collection.peek().metadatas]), 200

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    ensure_indexed()

    frm = request.values.get("From")
    txt = (request.values.get("Body") or "").strip()
    logger.info(f"❓ Received from {frm!r}: {txt!r}")

    tw = MessagingResponse()

    # 1) Phase 2 learning?
    if frm in _pending_learns:
        orig = _pending_learns.pop(frm)
        learn_record(orig, txt)
        tw.message("Thanks! I’ve learned that and will remember it.")
        return str(tw)

    # 2) Last‐job?
    import re
    m = re.search(r"what did (\w+) do on (?:his|her|their) last job", txt, re.IGNORECASE)
    if m:
        ans = find_last_job(m.group(1)) or "I couldn’t find that technician."
        tw.message(ans)
        return str(tw)

    # 3) Retrieval + Chat
    prompt = create_prompt(txt)
    try:
        chat = openai_api.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        answer = chat.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error: {e}"

    # 4) Queue learn if asked
    if "Would you like me to learn" in answer:
        _pending_learns[frm] = txt

    tw.message(answer)
    return str(tw)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
