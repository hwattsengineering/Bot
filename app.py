import os
import glob
import threading
import datetime
import csv

import pandas as pd
import chromadb
from openai import OpenAI
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# ——————————————————————————————
# 1) Load & concatenate all CSVs under data/
# ——————————————————————————————
BASE     = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
csv_paths= glob.glob(os.path.join(DATA_DIR, "*.csv"))

dfs = []
for path in csv_paths:
    try:
        dfs.append(pd.read_csv(path, dtype=str))
    except Exception as e:
        print(f"⚠️ Warning, couldn’t read {path}: {e}")

all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ——————————————————————————————
# 2) Normalize columns to our schema
# ——————————————————————————————
all_df.rename(columns={
    "report_id":         "id",
    "equipment_id":      "equipment",
    "inspection_date":   "date",
    "fault_description": "issue",
    "corrective_action": "fix",
    "author":            "technician",
    "Prepared by":       "technician",
}, inplace=True)

# keep only the columns we need, fill blanks
wanted = ["id", "equipment", "date", "issue", "fix", "technician"]
all_df = all_df.reindex(columns=wanted).fillna("").astype(str)
# drop any truly blank rows
all_df = all_df[all_df["id"].str.strip() != ""]

# ——————————————————————————————
# 3) Set up ChromaDB & OpenAI client
# ——————————————————————————————
client     = chromadb.Client()
collection = client.get_or_create_collection("service_reports")
openai_api = OpenAI()        # uses OPENAI_API_KEY

# ——————————————————————————————
# 4) Index all records once (thread‑safe)
# ——————————————————————————————
_index_lock   = threading.Lock()
_indexed_flag = False

def ensure_indexed():
    global _indexed_flag
    if _indexed_flag:
        return
    with _indexed_lock:
        if _indexed_flag:
            return

        for _, row in all_df.iterrows():
            text = f"{row['issue']} {row['fix']}"
            emb  = openai_api.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            ).data[0].embedding

            collection.add(
                ids=[row["id"]],
                embeddings=[emb],
                metadatas=[row.to_dict()],
                documents=[text]
            )

        _indexed_flag = True
        print("✅ Indexed all service records")

# ——————————————————————————————
# 5) Learning machinery
# ——————————————————————————————
LEARNED_CSV     = os.path.join(DATA_DIR, "learned.csv")
_pending_learns = {}   # phone_number → original_question

def learn_record(original_q: str, provided_a: str):
    """Append Q&A to learned.csv and index immediately."""
    ts     = int(datetime.datetime.now().timestamp())
    new_id = f"LEARNED-{ts}"
    today  = datetime.date.today().isoformat()
    row    = {
        "id":         new_id,
        "equipment":  "",
        "date":       today,
        "issue":      original_q,
        "fix":        provided_a,
        "technician": "",
    }

    # append to CSV
    exists = os.path.isfile(LEARNED_CSV)
    with open(LEARNED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    # immediately index it
    text = f"{row['issue']} {row['fix']}"
    emb  = openai_api.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding

    collection.add(
        ids=[row["id"]],
        embeddings=[emb],
        metadatas=[row],
        documents=[text]
    )

# ——————————————————————————————
# 6) Helper: last‑job lookup by technician name
# ——————————————————————————————
def find_last_job(name: str):
    df = all_df[all_df["technician"].str.contains(name, case=False, na=False)]
    if df.empty:
        return None
    df2  = df.assign(_dt=pd.to_datetime(df["date"], errors="coerce"))
    last = df2.sort_values("_dt", ascending=False).iloc[0]
    return (
        f"{name}'s last job (Report {last['id']} on {last['date']}):\n"
        f"Issue: {last['issue']}\n"
        f"Fix:   {last['fix']}"
    )

# ——————————————————————————————
# 7) Build a GPT prompt from top‑k similar records
# ——————————————————————————————
def create_prompt(question: str, k=5):
    qemb = openai_api.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding
    results = collection.query(query_embeddings=[qemb], n_results=k)
    hits    = results["metadatas"][0]

    prompt = (
        "You are a CryoFERM AI assistant. You ONLY answer from the records below. "
        "If none applies, you will ask to learn.\n\n"
    )
    for rpt in hits:
        prompt += (
            f"- Report {rpt['id']} | Equip: {rpt['equipment']} | Date: {rpt['date']} | Tech: {rpt['technician']}\n"
            f"  Issue: {rpt['issue']}\n"
            f"  Fix:   {rpt['fix']}\n\n"
        )

    prompt += f"Tech question: {question}\nAnswer:"
    return prompt

# ——————————————————————————————
# 8) Flask + Twilio integration
# ——————————————————————————————
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    ensure_indexed()

    frm = request.values.get("From")
    txt = request.values.get("Body", "").strip() or ""
    resp= MessagingResponse()

    # — Phase 2 of learning: user just sent the answer
    if frm in _pending_learns:
        orig = _pending_learns.pop(frm)
        learn_record(orig, txt)
        resp.message("Thanks! I’ve learned that and will remember it.")
        return str(resp)

    # — Special: “what did X do on their last job?”
    import re
    m = re.search(r"what did (\w+) do on (?:his|her|their) last job", txt, re.IGNORECASE)
    if m:
        answer = find_last_job(m.group(1))
        resp.message(answer or "I couldn’t find that technician in the records.")
        return str(resp)

    # — Standard: similarity search + ChatCompletion
    prompt = create_prompt(txt)
    chat   = openai_api.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    answer = chat.choices[0].message.content.strip()

    # — If our answer invited learning, queue phase 2
    if "Would you like me to learn" in answer:
        _pending_learns[frm] = txt

    resp.message(answer)
    return str(resp)

if __name__ == "__main__":
    # Bind to 0.0.0.0 and the `$PORT` env var (for Render/Railway/etc.)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
