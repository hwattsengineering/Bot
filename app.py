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
BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
csv_paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))

dfs = []
for path in csv_paths:
    try:
        dfs.append(pd.read_csv(path))
    except Exception as e:
        print(f"⚠️ Warning, couldn’t read {path}: {e}")

all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ——————————————————————————————
# 2) Rename & reindex to our schema
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

wanted = ["id", "equipment", "date", "issue", "fix", "technician"]
all_df = all_df.reindex(columns=wanted).fillna("").astype(str)
all_df = all_df[all_df["id"].str.strip() != ""]  # drop blanks

# ——————————————————————————————
# 3) Set up Chroma & OpenAI
# ——————————————————————————————
client     = chromadb.Client()
collection = client.get_or_create_collection("service_reports")
openai_api = OpenAI()  # relies on OPENAI_API_KEY env var

# ——————————————————————————————
# 4) Index once on first request
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
_pending_learns = {}  # phone_number -> original_question

def learn_record(original_q: str, provided_a: str):
    """Append to learned.csv and index immediately."""
    new_id = f"LEARNED-{int(datetime.datetime.now().timestamp())}"
    today  = datetime.date.today().isoformat()
    row = {
        "id":         new_id,
        "equipment":  "",
        "date":       today,
        "issue":      original_q,
        "fix":        provided_a,
        "technician": "",
    }
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
# 6) Flask & Twilio
# ——————————————————————————————
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

def find_last_job(name: str):
    matches = all_df[all_df["technician"].str.contains(name, case=False, na=False)]
    if matches.empty:
        return None
    df2 = matches.assign(_dt=pd.to_datetime(matches["date"], errors="coerce"))
    last = df2.sort_values("_dt", ascending=False).iloc[0]
    return (f"{name}'s last job (Report {last['id']} on {last['date']}):\n"
            f"Issue: {last['issue']}\nFix: {last['fix']}")

def create_prompt(question: str, k=5):
    # embed the question
    qemb = openai_api.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding
    results = collection.query(query_embeddings=[qemb], n_results=k)
    hits    = results["metadatas"][0]

    prompt = (
        "You are a CryoFERM AI assistant. You only answer from the records below. "
        "If it’s not here, you’ll ask to learn.\n\n"
    )
    for rpt in hits:
        prompt += (
            f"- Report {rpt['id']} | Equip: {rpt['equipment']} | Date: {rpt['date']} | Tech: {rpt['technician']}\n"
            f"  Issue: {rpt['issue']}\n"
            f"  Fix:   {rpt['fix']}\n\n"
        )
    prompt += f"Tech question: {question}\nAnswer:"
    return prompt

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    ensure_indexed()
    frm = request.values.get("From")
    txt = request.values.get("Body", "").strip()
    if not txt:
        return "OK"

    # Phase 2 of learning?
    if frm in _pending_learns:
        orig = _pending_learns.pop(frm)
        learn_record(orig, txt)
        resp = MessagingResponse()
        resp.message("Thanks! I’ve learned that and will remember it.")
        return str(resp)

    # direct “last job” questions
    import re
    m = re.search(r"what did (\w+) do on (?:his|her|their) last job", txt, re.IGNORECASE)
    if m:
        answer = find_last_job(m.group(1))
        if answer:
            resp = MessagingResponse()
            resp.message(answer)
            return str(resp)

    # standard similarity search + Chat
    prompt = create_prompt(txt)
    chat   = openai_api.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    answer = chat.choices[0].message.content.strip()

    # if we asked it to learn, queue phase 2
    if "Would you like me to learn" in answer:
        _pending_learns[frm] = txt

    tw = MessagingResponse()
    tw.message(answer)
    return str(tw)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
