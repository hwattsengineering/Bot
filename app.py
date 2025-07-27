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
        print(f"Warning: failed to read {path}: {e}")

all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ——————————————————————————————
# 2) Normalize & map columns
# ——————————————————————————————
all_df.rename(columns={
    "report_id":                                  "id",
    "equipment_id":                               "equipment",
    "inspection_date":                            "date",
    "fault_description":                          "issue",
    "Description of works carried out":           "fix",
    "Fix_Solution":                               "fix",
    "Fix_solution":                               "fix",
    "FST Report_Completion _Sign Off Service Tech:": "technician",
    "Prepared by":                                "technician",
}, inplace=True)

# Reindex once to avoid fragmentation, fill missing with empty strings, cast all to str
desired_cols = ["id","equipment","date","issue","fix","technician"]
all_df = all_df.reindex(columns=desired_cols).fillna("").astype(str)

# Drop any records without an ID
all_df = all_df[all_df["id"].str.strip() != ""]

# ——————————————————————————————
# 3) Prepare ChromaDB & OpenAI
# ——————————————————————————————
client     = chromadb.Client()  
collection = client.get_or_create_collection("service_reports")
openai_api = OpenAI()  # uses OPENAI_API_KEY

# ——————————————————————————————
# 4) Lazy indexing setup
# ——————————————————————————————
_index_lock   = threading.Lock()
_indexed_flag = False

def ensure_index():
    global _indexed_flag
    if _indexed_flag:
        return
    with _indexed_lock:
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
                    "id":         row["id"],
                    "equipment":  row["equipment"],
                    "date":       row["date"],
                    "issue":      row["issue"],
                    "fix":        row["fix"],
                    "technician": row["technician"],
                }],
                documents=[text]
            )
        _indexed_flag = True
        print("🔍 Indexed all service records into ChromaDB")

# ——————————————————————————————
# 5) Flask & Twilio setup
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
        return f"Sorry, I don’t see any jobs by {name} in our records."
    df = df.assign(_dt=pd.to_datetime(df["date"], errors="coerce"))
    last = df.sort_values("_dt", ascending=False).iloc[0]
    return (
        f"{name}'s last job (Report {last['id']} on {last['date']}):\n"
        f"Issue: {last['issue']}\n"
        f"Fix: {last['fix']}"
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
        "You are a CryoFERM AI assistant. You have access to these service records.\n"
        "ONLY answer based on these records. Do NOT invent any details.\n"
        "If the answer isn’t in the records above, say so and ask if it’s OK to search externally.\n\n"
    )
    for rpt in hits:
        prompt += (
            f"- Report {rpt['id']} | Equipment: {rpt['equipment']} | Date: {rpt['date']} |"
            f" Technician: {rpt['technician']}\n"
            f"  Issue: {rpt['issue']}\n"
            f"  Fix: {rpt['fix']}\n\n"
        )
    prompt += f"Technician question: {question}\nAnswer:"
    return prompt

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    ensure_index()

    incoming = request.values.get("Body", "").strip()
    if not incoming:
        return "OK"

    direct = handle_last_job_query(incoming)
    if direct:
        reply = direct
    else:
        prompt = generate_prompt(incoming, top_k=5)
        resp   = openai_api.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3
        )
        answer = resp.choices[0].message.content.strip()
        if re.search(r"don’t have|don't have|no records", answer, re.IGNORECASE):
            reply = f"{answer}\n\nWould you like me to search external resources?"
        else:
            reply = answer

    twiml = MessagingResponse()
    twiml.message(reply)
    return str(twiml)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
