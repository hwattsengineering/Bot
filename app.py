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
# 1) Paths & load CSVs
# ——————————————————————————————
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CSV_PATHS = glob.glob(os.path.join(DATA_DIR, "*.csv"))

def load_all_data():
    dfs = []
    for path in CSV_PATHS:
        try:
            dfs.append(pd.read_csv(path, dtype=str))
        except Exception as e:
            print(f"Warning reading {path}: {e}")
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    # ensure we have the expected columns
    desired = ["id","date","equipment_code","equipment_name","issue","solution","technician","tags"]
    df = df.reindex(columns=desired).fillna("").astype(str)
    # drop invalid rows
    df = df[df["id"].str.strip() != ""]
    df = df[df["date"].str.strip() != ""]
    return df

all_df = load_all_data()

# ——————————————————————————————
# 2) ChromaDB & OpenAI setup
# ——————————————————————————————
client     = chromadb.Client()
collection = client.get_or_create_collection("service_records")
openai_api = OpenAI()  # uses OPENAI_API_KEY

# ——————————————————————————————
# 3) Lazy indexing
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
            text = f"Issue: {row['issue']}  Solution: {row['solution']}  Tags: {row['tags']}"
            resp = openai_api.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            emb = resp.data[0].embedding
            collection.add(
                ids=[row["id"]],
                embeddings=[emb],
                metadatas=[row.to_dict()],
                documents=[text]
            )
        _indexed_flag = True
        print("🔍 Indexed all service records")

# ——————————————————————————————
# 4) Feedback handler
# ——————————————————————————————
feedback_re = re.compile(r"^CORRECT\s+(\w+)\s*:\s*(.+)$", re.IGNORECASE)

def handle_feedback(report_id: str, new_solution: str) -> str:
    # Update CSV file(s) on disk
    updated = False
    for path in CSV_PATHS:
        df = pd.read_csv(path, dtype=str)
        if report_id in df.get("id", []):
            df.loc[df["id"] == report_id, "solution"] = new_solution
            df.to_csv(path, index=False)
            updated = True
    if not updated:
        return f"Sorry, I couldn’t find report {report_id}."

    # Reload all_df & clear index flag so we can re-index
    global all_df, _indexed_flag
    all_df = load_all_data()
    _indexed_flag = False
    try:
        # Remove old record then re-index only the corrected one
        collection.delete(ids=[report_id])
    except Exception:
        pass
    ensure_index()
    return f"✅ Updated report {report_id} with new solution."

# ——————————————————————————————
# 5) Flask & Twilio setup
# ——————————————————————————————
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return "OK", 200

def handle_last_job_query(question: str):
    m = re.search(r"what did (\w+) do on (?:his|her|their) last job", question, re.IGNORECASE)
    if not m:
        return None
    name = m.group(1)
    df = all_df[all_df["technician"].str.contains(name, case=False)]
    if df.empty:
        return f"Sorry, no jobs found for {name}."
    df = df.assign(_dt=pd.to_datetime(df["date"], errors="coerce"))
    last = df.sort_values("_dt", ascending=False).iloc[0]
    return (f"{name}'s last job (Report {last['id']} on {last['date']}):\n"
            f"Issue: {last['issue']}\n"
            f"Solution: {last['solution']}")

def generate_prompt(question: str, top_k: int = 5) -> str:
    resp  = openai_api.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    q_emb = resp.data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    hits = results["metadatas"][0]

    prompt = (
        "You are a CryoFERM AI assistant. Use these service records to answer.\n"
        "ONLY answer based on these records. Do NOT invent details.\n"
        "If no record answers the question, say so and ask permission to search externally.\n\n"
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
    incoming = request.values.get("Body", "").strip()
    if not incoming:
        return "OK"

    # 1) Feedback?
    m = feedback_re.match(incoming)
    if m:
        report_id, new_solution = m.group(1), m.group(2)
        reply = handle_feedback(report_id, new_solution)
    else:
        ensure_index()
        # 2) Last-job?
        direct = handle_last_job_query(incoming)
        if direct:
            reply = direct
        else:
            # 3) Retrieval + LLM
            prompt = generate_prompt(incoming)
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
