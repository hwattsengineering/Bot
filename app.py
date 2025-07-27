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
# 1) Load & normalize all CSVs
# ——————————————————————————————
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CSV_PATHS = glob.glob(os.path.join(DATA_DIR, "*.csv"))

def load_all_data():
    dfs = []
    for path in CSV_PATHS:
        try:
            df = pd.read_csv(path, dtype=str)
            if "id" not in df.columns:
                print(f"Skipping {os.path.basename(path)} (no 'id')")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"Warning reading {os.path.basename(path)}: {e}")
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    want = [
        "id", "date", "equipment_code", "equipment_name",
        "issue", "solution", "technician", "tags"
    ]
    combined = combined.reindex(columns=want).fillna("").astype(str)
    combined = combined[combined["id"].str.strip() != ""]
    combined = combined[combined["date"].str.strip() != ""]
    print(f"Loaded {len(combined)} records")
    return combined

all_df = load_all_data()

# ——————————————————————————————
# 2) ChromaDB & OpenAI setup
# ——————————————————————————————
client     = chromadb.Client()
collection = client.get_or_create_collection("service_records")
openai_api = OpenAI()  # requires OPENAI_API_KEY in env

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
        print("🔍 Indexing records…")
        for _, row in all_df.iterrows():
            text = f"Issue: {row['issue']}  Solution: {row['solution']}  Tags: {row['tags']}"
            emb = openai_api.embeddings.create(
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
        print("✅ Indexing complete")

# ——————————————————————————————
# 4) Feedback loop
# ——————————————————————————————
feedback_re = re.compile(r"^CORRECT\s+(\w+)\s*:\s*(.+)$", re.IGNORECASE)

def handle_feedback(report_id: str, new_solution: str) -> str:
    print(f"🔧 Feedback for {report_id}: {new_solution}")
    found = False
    for path in CSV_PATHS:
        df = pd.read_csv(path, dtype=str)
        if report_id in df.get("id", []):
            df.loc[df["id"] == report_id, "solution"] = new_solution
            df.to_csv(path, index=False)
            found = True
            print(f" • Updated {os.path.basename(path)}")
    if not found:
        return f"❌ Report {report_id} not found."

    # reload data and re-index
    global all_df, _indexed_flag
    all_df = load_all_data()
    _indexed_flag = False
    try:
        collection.delete(ids=[report_id])
    except Exception:
        pass
    ensure_index()
    return f"✅ Report {report_id} updated."

# ——————————————————————————————
# 5) Flask & Twilio webhook
# ——————————————————————————————
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return "OK", 200

def handle_last_job(question: str):
    m = re.search(r"what did (\w+) do on (?:his|her|their) last job", question, re.IGNORECASE)
    if not m:
        return None
    name = m.group(1)
    df = all_df[all_df["technician"].str.contains(name, case=False)]
    if df.empty:
        return f"Sorry, no jobs for {name}."
    df = df.assign(_dt=pd.to_datetime(df["date"], errors="coerce"))
    last = df.sort_values("_dt", ascending=False).iloc[0]
    return (
        f"{name}'s last job (Report {last['id']} on {last['date']}):\n"
        f"Issue: {last['issue']}\n"
        f"Solution: {last['solution']}"
    )

def generate_prompt(question: str, k: int = 5) -> str:
    emb = openai_api.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding
    results = collection.query(query_embeddings=[emb], n_results=k)
    prompt = (
        "You are a CryoFERM AI assistant. Use ONLY these records:\n"
        "Do NOT invent any details.\n\n"
    )
    for meta in results["metadatas"][0]:
        prompt += (
            f"- ID {meta['id']} | {meta['equipment_code']} {meta['equipment_name']} |"
            f" Date {meta['date']} | Tech {meta['technician']}\n"
            f"  Issue: {meta['issue']}\n"
            f"  Solution: {meta['solution']}\n\n"
        )
    prompt += f"Question: {question}\nAnswer:"
    print("📝 Prompt:", prompt.replace("\n", " | "))
    return prompt

@app.route("/whatsapp", methods=["GET", "POST"])
def whatsapp_bot():
    if request.method == "GET":
        # Twilio might GET to verify
        return "OK", 200

    incoming = request.values.get("Body", "").strip()
    print("📩 Received:", repr(incoming))

    # default response
    reply = "Sorry, something went wrong. Please try again."

    # 1) Feedback?
    m = feedback_re.match(incoming)
    if m:
        reply = handle_feedback(m.group(1), m.group(2))
    else:
        # 2) Ensure index for retrieval
        ensure_index()
        # 3) Last-job?
        last = handle_last_job(incoming)
        if last:
            reply = last
        else:
            # 4) Retrieval + LLM
            prompt = generate_prompt(incoming)
            resp   = openai_api.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = resp.choices[0].message.content.strip()
            print("💬 Answer:", answer)
            reply = answer or reply

    print("➡️ Replying:", reply)
    tw = MessagingResponse()
    tw.message(reply)
    return str(tw)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
