import os
import io
import glob
import re
import json
import time
import threading
import requests
import numpy as np
import pandas as pd
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

print("🚀 Starting CryoFERM bot — CSV+GitHub, no-Chroma")

# =========================
# Config
# =========================
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
CSV_GITHUB_URLS = [u.strip() for u in os.getenv("CSV_URLS", "").split(",") if u.strip()]
CHAT_MODEL     = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# =========================
# Canonical schema
# =========================
CANON_COLS = ["id","date","equipment","issue","solution","technician","tags"]

RENAME_MAP = {
    # try to normalize different sheets/headers into our canon
    "job_number": "id",
    "report_id": "id",
    "job id": "id",
    "jobid": "id",

    "equipment_id": "equipment",
    "equipment_code": "equipment",
    "equipment_name": "equipment",

    "fault_description": "issue",
    "issue_description": "issue",
    "problem": "issue",

    "solution_description": "solution",
    "corrective_action": "solution",
    "description of works carried out": "solution",
    "fix_solution": "solution",
    "fix solution": "solution",

    "author": "technician",
    "tech": "technician",
    "service_tech": "technician",
}

# =========================
# Data loading
# =========================
def _read_local_csvs() -> list[pd.DataFrame]:
    paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p, dtype=str))
        except Exception as e:
            print(f"⚠️ Could not read {p}: {e}")
    return dfs

def _read_remote_csvs(urls: list[str]) -> list[pd.DataFrame]:
    dfs = []
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            dfs.append(pd.read_csv(io.StringIO(r.text), dtype=str))
        except Exception as e:
            print(f"⚠️ Could not fetch {url}: {e}")
    return dfs

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c for c in df.columns}
    # rename by map if a clean hit
    for c in list(df.columns):
        c_norm = c.strip().lower()
        if c_norm in RENAME_MAP:
            cols[c] = RENAME_MAP[c_norm]
    df = df.rename(columns=cols)

    # reindex to canon, fill empties
    for col in CANON_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[CANON_COLS].fillna("").astype(str)

    # drop rows with no id or date (too ambiguous)
    df = df[df["id"].str.strip() != ""]
    df = df[df["date"].str.strip() != ""]
    return df

def load_all_records() -> pd.DataFrame:
    dfs = []
    if CSV_GITHUB_URLS:
        dfs.extend(_read_remote_csvs(CSV_GITHUB_URLS))
    else:
        dfs.extend(_read_local_csvs())
    if not dfs:
        print("⚠️ No CSVs found. Bot will have no knowledge.")
        return pd.DataFrame(columns=CANON_COLS)
    merged = pd.concat(dfs, ignore_index=True)
    normed = _normalize_df(merged)
    print(f"📦 Loaded {len(normed)} rows from {'remote' if CSV_GITHUB_URLS else 'local'} CSVs.")
    return normed

# =========================
# Embedding index (in-memory)
# =========================
_index_lock = threading.Lock()
_index_ready = False
_records_df: pd.DataFrame | None = None
_doc_texts: list[str] = []
_doc_embs: np.ndarray | None = None  # shape (N, D), unit-normalized

def _row_to_text(row: pd.Series) -> str:
    return (
        f"ID: {row['id']}\n"
        f"Date: {row['date']}\n"
        f"Technician: {row['technician']}\n"
        f"Equipment: {row['equipment']}\n"
        f"Issue: {row['issue']}\n"
        f"Solution: {row['solution']}\n"
        f"Tags: {row['tags']}"
    )

def _embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    # batch embed to reduce API round-trips
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs / norms

def rebuild_index():
    global _index_ready, _records_df, _doc_texts, _doc_embs
    with _index_lock:
        df = load_all_records()
        _records_df = df.reset_index(drop=True)
        _doc_texts = [_row_to_text(r) for _, r in _records_df.iterrows()]
        _doc_embs = _embed_texts(_doc_texts)
        _index_ready = True
        print(f"🔍 Indexed {_doc_embs.shape[0]} records")

def ensure_index():
    if not _index_ready:
        rebuild_index()

def search(query: str, k: int = 5) -> list[tuple[float, pd.Series]]:
    ensure_index()
    if _doc_embs is None or _doc_embs.shape[0] == 0:
        return []
    q_emb = _embed_texts([query])[0]
    sims = (_doc_embs @ q_emb).astype(float)  # cosine similarity
    idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), _records_df.iloc[i]) for i in idx]

# =========================
# QA prompt
# =========================
def build_prompt(question: str, hits: list[tuple[float, pd.Series]]) -> str:
    preface = (
        "You are a CryoFERM AI assistant for technicians.\n"
        "Answer ONLY using the records shown. If the records don't contain the answer, say you don't have it.\n"
        "Be concise and practical.\n\n"
        "Records:\n"
    )
    body = ""
    for score, row in hits:
        body += (
            f"- ID {row['id']} | Date {row['date']} | Tech {row['technician']} | Equip {row['equipment']}\n"
            f"  Issue: {row['issue']}\n"
            f"  Solution: {row['solution']}\n\n"
        )
    tail = f"Question: {question}\nAnswer:"
    return preface + body + tail

# =========================
# Special intent: last job for a tech
# =========================
_last_job_re = re.compile(
    r"what did\s+([a-zA-Z][a-zA-Z\s\.-]+?)\s+do\s+on\s+(?:his|her|their|the)\s+last\s+job\??",
    re.IGNORECASE
)

def answer_last_job(question: str) -> str | None:
    m = _last_job_re.search(question)
    if not m or _records_df is None or _records_df.empty:
        return None
    name = m.group(1).strip()
    df = _records_df[_records_df["technician"].str.contains(name, case=False, na=False)].copy()
    if df.empty:
        return f"Sorry, I can’t find any jobs for {name}."
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    last = df.sort_values("_dt", ascending=False).iloc[0]
    return (
        f"{name}'s last job — Report {last['id']} on {last['date']} (Equip {last['equipment']}):\n"
        f"Issue: {last['issue']}\n"
        f"Solution: {last['solution']}"
    )

# =========================
# Flask app
# =========================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/sync", methods=["POST","GET"])
def sync_now():
    rebuild_index()
    return f"Synced. Records: {_records_df.shape[0] if _records_df is not None else 0}\n", 200

@app.route("/whatsapp", methods=["GET","POST"])
def whatsapp_bot():
    if request.method == "GET":
        return "OK", 200

    incoming = (request.values.get("Body") or "").strip()
    resp = MessagingResponse()

    if not incoming:
        resp.message("Please send a question.")
        return str(resp)

    # Admin command to refresh from GitHub without redeploy
    if incoming.upper() == "SYNC":
        rebuild_index()
        count = 0 if _records_df is None else _records_df.shape[0]
        resp.message(f"✅ Reloaded records. Count: {count}")
        return str(resp)

    # “last job” intent
    lj = answer_last_job(incoming)
    if lj:
        resp.message(lj)
        return str(resp)

    # Otherwise: retrieval-augmented answer from CSV records only
    hits = search(incoming, k=5)
    if not hits:
        resp.message("I don't have records for that yet. Push an updated CSV to GitHub, then send SYNC.")
        return str(resp)

    prompt = build_prompt(incoming, hits)
    try:
        out = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        answer = out.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"

    # If the model still says it can't answer, guide user to update CSVs
    if re.search(r"\b(don’t|don't)\s+have\b|\bno record\b|\bnot (?:in|found)\b", answer, re.IGNORECASE):
        answer += "\n\nYou can update the CSVs in GitHub and send SYNC to reload."

    resp.message(answer or "I couldn't produce an answer.")
    return str(resp)

# Build the initial index at startup
try:
    rebuild_index()
except Exception as e:
    print("⚠️ Initial index build failed:", e)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
