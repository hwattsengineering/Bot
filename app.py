import os
import io
import re
import csv
import json
import base64
import requests
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# Semantic search + fuzzy fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import fuzz

# ---------- Config ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LEARNED_PATH = os.path.join(DATA_DIR, "Learned.csv")

# Optional GitHub persistence
GH_TOKEN = os.getenv("GH_TOKEN", "").strip()      # Personal Access Token (repo scope)
GH_REPO  = os.getenv("GH_REPO", "").strip()       # e.g. "hwattsengineering/Bot"

# OpenAI is optional (used only to nicely word answers). If not set, we reply with concise facts.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------- Utilities ----------
COLUMNS = ["id", "date", "equipment", "issue", "solution", "technician", "tags"]

def _ensure_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LEARNED_PATH):
        with open(LEARNED_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)

def load_df() -> pd.DataFrame:
    _ensure_file()
    try:
        df = pd.read_csv(LEARNED_PATH, dtype=str).fillna("")
    except Exception:
        df = pd.DataFrame(columns=COLUMNS)
    # ensure all columns exist and are ordered
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLUMNS]
    # drop empty id rows
    df = df[df["id"].astype(str).str.strip() != ""]
    return df.reset_index(drop=True)

def save_df_local(df: pd.DataFrame):
    df[COLUMNS].to_csv(LEARNED_PATH, index=False)

def github_get_file():
    """Return (sha, text) or (None, None) if not found or not configured."""
    if not (GH_TOKEN and GH_REPO):
        return None, None
    url = f"https://api.github.com/repos/{GH_REPO}/contents/data/Learned.csv"
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        js = r.json()
        content = base64.b64decode(js["content"]).decode("utf-8", errors="ignore")
        return js.get("sha"), content
    return None, None

def github_put_file(new_text: str, sha: str | None, message: str):
    if not (GH_TOKEN and GH_REPO):
        return False, "GitHub not configured"
    url = f"https://api.github.com/repos/{GH_REPO}/contents/data/Learned.csv"
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"}
    payload = {
        "message": message,
        "content": base64.b64encode(new_text.encode("utf-8")).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, data=json.dumps(payload))
    if r.status_code in (200, 201):
        return True, "Saved to GitHub"
    return False, f"GitHub save failed: {r.status_code} {r.text[:120]}"

def sync_from_github() -> str:
    """Pull latest file from GitHub and overwrite local. No-op if not configured."""
    sha, text = github_get_file()
    if text is None:
        return "GitHub not configured; using local data only."
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LEARNED_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return "Pulled latest from GitHub."

def push_to_github(df: pd.DataFrame, commit_msg: str) -> str:
    """Push current df to GitHub, or say not configured."""
    if not (GH_TOKEN and GH_REPO):
        return "Saved locally (no GitHub config). Data may reset on redeploy."
    sha, _ = github_get_file()
    buf = io.StringIO()
    df[COLUMNS].to_csv(buf, index=False)
    ok, msg = github_put_file(buf.getvalue(), sha, commit_msg)
    return msg

def now_date() -> str:
    # UTC date string
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def new_id() -> str:
    return "L" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

def parse_teach(body: str) -> Dict[str, str]:
    """
    TEACH id=..., date=..., equipment=..., issue=..., solution=..., technician=..., tags=...
    Semicolons separate pairs. Values may be "quoted".
    """
    text = body.strip()
    if text.upper().startswith("TEACH"):
        text = text[5:].strip()

    # Split on semicolons NOT inside double quotes
    parts = re.split(r';(?=(?:[^"]*"[^"]*")*[^"]*$)', text)
    data: Dict[str, str] = {}
    for p in parts:
        p = p.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        # strip wrapping quotes
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        data[k] = v
    return data

def format_hits(rows: List[Dict[str, str]]) -> str:
    bullets = []
    for r in rows:
        bullets.append(
            f"- ID {r['id']} | {r['date']} | {r['equipment']} | Tech: {r['technician']}\n"
            f"  Issue: {r['issue']}\n"
            f"  Solution: {r['solution']}"
        )
    return "\n".join(bullets)

# ---------- Semantic Search (TF-IDF) ----------
_tfidf: TfidfVectorizer | None = None
_tfidf_matrix = None
_tfidf_index_rows: List[int] | None = None

def _build_tfidf(df: pd.DataFrame):
    """Build or rebuild the TF-IDF index from the dataframe."""
    global _tfidf, _tfidf_matrix, _tfidf_index_rows
    if df.empty:
        _tfidf = None
        _tfidf_matrix = None
        _tfidf_index_rows = None
        return
    texts = (
        df['equipment'].fillna('') + ' ' +
        df['issue'].fillna('') + ' ' +
        df['solution'].fillna('') + ' ' +
        df['technician'].fillna('') + ' ' +
        df['tags'].fillna('')
    ).astype(str).tolist()
    _tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),  # capture short phrases
        min_df=1
    )
    _tfidf_matrix = _tfidf.fit_transform(texts)
    _tfidf_index_rows = df.index.tolist()

def semantic_search(df: pd.DataFrame, query: str, k: int = 6) -> pd.DataFrame:
    """Return top-k rows by cosine similarity. Builds the index on-demand if needed."""
    global _tfidf, _tfidf_matrix, _tfidf_index_rows
    if _tfidf is None or _tfidf_matrix is None or not _tfidf_index_rows:
        _build_tfidf(df)
        if _tfidf is None:
            return pd.DataFrame(columns=df.columns)
    q_vec = _tfidf.transform([query])
    sims = linear_kernel(q_vec, _tfidf_matrix).ravel()  # cosine similarity
    if sims.max(initial=0) <= 0:
        return pd.DataFrame(columns=df.columns)
    top_idx = sims.argsort()[::-1][:k]
    rows = [_tfidf_index_rows[i] for i in top_idx]
    out = df.loc[rows].copy()
    out["score"] = sims[top_idx]
    return out.sort_values("score", ascending=False)

# ---------- Fuzzy fallback (RapidFuzz) ----------
def fuzzy_search(df: pd.DataFrame, question: str, limit: int = 6, threshold: int = 50) -> pd.DataFrame:
    """Fallback fuzzy search to handle typos/near-misses."""
    if df.empty:
        return df
    q = question.lower()

    hay = df.assign(
        search_text=(
            df['equipment'].str.lower() + ' ' +
            df['issue'].str.lower() + ' ' +
            df['solution'].str.lower() + ' ' +
            df['technician'].str.lower() + ' ' +
            df['tags'].str.lower()
        ).fillna("")
    )

    hay["score"] = hay["search_text"].apply(lambda t: max(
        fuzz.partial_ratio(q, t),
        fuzz.token_set_ratio(q, t),
        fuzz.QRatio(q, t)
    ))

    hits = hay[hay["score"] >= threshold].copy()
    hits = hits.sort_values("score", ascending=False)
    return hits.drop(columns=["search_text"]).head(limit)

# ---------- Optional: nicer phrasing via OpenAI ----------
def compose_answer(question: str, rows: List[Dict[str, str]]) -> str:
    """If OPENAI_API_KEY is set, ask it to phrase the answer using ONLY rows."""
    if not OPENAI_API_KEY:
        if not rows:
            return ("I don't have any records for that yet.\n\n"
                    "Teach me with:\n"
                    "TEACH equipment=..., issue=..., solution=..., technician=..., tags=...\n"
                    "Optional: id=..., date=YYYY-MM-DD")
        facts = format_hits(rows)
        return f"Based on what I've learned:\n{facts}"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        context = format_hits(rows) if rows else "(no matching rows)"
        system = (
            "You are a strict retrieval bot. Answer ONLY with information provided in the records. "
            "If nothing matches, say you have no record and ask the user to TEACH."
        )
        user = f"Question: {question}\n\nRecords:\n{context}"
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        facts = format_hits(rows)
        return f"Based on what I've learned:\n{facts}\n\n(Note: phrasing fallback due to: {e})"

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    body = (request.values.get("Body") or "").strip()
    from_num = request.values.get("From", "<unknown>")
    print(f"📩 From {from_num} :: {body}")

    tw = MessagingResponse()
    upper = body.upper()

    if upper.startswith("TEACH"):
        data = parse_teach(body)
        if not data.get("solution") and not data.get("issue"):
            tw.message("Please include at least issue=... or solution=... in your TEACH command.")
            return str(tw)

        df = load_df()
        row = {
            "id":         data.get("id", new_id()),
            "date":       data.get("date", now_date()),
            "equipment":  data.get("equipment", ""),
            "issue":      data.get("issue", ""),
            "solution":   data.get("solution", ""),
            "technician": data.get("technician", ""),
            "tags":       data.get("tags", ""),
        }
        # upsert by id (replace any existing row with same id)
        df = df[df["id"] != row["id"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_df_local(df)
        gh_msg = push_to_github(df, f"Teach {row['id']} from WhatsApp")

        # Rebuild TF-IDF index after update
        _build_tfidf(df)

        tw.message(f"✅ Learned {row['id']} | {row['equipment']} | {row['issue']}\n{gh_msg}")
        return str(tw)

    if upper in ("SYNC", "RELOAD"):
        note = sync_from_github()
        df = load_df()

        # Rebuild TF-IDF after sync
        _build_tfidf(df)

        tw.message(f"🔄 Reloaded records ({len(df)}). {note}")
        return str(tw)

    if upper.startswith("DELETE"):
        m = re.search(r"id\s*=\s*([\w\-]+)", body, flags=re.IGNORECASE)
        if not m:
            tw.message("Usage: DELETE id=YOUR_ID")
            return str(tw)
        rid = m.group(1)
        df = load_df()
        before = len(df)
        df = df[df["id"] != rid]
        after = len(df)
        save_df_local(df)
        gh_msg = push_to_github(df, f"Delete {rid} from WhatsApp")

        # Rebuild TF-IDF after deletion
        _build_tfidf(df)

        tw.message(f"🗑️ Deleted {rid}. {before - after} row(s) removed.\n{gh_msg}")
        return str(tw)

    if upper.startswith("LIST"):
        n = 5
        m = re.search(r"LIST\s+(\d+)", upper)
        if m:
            n = max(1, min(20, int(m.group(1))))
        df = load_df().tail(n)
        if df.empty:
            tw.message("No learned records yet. Use TEACH to add one.")
            return str(tw)
        lines = []
        for _, r in df.iterrows():
            lines.append(f"{r['id']} | {r['date']} | {r['equipment']} | {r['issue']} -> {r['solution']}")
        tw.message("Recent:\n" + "\n".join(lines))
        return str(tw)

    if upper in ("HELP", "?"):
        tw.message(
            "Commands:\n"
            "TEACH id=O2CLEAN; date=2025-08-08; equipment=\"O2 systems\"; "
            "issue=Oxygen cleaning; solution=Use Blue Gold; technician=Angus; tags=oxygen,cleaning\n"
            "LIST 5\nDELETE id=...\nSYNC (pull latest from GitHub)\n"
            "Ask free-form questions too. I only answer from what I've learned."
        )
        return str(tw)

    # Free-form Q&A → semantic first, fuzzy fallback
    df = load_df()
    hits = semantic_search(df, body, k=6)
    if hits.empty:
        hits = fuzzy_search(df, body, limit=6, threshold=50)

    if not hits.empty:
        rows = hits.drop(columns=["score"], errors="ignore").to_dict(orient="records")
        answer = compose_answer(body, rows)
        tw.message(answer)
        return str(tw)
    else:
        # No matches found. Learn as new, unsolved issue.
        new_entry_id = new_id()
        row = {
            "id":         new_entry_id,
            "date":       now_date(),
            "equipment":  "",
            "issue":      body,
            "solution":   "",
            "technician": "",
            "tags":       "unsolved",
        }

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_df_local(df)
        gh_msg = push_to_github(df, f"Auto-learn new issue {new_entry_id} from WhatsApp")

        # Rebuild TF-IDF after adding new row
        _build_tfidf(df)

        reply_msg = (
            f"✅ Learned new issue: \"{body}\"\n"
            f"ID: {new_entry_id}\n"
            f"{gh_msg}\n\n"
            "I don't have a solution yet. When you do, teach me with:\n"
            f"TEACH id={new_entry_id}; solution=..."
        )
        tw.message(reply_msg)
        return str(tw)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    # Local run: Flask dev server. On Render with gunicorn, this block is ignored.
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
