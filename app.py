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
GH_TOKEN = os.getenv("GH_TOKEN", "").strip()
GH_REPO  = os.getenv("GH_REPO", "").strip()       # e.g. "yourorg/yourrepo"

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
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLUMNS]
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
    sha, text = github_get_file()
    if text is None:
        return "GitHub not configured; using local data only."
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LEARNED_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return "Pulled latest from GitHub."

def push_to_github(df: pd.DataFrame, commit_msg: str) -> str:
    if not (GH_TOKEN and GH_REPO):
        return "Saved locally (no GitHub config). Data may reset on redeploy."
    sha, _ = github_get_file()
    buf = io.StringIO()
    df[COLUMNS].to_csv(buf, index=False)
    ok, msg = github_put_file(buf.getvalue(), sha, commit_msg)
    return msg

def now_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def new_id() -> str:
    return "L" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

# ---------- Robust TEACH parser ----------
_TEACH_KEYS = {"id","date","equipment","issue","solution","technician","tags"}

def _extract_triple_backtick_block(text: str) -> str | None:
    """
    Allow solution in a fenced block:
      solution: ``` ...multiline... ```
    Returns the inner text or None.
    """
    m = re.search(r'solution\s*[:=]\s*```(.*?)```', text, flags=re.IGNORECASE|re.DOTALL)
    return m.group(1).strip() if m else None

def parse_teach(body: str) -> Dict[str, str]:
    """
    Accepts:
      - Pairs separated by semicolons OR new lines
      - key=value OR key: value
      - Straight or smart quotes
      - Long multi-line solution either in "quotes" or fenced as ```...```
    """
    text = body.strip()
    if text.upper().startswith("TEACH"):
        text = text[5:].strip()

    # Normalise smart quotes
    text = text.replace("“", '"').replace("”", '"')

    data: Dict[str, str] = {}

    # Prefer fenced block for solution, if present
    fenced = _extract_triple_backtick_block(text)
    if fenced:
        data["solution"] = fenced
        # Remove it from text so we don't double-parse
        text = re.sub(r'solution\s*[:=]\s*```(.*?)```', 'solution="<fenced>"', text, flags=re.IGNORECASE|re.DOTALL)

    # Split on ; or newline not inside double quotes
    parts = re.split(r'[\n;]+(?=(?:[^"]*"[^"]*")*[^"]*$)', text)

    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r'^([A-Za-z_]+)\s*[:=]\s*(.+)$', p)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()

        if key not in _TEACH_KEYS:
            continue

        # If value was placeholder for fenced content, keep earlier extracted solution
        if key == "solution" and val == "<fenced>" and "solution" in data:
            continue

        # Strip one layer of wrapping quotes
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]

        data[key] = val

    return data

# ---------- Search helpers ----------
def format_hits(rows: List[Dict[str, str]]) -> str:
    bullets = []
    for r in rows:
        bullets.append(
            f"- ID {r['id']} | {r['date']} | {r['equipment']} | Tech: {r['technician']}\n"
            f"  Issue: {r['issue']}\n"
            f"  Solution: {r['solution']}"
        )
    return "\n".join(bullets)

_tfidf: TfidfVectorizer | None = None
_tfidf_matrix = None
_tfidf_index_rows: List[int] | None = None

def _build_tfidf(df: pd.DataFrame):
    global _tfidf, _tfidf_matrix, _tfidf_index_rows
    if df.empty:
        _tfidf = _tfidf_matrix = _tfidf_index_rows = None
        return
    texts = (
        df['equipment'].fillna('') + ' ' +
        df['issue'].fillna('') + ' ' +
        df['solution'].fillna('') + ' ' +
        df['technician'].fillna('') + ' ' +
        df['tags'].fillna('')
    ).astype(str).tolist()
    _tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,3), min_df=1)
    _tfidf_matrix = _tfidf.fit_transform(texts)
    _tfidf_index_rows = df.index.tolist()

def semantic_search(df: pd.DataFrame, query: str, k: int = 6) -> pd.DataFrame:
    global _tfidf, _tfidf_matrix, _tfidf_index_rows
    if _tfidf is None or _tfidf_matrix is None or not _tfidf_index_rows:
        _build_tfidf(df)
        if _tfidf is None:
            return pd.DataFrame(columns=df.columns)
    q_vec = _tfidf.transform([query])
    sims = linear_kernel(q_vec, _tfidf_matrix).ravel()
    if sims.max(initial=0) <= 0:
        return pd.DataFrame(columns=df.columns)
    top_idx = sims.argsort()[::-1][:k]
    rows = [_tfidf_index_rows[i] for i in top_idx]
    out = df.loc[rows].copy()
    out["score"] = sims[top_idx]
    return out.sort_values("score", ascending=False)

def fuzzy_search(df: pd.DataFrame, question: str, limit: int = 6, threshold: int = 45) -> pd.DataFrame:
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
    hits = hay[hay["score"] >= threshold].copy().sort_values("score", ascending=False)
    return hits.drop(columns=["search_text"]).head(limit)

# ---------- Optional: nicer phrasing via OpenAI ----------
def compose_answer(question: str, rows: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        if not rows:
            return ("I don't have any records for that yet.\n\n"
                    "Teach me with:\n"
                    "TEACH issue=...; solution=\"...multi-line...\"; technician=...; tags=...\n"
                    "You can also use new lines and key: value.")
        return "Based on what I've learned:\n" + format_hits(rows)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        context = format_hits(rows) if rows else "(no matching rows)"
        system = ("You are a strict retrieval bot. Answer ONLY with information provided in the records. "
                  "If nothing matches, say you have no record and ask the user to TEACH.")
        user = f"Question: {question}\n\nRecords:\n{context}"
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Based on what I've learned:\n" + format_hits(rows)

# ---------- Flask ----------
app = Flask(__name__)

@app.get("/")
def health():
    return "OK", 200

@app.post("/sync")
def sync():
    note = sync_from_github()
    df = load_df()
    _build_tfidf(df)
    return {"status": "ok", "message": note, "rows": len(df)}, 200

@app.post("/whatsapp")
def whatsapp():
    body = (request.values.get("Body") or "").strip()
    from_num = request.values.get("From", "<unknown>")
    print(f"📩 From {from_num} :: {body[:120]}{'...' if len(body)>120 else ''}")

    tw = MessagingResponse()
    upper = body.upper()

    # ----- TEACH -----
    if upper.startswith("TEACH"):
        data = parse_teach(body)

        # If nothing parsed, tell the user how to send it
        if not any(data.get(k) for k in _TEACH_KEYS):
            tw.message(
                "I couldn't parse any TEACH fields.\n"
                "Use key=value (or key: value). Separate by ';' or new lines.\n"
                'Example:\nTEACH issue=O2 clean; solution="Use Blue Gold"; technician=Hamish; tags=oxygen,cleaning'
            )
            return str(tw)

        if not data.get("solution") and not data.get("issue"):
            tw.message('Please include at least issue=... or solution=... (wrap long text in quotes or ```fenced``` block).')
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

        # upsert by id
        df = df[df["id"] != row["id"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_df_local(df)
        gh_msg = push_to_github(df, f"Teach {row['id']} from WhatsApp")
        _build_tfidf(df)

        # Echo back what was learned (preview)
        preview = "; ".join(
            f"{k}={ (row[k][:60] + '...') if len(row[k])>60 else row[k] }"
            for k in ("equipment","issue","solution","technician","tags") if row[k]
        )
        tw.message(f"✅ Learned {row['id']} on {row['date']}\n{preview}\n{gh_msg}")
        return str(tw)

    # ----- SYNC / LIST / DELETE -----
    if upper in ("SYNC","RELOAD"):
        note = sync_from_github()
        df = load_df()
        _build_tfidf(df)
        tw.message(f"🔄 Reloaded {len(df)} records. {note}")
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
            lines.append(f"{r['id']} | {r['date']} | {r['equipment']} | {r['issue']} -> {r['solution'][:70]}{'...' if len(r['solution'])>70 else ''}")
        tw.message("Recent:\n" + "\n".join(lines))
        return str(tw)

    if upper.startswith("DELETE"):
        m = re.search(r"id\s*[:=]\s*([\w\-]+)", body, flags=re.IGNORECASE)
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
        _build_tfidf(df)
        tw.message(f"🗑️ Deleted {rid}. {before-after} row(s) removed.\n{gh_msg}")
        return str(tw)

    if upper in ("HELP","?"):
        tw.message(
            "Commands:\n"
            "TEACH (semicolon OR newline separated; = or : accepted):\n"
            'TEACH issue=O2 clean; solution="Use Blue Gold"; technician=Hamish; tags=oxygen,cleaning\n'
            "You can also send:\n"
            "TEACH\nissue: valve leaks\nsolution: ```\n1) Check PRV...\n2) Reseat...\n```\n\n"
            "LIST 5 | DELETE id=L2025... | SYNC\n"
            "Ask free-form questions too; I answer from what I've learned."
        )
        return str(tw)

    # ----- Free-form Q&A -----
    df = load_df()
    hits = semantic_search(df, body, k=6)
    if hits.empty:
        hits = fuzzy_search(df, body, limit=6, threshold=45)

    if not hits.empty:
        rows = hits.drop(columns=["score"], errors="ignore").to_dict(orient="records")
        answer = compose_answer(body, rows)
        tw.message(answer)
        return str(tw)

    # No match → auto-learn as unsolved
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
    _build_tfidf(df)

    tw.message(
        f"✅ Learned new issue: \"{body}\"\n"
        f"ID: {new_entry_id}\n{gh_msg}\n\n"
        "When you have the fix, reply with:\n"
        f"TEACH id={new_entry_id}; solution=\"...\""
    )
    return str(tw)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
