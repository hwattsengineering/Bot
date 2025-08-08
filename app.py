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

# ---------- Config ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LEARNED_PATH = os.path.join(DATA_DIR, "Learned.csv")

# Optional GitHub persistence
GH_TOKEN = os.getenv("GH_TOKEN", "").strip()      # Personal Access Token (repo scope)
GH_REPO  = os.getenv("GH_REPO", "").strip()        # e.g. "hwattsengineering/Bot"

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

    # make sure all expected columns exist
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    # keep just the columns we care about, in order
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
        return js["sha"], content
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
    Semicolons separate pairs. Keys case-insensitive.
    """
    # remove leading 'TEACH'
    text = body.strip()[5:].strip() if body.strip().upper().startswith("TEACH") else body
    parts = [p.strip() for p in text.split(";") if p.strip()]
    data: Dict[str, str] = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            data[k.strip().lower()] = v.strip()
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

# ----------[ MODIFIED FUNCTION ]----------
def simple_search(df: pd.DataFrame, question: str, limit: int = 6) -> pd.DataFrame:
    """
    Smarter, token-based keyword search. Splits the question into important words
    and scores rows based on how many words they contain.
    """
    # 1. Get important search terms from the question.
    # We ignore common "stop words" and words shorter than 2 characters.
    q_lower = question.lower()
    stop_words = {'is', 'a', 'an', 'the', 'what', 'why', 'how', 'when', 'who', 'for', 'on', 'in', 'it', 'to', 'of'}
    search_terms = {word for word in re.split(r'\W+', q_lower) if word and word not in stop_words and len(word) > 1}

    if not search_terms:
        return pd.DataFrame(columns=df.columns) # Return empty if question has no useful words

    # 2. Create a single, combined text field for each row to search within.
    searchable_df = df.copy()
    searchable_df['search_text'] = (
        searchable_df['id'].str.lower() + ' ' +
        searchable_df['equipment'].str.lower() + ' ' +
        searchable_df['issue'].str.lower() + ' ' +
        searchable_df['solution'].str.lower() + ' ' +
        searchable_df['technician'].str.lower() + ' ' +
        searchable_df['tags'].str.lower()
    )

    # 3. Score each row by counting how many search terms it contains.
    def count_matches(text_to_search):
        return sum(1 for term in search_terms if term in text_to_search)

    searchable_df['score'] = searchable_df['search_text'].apply(count_matches)

    # 4. Filter for rows that have at least one match.
    hits = searchable_df[searchable_df['score'] > 0].copy()

    # 5. Sort the best matches to the top (by score, then by date).
    if not hits.empty:
        if "date" in hits.columns:
            hits["_dt"] = pd.to_datetime(hits["date"], errors="coerce")
            hits = hits.sort_values(by=["score", "_dt"], ascending=[False, False])
            hits = hits.drop(columns=["_dt"], errors="ignore")
        else:
            hits = hits.sort_values(by="score", ascending=False)

    # 6. Return the top N hits, without the temporary 'score' and 'search_text' columns.
    return hits.drop(columns=['search_text', 'score'], errors='ignore').head(limit)


def compose_answer(question: str, rows: List[Dict[str, str]]) -> str
