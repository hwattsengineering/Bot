import os
import threading
import pandas as pd
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import openai

# ————————— CONFIG —————————
openai.api_key = os.getenv("OPENAI_API_KEY")

HERE    = os.path.dirname(__file__)
DATA    = os.path.join(HERE, "data")
INS_CSV = os.path.join(DATA, "inspections.csv")
LEA_CSV = os.path.join(DATA, "learned.csv")

# ————— FLASK INIT —————
app = Flask(__name__)

# ————— LOAD HISTORICAL RECORDS —————
# Expect columns: report_id,equipment_id,fault_description,corrective_action,inspection_date
ins_df = pd.read_csv(INS_CSV, dtype=str).fillna("")
ins_df.rename(columns={
    "report_id":         "rid",
    "equipment_id":      "equipment",
    "fault_description": "issue",
    "corrective_action": "fix",
    "inspection_date":   "date",
}, inplace=True)

# Build a list of (trigger_text, answer_text)
records = []
for _, r in ins_df.iterrows():
    trigger = f"{r['equipment']} on {r['date']}".lower()
    answer  = f"Issue: {r['issue']} → Fix: {r['fix']}"
    records.append((trigger, answer))

# ————— LOAD OR CREATE LEARNED FILE —————
if not os.path.exists(LEA_CSV):
    pd.DataFrame(columns=["question","answer"]).to_csv(LEA_CSV, index=False)

learned_df = pd.read_csv(LEA_CSV, dtype=str).fillna("")
lock = threading.Lock()

def save_learned(q, a):
    global learned_df
    with lock:
        pd.DataFrame([{"question":q, "answer":a}]) \
          .to_csv(LEA_CSV, mode="a", header=False, index=False)
        learned_df = pd.read_csv(LEA_CSV, dtype=str).fillna("")

# ————— USER STATE —————
# track per-phone-number learning flow
user_state = {}

# ————— LOOKUP HELPERS —————
def lookup_learned(q):
    m = learned_df[learned_df.question.str.lower() == q.strip().lower()]
    return m.answer.iloc[0] if not m.empty else None

def lookup_record(q):
    ql = q.strip().lower()
    for trigger, answer in records:
        if trigger in ql:
            return answer
    return None

# ————— WHATSAPP ENDPOINT —————
@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    from_number = request.values.get("From")
    incoming   = request.values.get("Body", "").strip()
    state      = user_state.setdefault(from_number, {
        "awaiting_confirm": False,
        "awaiting_answer":  False,
        "last_query":       ""
    })

    resp = MessagingResponse()

    # 1) If we are collecting the actual answer:
    if state["awaiting_answer"]:
        save_learned(state["last_query"], incoming)
        resp.message("✅ Got it — I’ve learned that.")
        state.update(awaiting_confirm=False, awaiting_answer=False, last_query="")
        return str(resp)

    # 2) If we’re waiting for yes/no to confirm learning:
    if state["awaiting_confirm"]:
        if incoming.lower() in ("yes","y"):
            state.update(awaiting_confirm=False, awaiting_answer=True)
            resp.message("📥 Please tell me how I should answer that question.")
        else:
            state.update(awaiting_confirm=False, awaiting_answer=False, last_query="")
            resp.message("👍 OK, no problem.")
        return str(resp)

    # 3) Standard query flow:
    # 3a) learned?
    ans = lookup_learned(incoming)
    if ans:
        resp.message(ans)
        return str(resp)

    # 3b) historical?
    rec_ans = lookup_record(incoming)
    if rec_ans:
        resp.message(rec_ans)
        return str(resp)

    # 3c) nothing → offer to learn
    resp.message(
        "I’m sorry, I don’t have that in the service records. "
        "Would you like me to learn how to answer that? (yes/no)"
    )
    state.update(awaiting_confirm=True, last_query=incoming)
    return str(resp)

# ————— RUN —————
if __name__ == "__main__":
    # local debug
    app.run(host="0.0.0.0", port=5000, debug=True)
