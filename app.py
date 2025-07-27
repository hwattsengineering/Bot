# app.py

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os, csv

# —————————————————————————————————————————————
# 1) CONFIG + DATA PATHS
# —————————————————————————————————————————————
app = Flask(__name__)
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
REPORT_CSV  = os.path.join(DATA_DIR, "inspections.csv")
LEARNED_CSV = os.path.join(DATA_DIR, "Learned.csv")

# Ensure data folder and learned file exist
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(LEARNED_CSV):
    open(LEARNED_CSV, "w", newline="", encoding="utf-8").close()

# —————————————————————————————————————————————
# 2) LOAD CSVS INTO MEMORY
# —————————————————————————————————————————————
def load_csv(path, fieldnames):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        return list(reader)

# adjust these to match your inspections.csv header row
REPORT_FIELDS = [
    "report_id","equipment_id","fault_description",
    "corrective_action","inspection_date","author"
]
reports  = load_csv(REPORT_CSV, REPORT_FIELDS)

LEARNED_FIELDS = ["trigger","response"]
learned = load_csv(LEARNED_CSV, LEARNED_FIELDS)

# —————————————————————————————————————————————
# 3) HELPER FUNCTIONS
# —————————————————————————————————————————————
def find_reports(query):
    q = query.lower()
    hits = []
    for r in reports:
        # look in equipment, issue, fix, author
        if any(q in (r.get(f) or "").lower() for f in (
            "equipment_id","fault_description","corrective_action","author"
        )):
            hits.append(
                f"Report {r['report_id']} by {r.get('author','?')} on {r['inspection_date']}:\n"
                f" • Equipment: {r['equipment_id']}\n"
                f" • Issue: {r['fault_description']}\n"
                f" • Fix: {r['corrective_action']}"
            )
    return hits

def find_learned(query):
    q = query.lower()
    for e in learned:
        if q == (e["trigger"] or "").lower():
            return e["response"]
    return None

def save_learned(trigger, response):
    with open(LEARNED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([trigger, response])
    learned.append({"trigger":trigger, "response":response})

# —————————————————————————————————————————————
# 4) PER‑USER STATE
# —————————————————————————————————————————————
# In a real multi‑user app you'd key by phone number; here we store:
user_state = {
    # e.g. "+61434117566": {"awaiting_learn": False, "last_query": ""}
}

# —————————————————————————————————————————————
# 5) FLASK ENDPOINT
# —————————————————————————————————————————————
@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    sender = request.values.get("From")    # e.g. "whatsapp:+61434117566"
    body   = request.values.get("Body","").strip()
    if not sender or not body:
        return ("",204)

    # init state for this user if needed
    state = user_state.setdefault(sender, {
        "awaiting_learn": False,
        "last_query":     ""
    })

    resp = MessagingResponse()

    # 1) If we were waiting *for* a learned answer, store this message as the response
    if state["awaiting_learn"]:
        trigger = state["last_query"]
        answer  = body
        save_learned(trigger, answer)
        resp.message("✅ Got it—I've learned that.")
        # reset state
        state["awaiting_learn"] = False
        state["last_query"]     = ""
        return str(resp)

    # 2) Check learned entries first
    learned_ans = find_learned(body)
    if learned_ans:
        resp.message(learned_ans)
        return str(resp)

    # 3) Search reports
    hits = find_reports(body)
    if hits:
        # send up to 3 matches
        resp.message("\n\n".join(hits[:3]))
        return str(resp)

    # 4) No matches: ask if we should learn
    resp.message(
        "I’m sorry, I don’t have that in the service records. "
        "Would you like me to learn how to answer that? (yes/no)"
    )
    # set state so next message is captured
    state["awaiting_learn"] = True
    state["last_query"]     = body
    return str(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
