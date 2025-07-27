# app.py
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse
import csv, os

# —————————————————————————————————————————————
# 1) DATA LOADING
# —————————————————————————————————————————————
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORT_CSV   = os.path.join(DATA_DIR, "inspections.csv")
LEARNED_CSV  = os.path.join(DATA_DIR, "Learned.csv")

def load_csv(path, fieldnames):
    """Load a CSV file into a list of dicts."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        return list(reader)

# define your service‑report fields here; adjust to your actual columns
REPORT_FIELDS = ["report_id","equipment_id","fault_description","corrective_action","inspection_date","author"]
reports  = load_csv(REPORT_CSV,  REPORT_FIELDS)

# learned entries: we’ll store two columns: “trigger” and “response”
LEARNED_FIELDS = ["trigger","response"]
learned = load_csv(LEARNED_CSV, LEARNED_FIELDS)


# —————————————————————————————————————————————
# 2) QUERY FUNCTIONS
# —————————————————————————————————————————————
def find_in_reports(query):
    """Look for records where query appears in equipment_id, fault_description,
       corrective_action or author. Return list of nicely formatted strings."""
    q = query.lower()
    hits = []
    for r in reports:
        if any(q in (r.get(f) or "").lower() for f in ("equipment_id","fault_description","corrective_action","author")):
            hits.append(
                f"Report {r['report_id']} by {r.get('author','?')} on {r['inspection_date']}:\n"
                f" • Equipment: {r['equipment_id']}\n"
                f" • Issue: {r['fault_description']}\n"
                f" • Fix: {r['corrective_action']}"
            )
    return hits

def find_in_learned(query):
    """Look for triggers in learned entries."""
    q = query.lower()
    for entry in learned:
        if q == (entry.get("trigger") or "").lower():
            return entry.get("response")
    return None

def append_learned(trigger, response):
    """Persist a learned response to CSV and in‑memory."""
    with open(LEARNED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([trigger, response])
    learned.append({"trigger": trigger, "response": response})


# —————————————————————————————————————————————
# 3) FLASK + TWILIO WEBHOOK
# —————————————————————————————————————————————
app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    body = request.values.get("Body", "").strip()
    if not body:
        return ("", 204)

    resp = MessagingResponse()

    # 1) Check learned first
    learned_resp = find_in_learned(body)
    if learned_resp:
        resp.message(learned_resp)
        return str(resp)

    # 2) Search service reports
    hits = find_in_reports(body)
    if hits:
        # join up to 3 matches
        reply = "\n\n".join(hits[:3])
        resp.message(reply)
        return str(resp)

    # 3) Zero matches: ask if we should learn
    #    We’ll treat “yes” answers separately below.
    if body.lower() in ("yes","y","sure","ok","please"):
        # in your flow, you’ll need to capture the *last* question
        # here we’ll cheat: assume the last question was stored
        # as `session['last_question']` in a real app.
        trigger = whatsapp_bot.last_question
        response_text = whatsapp_bot.last_proposed_response or "..."
        append_learned(trigger, response_text)
        resp.message("Got it! I've learned that.")
        return str(resp)

    # 4) No matches & not a “yes” answer: prompt learn
    whatsapp_bot.last_question = body
    whatsapp_bot.last_proposed_response = f"Okay, what should I answer when asked: “{body}”?"
    resp.message(
        "I’m sorry, I don’t have that in the service records. "
        "Would you like me to learn that? (yes/no)"
    )
    return str(resp)

if __name__ == "__main__":
    # ensure data folder + files exist
    os.makedirs(DATA_DIR, exist_ok=True)
    # touch Learned.csv if missing
    if not os.path.exists(LEARNED_CSV):
        open(LEARNED_CSV, "w", newline="", encoding="utf-8").close()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
