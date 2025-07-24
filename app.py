from flask import Flask, request
import openai
import os
import os
import glob
import pandas as pd

# ————————————————
# Load all CSVs from data/ as your knowledge base
# ————————————————
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
csv_paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))

# Read and concatenate all inspection CSVs
dfs = []
for path in csv_paths:
    dfs.append(pd.read_csv(path))
all_inspections_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Normalize column names to match your bot’s fields
all_inspections_df.rename(columns={
    "report_id":        "id",
    "equipment_id":     "equipment",
    "fault_description":"issue",
    "corrective_action":"fix",
    "inspection_date":  "date"
}, inplace=True)

# Ensure all expected columns exist
for col in ("id","equipment","issue","fix","date"):
    if col not in all_inspections_df.columns:
        all_inspections_df[col] = ""

# Build the REPORTS list for prompting
REPORTS = all_inspections_df[["id","equipment","issue","fix","date"]] \
              .fillna("") \
              .astype(str) \
              .to_dict(orient="records")

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

import pandas as pd
import os

# Path to your extracted CSV file
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "inspections.csv")

def load_inspections():
    """Load inspection records from CSV into a list of dicts."""
    df = pd.read_csv(DATA_PATH)
    # Convert each row to a dict; rename or filter columns as needed
    records = df.to_dict(orient="records")
    # Example: unify keys with existing REPORTS format
    inspections = []
    for r in records:
        inspections.append({
            "id":        str(r.get("report_id", "")),
            "equipment": r.get("equipment_id", r.get("Equipment", "")),
            "issue":     r.get("fault_description", r.get("Issue", "")),
            "fix":       r.get("corrective_action", r.get("Fix", "")),
            "date":      r.get("inspection_date", r.get("Date", "")),
        })
    return inspections

# Load the CSV data once at startup
INSPECTIONS = load_inspections()





def generate_prompt(user_question):
    prompt = "You are a CryoFERM AI assistant. Help technicians troubleshoot faults using past service reports.\n\n"
    for report in REPORTS:
        prompt += f"- Report {report['id']} | Equipment: {report['equipment']} | Date: {report['date']}\n  Issue: {report['issue']}\n  Fix: {report['fix']}\n"
    prompt += f"\nTechnician's question: {user_question}\nAnswer:"
    return prompt

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    incoming_msg = request.values.get("Body", "").strip()
    if not incoming_msg:
        return "OK"

    prompt = generate_prompt(incoming_msg)

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Error: {str(e)}"

    from twilio.twiml.messaging_response import MessagingResponse
    twiml = MessagingResponse()
    twiml.message(reply)
    return str(twiml)

if __name__ == "__main__":
    app.run(debug=True)
