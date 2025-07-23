from flask import Flask, request
import openai
import os

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

REPORTS = [
    {
        "id": "005157",
        "equipment": "ISO DHGU2348103",
        "issue": "Leaking flange",
        "fix": "Repaired flange, replaced industrial pump belt A96, tested",
        "date": "22 July 2025",
    },
    {
        "id": "005158",
        "equipment": "Tanker 606",
        "issue": "Supply coupling gasket leak, bleed valve seal issue",
        "fix": "Replaced gasket, tightened gland, replaced bleed valve seals",
        "date": "22 July 2025",
    },
    {
        "id": "004878",
        "equipment": "ISO DHGU2348206",
        "issue": "Gauge non-functional",
        "fix": "Manually opened liquid valve and filled vessel",
        "date": "15 May 2025",
    }
]

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
            model="gpt-4",
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
