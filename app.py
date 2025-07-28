from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route("/whatsapp", methods=["GET","POST"])
def whatsapp_bot():
    if request.method == "GET":
        return "OK", 200

    body = request.values.get("Body","<none>")
    from_num = request.values.get("From","<none>")
    print(f"📩 GOT POST /whatsapp Body={body} From={from_num}")

    resp = MessagingResponse()
    resp.message(f"ECHO: you said “{body}” from {from_num}")
    return str(resp)

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
