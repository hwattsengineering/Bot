import os
import glob
import threading
import re
import pandas as pd
import chromadb
from openai import OpenAI
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# ——— 1) CSV loading (keep for later real logic) ——————————————
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CSV_PATHS = glob.glob(os.path.join(DATA_DIR, "*.csv"))

def load_all_data():
    dfs = []
    for path in CSV_PATHS:
        try:
            df = pd.read_csv(path, dtype=str)
            if "id" not in df.columns:
                continue
            dfs.append(df)
        except:
            continue
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined

all_df = load_all_data()

# ——— 2) ChromaDB & OpenAI setup (keep for later real logic) ——————————————
client     = chromadb.Client()
collection = client.get_or_create_collection("service_records")
openai_api = OpenAI()  # requires OPENAI_API_KEY

# ——— 3) Flask & Twilio webhook —————————————————————
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return "OK", 200

@app.route("/whatsapp", methods=["GET", "POST"])
def whatsapp_bot():
    # Twilio sometimes GETs to verify
    if request.method == "GET":
        return "OK", 200

    # Grab incoming WhatsApp message
    incoming = request.values.get("Body", "").strip()
    print("📩 DEBUG Received incoming:", repr(incoming))

    #— IMMEDIATE DEBUG REPLY ——
    # This ensures you see a reply in WhatsApp
    tw = MessagingResponse()
    tw.message("DEBUG: Handler invoked and received your message!")
    return str(tw)

    # ——————————————————————————————————————
    # Once you see the DEBUG reply working, you can
    # comment out the above block and re‑insert your
    # real logic below (feedback, indexing, retrieval, etc.)
    # ——————————————————————————————————————

    # Example real logic stub:
    # reply = "Sorry, I don't have an answer yet."
    # tw = MessagingResponse()
    # tw.message(reply)
    # return str(tw)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
