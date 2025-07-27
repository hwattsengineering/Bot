# … everything above stays the same …

# per‑user state now holds two flags instead of one
user_state = {}  # same dict, keyed by phone

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    sender = request.values.get("From")
    body   = request.values.get("Body","").strip().lower()

    # init
    state = user_state.setdefault(sender, {
        "awaiting_confirm": False,
        "awaiting_answer":  False,
        "last_query":       ""
    })
    resp = MessagingResponse()

    # 1) If we're waiting *for* the actual answer to a confirmed learn:
    if state["awaiting_answer"]:
        save_learned(state["last_query"], body)
        resp.message("✅ Got it — I’ve learned that.")
        state.update({
            "awaiting_confirm": False,
            "awaiting_answer":  False,
            "last_query":       ""
        })
        return str(resp)

    # 2) If we're waiting for your yes/no confirmation:
    if state["awaiting_confirm"]:
        if body in ("yes","y"):
            # move into answer‑collection mode
            state["awaiting_confirm"] = False
            state["awaiting_answer"]  = True
            resp.message("📥 Okay, please tell me how I should answer that.")
        else:
            # user said no or something else
            state.update({
                "awaiting_confirm": False,
                "awaiting_answer":  False,
                "last_query":       ""
            })
            resp.message("👍 No problem—let me know if you need anything else.")
        return str(resp)

    # 3) Normal flow: check learned, check reports, then offer to learn
    # (same as before)
    learned_ans = find_learned(body)
    if learned_ans:
        resp.message(learned_ans)
        return str(resp)

    hits = find_reports(body)
    if hits:
        resp.message("\n\n".join(hits[:3]))
        return str(resp)

    # nothing found → ask to learn
    resp.message(
        "I’m sorry, I don’t have that in the service records. "
        "Would you like me to learn how to answer that? (yes/no)"
    )
    state["awaiting_confirm"] = True
    state["last_query"]       = body
    return str(resp)
