# server.py
# Backend for the HTML/CSS/JS frontend (uses Flask).

import os
from flask import Flask, request, jsonify, send_from_directory

KEY_NAMES = ["QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"]

 
missing = [n for n in KEY_NAMES if not os.getenv(n)]
if missing:
    print("These keys are not set:", ", ".join(missing))
    print('In PowerShell set them like this:  $env:GEMINI_API_KEY="your_key"')
    raise SystemExit(1)

import config
from embedding_store import load_embed_model, get_qdrant_client
from generation import get_gemini_client, NOT_FOUND
from rag_pipeline import answer_question

app = Flask(__name__, static_folder="frontend", static_url_path="")

 
model = None
client = None
gemini = None


@app.route("/")
def home():
    # show the website
    return send_from_directory("frontend", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please type a question."}), 400

    try:
        # run the whole pipeline: retrieve -> check context -> generate -> validate
        out = answer_question(client, model, gemini, question)
        answer, hits = out["answer"], out["hits"]
        print("FINAL ANSWER:", answer)
        print("FINAL REASON:", out["reason"])
        print("BEST SCORE:", out["top_score"])
        print("CHUNKS USED:", len(hits))
        # send the answer and the sources back to the page
        sources = []
        for p in hits:
            sources.append({
                "source": p.payload.get("source"),
                "page": p.payload.get("page_number"),
                "score": round(p.score, 2),
                "text": p.payload.get("text"),
            })

        return jsonify({
            "answer": answer,
            "found": answer.strip() != NOT_FOUND,
            "top_score": round(out["top_score"], 2),
            "reason": out["reason"],        # why it was refused (None if answered)
            "sources": sources,
        })

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500


if __name__ == "__main__":
    print("Loading models, please wait...")
    model = load_embed_model()
    client = get_qdrant_client()
    gemini = get_gemini_client()
    print("Ready. Open http://localhost:5000")
    app.run(port=5000, debug=False)
