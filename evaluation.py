 
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import config
from generation import build_context, call_gemini, NOT_FOUND


PROGRESS_FILE = "evaluation_progress2.jsonl"
RESULTS_CSV = "evaluation_results2.csv"
METRICS_JSON = "evaluation_metrics2.json"
DIAGNOSTICS_JSON = "evaluation_diagnostics2.json"


# ---------- Step 1: read the JSON ----------
def load_eval_file(path=None):
    path = path or config.EVAL_FILE

    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.basename(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    with open(path, encoding="utf-8") as f:
        items = json.load(f)

    for number, item in enumerate(items, start=1):
        if not str(item.get("question", "")).strip() or not str(item.get("answer", "")).strip():
            raise ValueError(f"Item {number} has an empty question or answer.")
        if "PUT THE CORRECT ANSWER" in item["answer"]:
            raise ValueError("Please replace the 'PUT THE CORRECT ANSWER...' text "
                             "in the JSON with the real answers first.")
        item["question"] = item["question"].strip()

    # same question with different answers is ambiguous for the system
    seen = {}
    for item in items:
        seen.setdefault(item["question"].lower(), set()).add(item["answer"])
    repeated = sum(1 for answers in seen.values() if len(answers) > 1)
    if repeated:
        print(f"WARNING: {repeated} questions appear more than once with different "
              f"answers. Add the scheme/document name to the question text.")

    return items


def is_refusal(text):
    """True if the text is the 'not found' message."""
    return NOT_FOUND.lower() in str(text).lower()


def item_key(question, answer):
    """A question is identified by its text AND its correct answer."""
    return (question, answer)


# ---------- Step 2: embed the correct answers and store them ----------
def build_answer_embeddings(model, items):
    """
    Embeds every correct answer once (same model as the chunks) and caches the
    vectors in a file.  Returns a dict:  answer text -> vector
    """
    stored = {}
    if os.path.exists(config.ANSWER_EMB_FILE):
        data = np.load(config.ANSWER_EMB_FILE, allow_pickle=False)
        stored = dict(zip(data["texts"].tolist(), data["vectors"]))

    new_texts = [it["answer"] for it in items if not is_refusal(it["answer"])]
    new_texts = [t for t in dict.fromkeys(new_texts) if t not in stored]

    if new_texts:
        print(f"Embedding {len(new_texts)} correct answers...")
        vectors = model.encode(new_texts, batch_size=16, show_progress_bar=True,
                               normalize_embeddings=True, convert_to_numpy=True)
        for text, vec in zip(new_texts, vectors):
            stored[text] = vec
        np.savez(config.ANSWER_EMB_FILE,
                 texts=np.array(list(stored.keys())),
                 vectors=np.array(list(stored.values())))
        print("Saved:", config.ANSWER_EMB_FILE)
    else:
        print("Correct answer embeddings loaded from file.")

    return stored


# ---------- Step 3: the two checks ----------
def chunk_vectors(model, hits):
    """
    Returns a (num_hits, dim) array of normalized chunk vectors.
    Uses the vectors returned by Qdrant when available, otherwise
    re-embeds the chunk text (so it works even without with_vectors=True).
    """
    vectors = [getattr(h, "vector", None) for h in hits]
    usable = all(v is not None and not isinstance(v, dict) for v in vectors)

    if usable:
        array = np.array(vectors, dtype=float)
    else:
        texts = [(h.payload or {}).get("text", "") for h in hits]
        array = np.array(model.encode(texts, normalize_embeddings=True,
                                      convert_to_numpy=True), dtype=float)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def best_chunk_match(answer_vec, hits, chunk_vecs):
    """Finds the retrieved chunk closest to the correct answer (cosine)."""
    if not hits:
        return 0.0, None
    sims = chunk_vecs @ np.asarray(answer_vec, dtype=float)
    best = int(np.argmax(sims))
    return float(sims[best]), hits[best]


def check_faithfulness(gemini, question, answer, hits):
    """Ask Gemini: is every fact in the answer supported by the context? YES/NO"""
    prompt = f"""You are checking an answer for hallucination.

Context:
{build_context(hits)}

Question: {question}
Answer: {answer}

Is every fact in the Answer supported by the Context?
Reply with only one word: YES or NO."""
    return call_gemini(gemini, prompt).upper().startswith("YES")


def source_name(hit):
    """File name only (no folders), for the CSV."""
    raw = str((hit.payload or {}).get("source", "")).replace("\\", "/")
    return os.path.basename(raw)


def evaluate_one(client, model, gemini, item, answer_vectors):
    """Runs one question through the pipeline and checks it against the correct answer."""
    from server import answer_question      # imported here so this file also works as a module

    question = item["question"]
    truth = item["answer"]

    start = time.time()
    out = answer_question(client, model, gemini, question, with_vectors=True)
    latency = time.time() - start

    generated, hits = out["answer"], out["hits"]

    should_refuse = is_refusal(truth)       # correct answer is "not found"
    did_refuse = is_refusal(generated)      # system said "not found"

    best_sim = None
    best_hit = None
    retrieval_match = None

    if should_refuse:
        similarity = 1.0 if did_refuse else 0.0
        correct = did_refuse
        problem = "ok" if did_refuse else "should_have_refused"
    else:
        truth_vec = answer_vectors[truth]

        # a) retrieval check: best retrieved chunk vs correct answer
        if hits:
            best_sim, best_hit = best_chunk_match(truth_vec, hits, chunk_vectors(model, hits))
            retrieval_match = best_sim >= config.RETRIEVAL_MATCH_THRESHOLD
        else:
            best_sim, retrieval_match = 0.0, False

        # b) answer check: generated answer vs correct answer
        if did_refuse:
            similarity = 0.0
        else:
            gen_vec = model.encode([generated], normalize_embeddings=True,
                                   convert_to_numpy=True)[0]
            similarity = float(np.dot(gen_vec, truth_vec))
        correct = similarity >= config.SEMANTIC_THRESHOLD

        if correct:
            problem = "ok"
        elif not retrieval_match:
            problem = "retrieval"        
        else:
            problem = "generation"      # right chunk found, answer wrong or refused

    # hallucination check (only when the system gave an answer)
    faithful = None
    if config.RUN_FAITHFULNESS and not did_refuse:
         
        faithful = check_faithfulness(gemini, question, generated, hits)

    return {
        "question": question,
        "correct_answer": truth,
        "generated_answer": generated,
        "similarity": round(similarity, 3),
        "correct": bool(correct),
        "best_chunk_sim": round(best_sim, 3) if best_sim is not None else None,
        "retrieval_match": retrieval_match,
        "best_chunk_source": source_name(best_hit) if best_hit else None,
        "best_chunk_page": (best_hit.payload or {}).get("page_number") if best_hit else None,
        "problem": problem,
        "reason": out.get("reason"),        # why the pipeline refused (None if it answered)
        "should_refuse": should_refuse,
        "did_refuse": did_refuse,
        "faithful": faithful,
        "top_score": round(float(out["top_score"]), 3),
        "latency_sec": round(latency, 2),
    }


# ---------- progress file (for resume) ----------
def load_progress():
    rows = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def save_progress(row):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


# ---------- Step 4: run all questions one by one ----------
def run_evaluation(client, model, gemini, json_path=None):
    """Runs every question. A failed question does not stop the whole run."""
    items = load_eval_file(json_path)
    answer_vectors = build_answer_embeddings(model, items)

    wanted = {item_key(it["question"], it["answer"]) for it in items}
    rows = [r for r in load_progress()
            if item_key(r.get("question"), r.get("correct_answer")) in wanted
            and "retrieval_match" in r]
    done = {item_key(r["question"], r["correct_answer"]) for r in rows}
    if done:
        print(f"Resuming: {len(done)} questions already finished, skipping them.")

    failed = []
    in_a_row = 0
    limit = getattr(config, "MAX_CONSECUTIVE_FAILURES", 5)
    for number, item in enumerate(items, start=1):
        if item_key(item["question"], item["answer"]) in done:
            continue
        print(f"[{number}/{len(items)}] {item['question']}")

        try:
            row = evaluate_one(client, model, gemini, item, answer_vectors)
        except Exception as error:
            print(f"  FAILED: {error}")
            failed.append(item["question"])
            in_a_row += 1
            if in_a_row >= limit:
                print(f"\nSTOPPED: {in_a_row} questions failed in a row "
                      f"(API limit / quota / network). Finished questions are saved.")
                print("Wait a while (or until the quota resets) and run the SAME command again.")
                break
            continue

        in_a_row = 0
        rows.append(row)
        save_progress(row)
        time.sleep(config.EVAL_DELAY)

    if failed:
        print(f"\n{len(failed)} questions failed (Gemini busy or network problem).")
        print("Run the evaluation again. Finished questions will be skipped.")

    return pd.DataFrame(rows)


# ---------- results ----------
def _rate(series):
    """Mean of a True/False column, ignoring empty (None) values."""
    values = series.dropna().astype(float)
    return float(values.mean()) if len(values) else None


def _pct(value):
    return round(value * 100, 2) if value is not None else None


def calculate_metrics(df):
    """The five report metrics (percent)."""
    answerable = df[~df["should_refuse"]]
    unanswerable = df[df["should_refuse"]]
    faithfulness = _rate(df["faithful"])

    return {
        "Total questions tested": len(df),
        "Retrieval hit rate": _pct(_rate(answerable["retrieval_match"])),
        "Answer accuracy": _pct(_rate(df["correct"])),
        "Correct refusal rate": _pct(_rate(unanswerable["did_refuse"])),
        "Hallucination rate": _pct(None if faithfulness is None else 1 - faithfulness),
    }


def calculate_diagnostics(df):
    """Extra numbers that explain WHY the main metrics look the way they do."""
    answerable = df[~df["should_refuse"]]
    answered = answerable[~answerable["did_refuse"]]

    codes = df["reason"].dropna().astype(str).str.split(":").str[0]

    return {
        "answerable_questions": len(answerable),
        "out_of_scope_questions": int(df["should_refuse"].sum()),
        "system_answered": len(answered),
        "accuracy_when_answered": _pct(_rate(answered["correct"])),
        "false_refusal_rate": _pct(_rate(answerable["did_refuse"])),
        "avg_best_chunk_similarity": round(float(df["best_chunk_sim"].dropna().astype(float).mean()), 3)
                                     if df["best_chunk_sim"].notna().any() else None,
        "problems": {k: int(v) for k, v in df["problem"].value_counts().items()},
        "refusal_reasons": {k: int(v) for k, v in codes.value_counts().items()},
        "avg_latency_sec": round(float(df["latency_sec"].mean()), 2),
    }


def print_summary(metrics, diagnostics):
    print("\n===== Evaluation Results =====")
    for name, value in metrics.items():
        if value is None:
            note = ""
            if name == "Correct refusal rate":
                note = "  (no out-of-scope questions in the JSON)"
            print(f"{name:27s}: Not available{note}")
        elif name == "Total questions tested":
            print(f"{name:27s}: {value}")
        else:
            print(f"{name:27s}: {value:.2f}%")

    print("\n----- Diagnostics -----")
    for name, value in diagnostics.items():
        print(f"{name:27s}: {value}")


def save_results(df, metrics, diagnostics, partial=False):
    tag = "_partial" if partial else ""
    paths = [name.replace(".", tag + ".") for name in (RESULTS_CSV, METRICS_JSON, DIAGNOSTICS_JSON)]
    df.to_csv(paths[0], index=False)
    with open(paths[1], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(paths[2], "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print("\nSaved:", ", ".join(paths))


if __name__ == "__main__":
    from embedding_store import load_embed_model, get_qdrant_client
    from generation import get_gemini_client

    if "--fresh" in sys.argv:
        clear_progress()
        print("Old progress cleared.")

    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    json_path = paths[0] if paths else config.EVAL_FILE

    print("Loading models...")
    model = load_embed_model()
    client = get_qdrant_client()
    gemini = get_gemini_client()

    df = run_evaluation(client, model, gemini, json_path)
    if df.empty:
        print("No question finished, nothing to calculate.")
        sys.exit(1)

    total = len(load_eval_file(json_path))
    complete = len(df) == total

    metrics = calculate_metrics(df)
    diagnostics = calculate_diagnostics(df)
    if not complete:
        print(f"\n*** PARTIAL RESULT: only {len(df)} of {total} questions finished. "
              f"Run again to finish; do not report these numbers. ***")
    print_summary(metrics, diagnostics)
    save_results(df, metrics, diagnostics, partial=not complete)

    # all questions done: remove the progress file  
    if complete:
        clear_progress()