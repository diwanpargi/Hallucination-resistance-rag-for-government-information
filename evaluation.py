# # # # evaluation.py
# # # # Section 5: test the pipeline using a json file of questions and correct answers.
# # # #
# # # # json format (list):
# # # # [
# # # #   {"question": "What is the income limit?", "answer": "The income limit is 2.5 lakh per year."},
# # # #   {"question": "Who won the IPL in 2020?", "answer": "Information not found in the provided documents."}
# # # # ]
# # # # Optional key "source": the PDF name where the answer should be (used for retrieval hit rate).
# # # #
# # # # Steps:
# # # #   1. read the json file
# # # #   2. embed all the correct answers once and store them in a file (so they are not repeated)
# # # #   3. give the questions to the pipeline one by one
# # # #   4. two checks for every question:
# # # #        a) retrieval check : best retrieved chunk  vs  correct answer
# # # #        b) answer check    : generated answer      vs  correct answer
# # # #
# # # # Every finished question is saved in evaluation_progress.jsonl. If the run stops
# # # # (Gemini busy, internet problem, Ctrl+C) just run again: finished questions are skipped.
# # # import json
# # # import os
# # # import time

# # # import numpy as np
# # # import pandas as pd

# # # import config
# # # from generation import build_context, call_gemini, NOT_FOUND


# # # PROGRESS_FILE = "evaluation_progress.jsonl"


# # # # ---------- Step 1: read the json ----------
# # # def load_eval_file(path=None):
# # #     path = path or config.EVAL_FILE

# # #     # if the file is not in the current folder, look next to this python file
# # #     if not os.path.exists(path):
# # #         path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(path))
# # #     if not os.path.exists(path):
# # #         raise FileNotFoundError(f"Evaluation file not found: {path}")

# # #     with open(path, encoding="utf-8") as f:
# # #         items = json.load(f)

# # #     # stop early if the sample answers were not replaced
# # #     for item in items:
# # #         if "PUT THE CORRECT ANSWER" in item["answer"]:
# # #             raise ValueError("Please replace the 'PUT THE CORRECT ANSWER...' text in "
# # #                              "eval_questions.json with the real answers first.")
# # #     return items


# # # def is_refusal(text):
# # #     """True if the text is the 'not found' message."""
# # #     return NOT_FOUND.lower() in text.lower()


# # # # ---------- Step 2: embed the correct answers and store them ----------
# # # def build_answer_embeddings(model, items):
# # #     """
# # #     Embeds every correct answer with the same model used for the chunks (bge-m3).
# # #     The vectors are saved in a file, so next time only new answers are embedded.
# # #     Returns a dict:  answer text -> vector
# # #     """
# # #     stored = {}
# # #     if os.path.exists(config.ANSWER_EMB_FILE):
# # #         data = np.load(config.ANSWER_EMB_FILE, allow_pickle=False)
# # #         stored = dict(zip(data["texts"].tolist(), data["vectors"]))

# # #     # "not found" answers need no embedding
# # #     new_texts = [it["answer"] for it in items if not is_refusal(it["answer"])]
# # #     new_texts = [t for t in dict.fromkeys(new_texts) if t not in stored]

# # #     if new_texts:
# # #         print(f"Embedding {len(new_texts)} correct answers...")
# # #         vectors = model.encode(new_texts, batch_size=16, show_progress_bar=True,
# # #                                normalize_embeddings=True, convert_to_numpy=True)
# # #         for text, vec in zip(new_texts, vectors):
# # #             stored[text] = vec
# # #         np.savez(config.ANSWER_EMB_FILE,
# # #                  texts=np.array(list(stored.keys())),
# # #                  vectors=np.array(list(stored.values())))
# # #         print("Saved:", config.ANSWER_EMB_FILE)
# # #     else:
# # #         print("Correct answer embeddings loaded from file.")

# # #     return stored


# # # # ---------- Step 4: the two checks ----------
# # # def best_chunk_match(answer_vec, hits):
# # #     """Finds the retrieved chunk whose vector is closest to the correct answer vector."""
# # #     best_sim, best_hit = 0.0, None
# # #     for p in hits:
# # #         sim = float(np.dot(answer_vec, np.array(p.vector)))   # vectors are normalized = cosine
# # #         if best_hit is None or sim > best_sim:
# # #             best_sim, best_hit = sim, p
# # #     return best_sim, best_hit


# # # def check_faithfulness(gemini, question, answer, hits):
# # #     """Ask Gemini: is every fact in the answer supported by the retrieved context? YES/NO"""
# # #     prompt = f"""You are checking an answer for hallucination.

# # # Context:
# # # {build_context(hits)}

# # # Question: {question}
# # # Answer: {answer}

# # # Is every fact in the Answer supported by the Context?
# # # Reply with only one word: YES or NO."""
# # #     return call_gemini(gemini, prompt).upper().startswith("YES")


# # # def evaluate_one(client, model, gemini, item, answer_vectors):
# # #     """Runs one question through the pipeline and checks it against the correct answer."""
# # #     question = item["question"]
# # #     truth = item["answer"]

# # #     # run the full RAG pipeline (with_vectors=True gives us the chunk vectors)
# # #     start = time.time()
# # #     out = answer_question(client, model, gemini, question, with_vectors=True)
# # #     latency = time.time() - start
# # #     generated, hits = out["answer"], out["hits"]

# # #     should_refuse = is_refusal(truth)       # correct answer is "not found"
# # #     did_refuse = is_refusal(generated)      # system said "not found"

# # #     best_sim = None
# # #     best_hit = None
# # #     retrieval_match = None

# # #     if should_refuse:
# # #         # nothing to match, the system just has to say "not found"
# # #         similarity = 1.0 if did_refuse else 0.0
# # #         correct = did_refuse
# # #         problem = "ok" if did_refuse else "should_have_refused"
# # #     else:
# # #         truth_vec = answer_vectors[truth]

# # #         # a) retrieval check: best chunk vs correct answer
# # #         best_sim, best_hit = best_chunk_match(truth_vec, hits)
# # #         retrieval_match = best_sim >= config.RETRIEVAL_MATCH_THRESHOLD

# # #         # b) answer check: generated answer vs correct answer
# # #         if did_refuse:
# # #             similarity = 0.0
# # #         else:
# # #             gen_vec = model.encode([generated], normalize_embeddings=True, convert_to_numpy=True)[0]
# # #             similarity = float(np.dot(gen_vec, truth_vec))
# # #         correct = similarity >= config.SEMANTIC_THRESHOLD

# # #         # where did it go wrong?
# # #         if correct:
# # #             problem = "ok"
# # #         elif not retrieval_match:
# # #             problem = "retrieval"       # the right chunk was not found
# # #         else:
# # #             problem = "generation"      # the right chunk was found but the answer is wrong

# # #     # hallucination check (only when the system gave an answer)
# # #     faithful = None
# # #     if config.RUN_FAITHFULNESS and not did_refuse:
# # #         faithful = check_faithfulness(gemini, question, generated, hits)

# # #     # source hit (only if "source" is given in the json)
# # #     retrieval_hit = None
# # #     if item.get("source"):
# # #         retrieval_hit = any(p.payload.get("source") == item["source"] for p in hits)

# # #     return {
# # #         "question": question,
# # #         "correct_answer": truth,
# # #         "generated_answer": generated,
# # #         "similarity": round(similarity, 3),
# # #         "correct": bool(correct),
# # #         "best_chunk_sim": round(best_sim, 3) if best_sim is not None else None,
# # #         "retrieval_match": retrieval_match,
# # #         "best_chunk_source": best_hit.payload.get("source") if best_hit else None,
# # #         "best_chunk_page": best_hit.payload.get("page_number") if best_hit else None,
# # #         "problem": problem,
# # #         "reason": out.get("reason"),        # why the pipeline refused (None if it answered)
# # #         "should_refuse": should_refuse,
# # #         "did_refuse": did_refuse,
# # #         "faithful": faithful,
# # #         "retrieval_hit": retrieval_hit,
# # #         "top_score": round(float(out["top_score"]), 3),
# # #         "latency_sec": round(latency, 2),
# # #     }


# # # # ---------- progress file (for resume) ----------
# # # def load_progress():
# # #     rows = []
# # #     if os.path.exists(PROGRESS_FILE):
# # #         with open(PROGRESS_FILE, encoding="utf-8") as f:
# # #             for line in f:
# # #                 if line.strip():
# # #                     rows.append(json.loads(line))
# # #     return rows


# # # def save_progress(row):
# # #     with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
# # #         f.write(json.dumps(row, ensure_ascii=False) + "\n")


# # # def clear_progress():
# # #     if os.path.exists(PROGRESS_FILE):
# # #         os.remove(PROGRESS_FILE)


# # # # ---------- Step 3: run all questions one by one ----------
# # # def run_evaluation(client, model, gemini, json_path=None):
# # #     """Runs every question one by one. A failed question does not stop the whole run."""
# # #     items = load_eval_file(json_path)
# # #     answer_vectors = build_answer_embeddings(model, items)

# # #     # questions finished in an earlier run are not repeated
# # #     wanted = {item["question"] for item in items}
# # #     rows = [r for r in load_progress() if r["question"] in wanted and "best_chunk_sim" in r]
# # #     done = {r["question"] for r in rows}
# # #     if done:
# # #         print(f"Resuming: {len(done)} questions were already finished, skipping them.")

# # #     failed = []
# # #     for n, item in enumerate(items, start=1):
# # #         if item["question"] in done:
# # #             continue
# # #         print(f"[{n}/{len(items)}] {item['question']}")

# # #         try:
# # #             row = evaluate_one(client, model, gemini, item, answer_vectors)
# # #         except Exception as e:
# # #             print(f"  FAILED: {e}")
# # #             failed.append(item["question"])
# # #             continue

# # #         rows.append(row)
# # #         save_progress(row)
# # #         time.sleep(config.EVAL_DELAY)

# # #     if failed:
# # #         print(f"\n{len(failed)} questions failed (Gemini busy or network problem).")
# # #         print("Run the evaluation again. The finished questions will be skipped.")

# # #     return pd.DataFrame(rows)


# # # # ---------- results ----------
# # # def _rate(series):
# # #     """Mean of a True/False column, ignoring empty (None) values."""
# # #     values = series.dropna().astype(float)
# # #     return round(float(values.mean()), 3) if len(values) else None


# # # def calculate_metrics(df):
# # #     """Calculates the main performance numbers from the result table."""
# # #     answerable = df[~df["should_refuse"]]
# # #     unanswerable = df[df["should_refuse"]]
# # #     answered = df[~df["did_refuse"]]

# # #     # correct answers given to answerable questions
# # #     tp = int(((~df["should_refuse"]) & (~df["did_refuse"]) & df["correct"]).sum())

# # #     precision = tp / len(answered) if len(answered) else 0.0
# # #     recall = tp / len(answerable) if len(answerable) else 0.0
# # #     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

# # #     faithfulness = _rate(df["faithful"])

# # #     # why the pipeline refused: only the short code before ":"
# # #     refusal_reasons = {}
# # #     if "reason" in df:
# # #         codes = df["reason"].dropna().astype(str).str.split(":").str[0]
# # #         refusal_reasons = {k: int(v) for k, v in codes.value_counts().items()}

# # #     chunk_sims = df["best_chunk_sim"].dropna().astype(float)

# # #     return {
# # #         "total_questions": len(df),
# # #         "accuracy": _rate(df["correct"]),
# # #         "avg_semantic_similarity": round(float(df["similarity"].mean()), 3),
# # #         "precision": round(precision, 3),
# # #         "recall": round(recall, 3),
# # #         "f1_score": round(f1, 3),
# # #         "retrieval_match_rate": _rate(df["retrieval_match"]),     # best chunk matched the correct answer
# # #         "avg_best_chunk_similarity": round(float(chunk_sims.mean()), 3) if len(chunk_sims) else None,
# # #         "faithfulness": faithfulness,
# # #         "hallucination_rate": round(1 - faithfulness, 3) if faithfulness is not None else None,
# # #         "correct_refusal_rate": _rate(unanswerable["did_refuse"]),   # out-of-scope questions refused
# # #         "false_refusal_rate": _rate(answerable["did_refuse"]),       # answerable questions refused
# # #         "source_hit_rate": _rate(df["retrieval_hit"]),
# # #         "avg_latency_sec": round(float(df["latency_sec"].mean()), 2),
# # #         "problems": {k: int(v) for k, v in df["problem"].value_counts().items()},
# # #         "refusal_reasons": refusal_reasons,
# # #     }


# # # def print_summary(metrics):
# # #     print("\n===== Evaluation Result =====")
# # #     for name, value in metrics.items():
# # #         print(f"{name:27s}: {value}")


# # # def save_results(df, metrics, csv_path="evaluation_results.csv", json_path="evaluation_metrics.json"):
# # #     df.to_csv(csv_path, index=False)
# # #     with open(json_path, "w", encoding="utf-8") as f:
# # #         json.dump(metrics, f, indent=2)
# # #     print("Saved:", csv_path, "and", json_path)


# # # if __name__ == "__main__":
# # #     # run:  python evaluation.py            (uses eval_questions.json)
# # #     #       python evaluation.py my_file.json
# # #     import sys
# # #     from embedding_store import load_embed_model, get_qdrant_client
# # #     from generation import get_gemini_client

# # #     json_path = sys.argv[1] if len(sys.argv) > 1 else config.EVAL_FILE

# # #     print("Loading models...")
# # #     model = load_embed_model()
# # #     client = get_qdrant_client()
# # #     gemini = get_gemini_client()

# # #     df = run_evaluation(client, model, gemini, json_path)
# # #     if df.empty:
# # #         print("No question finished, nothing to calculate.")
# # #         sys.exit(1)

# # #     metrics = calculate_metrics(df)
# # #     print_summary(metrics)
# # #     save_results(df, metrics)

# # #     # all questions done: remove the progress file so the next run starts fresh
# # #     if len(df) == len(load_eval_file(json_path)):
# # #         clear_progress()
# # #     else:
# # #         print("\nSome questions are missing, so these numbers are partial.")


# # # evaluation.py
# # # evaluation.py
# # # Section 5: test the pipeline using a JSON file of questions and correct answers.
# # #
# # # Optional key "source": the PDF name where the answer should be.
# # #
# # # Finished questions are saved in evaluation_progress.jsonl.
# # # If the run stops, run it again to skip completed questions.

# # import json
# # import os
# # import time

# # import numpy as np
# # import pandas as pd

# # import config
# # from generation import build_context, call_gemini, NOT_FOUND


# # PROGRESS_FILE = "evaluation_progress1.jsonl"


# # # ---------- Step 1: read the JSON ----------
# # def load_eval_file(path=None):
# #     path = path or config.EVAL_FILE

# #     if not os.path.exists(path):
# #         path = os.path.join(
# #             os.path.dirname(os.path.abspath(__file__)),
# #             os.path.basename(path)
# #         )

# #     if not os.path.exists(path):
# #         raise FileNotFoundError(f"Evaluation file not found: {path}")

# #     with open(path, encoding="utf-8") as f:
# #         items = json.load(f)

# #     for item in items:
# #         if "PUT THE CORRECT ANSWER" in item["answer"]:
# #             raise ValueError(
# #                 "Please replace the 'PUT THE CORRECT ANSWER...' text "
# #                 "in eval_questions.json with the real answers first."
# #             )

# #     return items


# # def is_refusal(text):
# #     """True if the text is the 'not found' message."""
# #     return NOT_FOUND.lower() in text.lower()


# # # ---------- Step 2: embed the correct answers and store them ----------
# # def build_answer_embeddings(model, items):
# #     """
# #     Embed correct answers once and cache their vectors.
# #     Returns a dictionary: answer text -> vector.
# #     """
# #     stored = {}

# #     if os.path.exists(config.ANSWER_EMB_FILE):
# #         data = np.load(config.ANSWER_EMB_FILE, allow_pickle=False)
# #         stored = dict(zip(data["texts"].tolist(), data["vectors"]))

# #     new_texts = [
# #         item["answer"] for item in items
# #         if not is_refusal(item["answer"])
# #     ]
# #     new_texts = [text for text in dict.fromkeys(new_texts) if text not in stored]

# #     if new_texts:
# #         print(f"Embedding {len(new_texts)} correct answers...")
# #         vectors = model.encode(
# #             new_texts,
# #             batch_size=16,
# #             show_progress_bar=True,
# #             normalize_embeddings=True,
# #             convert_to_numpy=True
# #         )

# #         for answer, vector in zip(new_texts, vectors):
# #             stored[answer] = vector

# #         np.savez(
# #             config.ANSWER_EMB_FILE,
# #             texts=np.array(list(stored.keys())),
# #             vectors=np.array(list(stored.values()))
# #         )
# #         print("Saved:", config.ANSWER_EMB_FILE)
# #     else:
# #         print("Correct answer embeddings loaded from file.")

# #     return stored


# # # ---------- Retrieval and faithfulness checks ----------
# # def best_chunk_match(answer_vec, hits):
# #     """Find the retrieved chunk most similar to the correct answer."""
# #     best_sim, best_hit = 0.0, None

# #     for hit in hits:
# #         sim = float(np.dot(answer_vec, np.array(hit.vector)))
# #         if best_hit is None or sim > best_sim:
# #             best_sim, best_hit = sim, hit

# #     return best_sim, best_hit


# # def check_faithfulness(gemini, question, answer, hits):
# #     """Check whether every answer fact is supported by retrieved context."""
# #     prompt = f"""You are checking an answer for hallucination.

# # Context:
# # {build_context(hits)}

# # Question: {question}
# # Answer: {answer}

# # Is every fact in the Answer supported by the Context?
# # Reply with only one word: YES or NO."""

# #     return call_gemini(gemini, prompt).upper().startswith("YES")


# # def evaluate_one(client, model, gemini, item, answer_vectors):
# #     """Run one question through the pipeline and evaluate its answer."""
# #     question = item["question"]
# #     truth = item["answer"]

# #     start = time.time()
# #     out = answer_question(client, model, gemini, question, with_vectors=True)
# #     latency = time.time() - start

# #     generated, hits = out["answer"], out["hits"]

# #     should_refuse = is_refusal(truth)
# #     did_refuse = is_refusal(generated)

# #     best_sim = None
# #     best_hit = None
# #     retrieval_match = None

# #     if should_refuse:
# #         similarity = 1.0 if did_refuse else 0.0
# #         correct = did_refuse
# #         problem = "ok" if did_refuse else "should_have_refused"
# #     else:
# #         truth_vec = answer_vectors[truth]

# #         best_sim, best_hit = best_chunk_match(truth_vec, hits)
# #         retrieval_match = best_sim >= config.RETRIEVAL_MATCH_THRESHOLD

# #         if did_refuse:
# #             similarity = 0.0
# #         else:
# #             gen_vec = model.encode(
# #                 [generated],
# #                 normalize_embeddings=True,
# #                 convert_to_numpy=True
# #             )[0]
# #             similarity = float(np.dot(gen_vec, truth_vec))

# #         correct = similarity >= config.SEMANTIC_THRESHOLD

# #         if correct:
# #             problem = "ok"
# #         elif not retrieval_match:
# #             problem = "retrieval"
# #         else:
# #             problem = "generation"

# #     faithful = None
# #     if config.RUN_FAITHFULNESS and not did_refuse:
# #         faithful = check_faithfulness(gemini, question, generated, hits)

# #     retrieval_hit = None
# #     if item.get("source"):
# #         retrieval_hit = any(
# #             hit.payload.get("source") == item["source"]
# #             for hit in hits
# #         )

# #     return {
# #         "question": question,
# #         "correct_answer": truth,
# #         "generated_answer": generated,
# #         "similarity": round(similarity, 3),
# #         "correct": bool(correct),
# #         "best_chunk_sim": round(best_sim, 3) if best_sim is not None else None,
# #         "retrieval_match": retrieval_match,
# #         "best_chunk_source": best_hit.payload.get("source") if best_hit else None,
# #         "best_chunk_page": best_hit.payload.get("page_number") if best_hit else None,
# #         "problem": problem,
# #         "reason": out.get("reason"),
# #         "should_refuse": should_refuse,
# #         "did_refuse": did_refuse,
# #         "faithful": faithful,
# #         "retrieval_hit": retrieval_hit,
# #         "top_score": round(float(out["top_score"]), 3),
# #         "latency_sec": round(latency, 2),
# #     }


# # # ---------- Progress file (for resume) ----------
# # def load_progress():
# #     rows = []

# #     if os.path.exists(PROGRESS_FILE):
# #         with open(PROGRESS_FILE, encoding="utf-8") as f:
# #             for line in f:
# #                 if line.strip():
# #                     rows.append(json.loads(line))

# #     return rows


# # def save_progress(row):
# #     with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
# #         f.write(json.dumps(row, ensure_ascii=False) + "\n")


# # def clear_progress():
# #     if os.path.exists(PROGRESS_FILE):
# #         os.remove(PROGRESS_FILE)


# # # ---------- Step 3: run all questions one by one ----------
# # def run_evaluation(client, model, gemini, json_path=None):
# #     """Run every question; a failed question does not stop the full run."""
# #     items = load_eval_file(json_path)
# #     answer_vectors = build_answer_embeddings(model, items)

# #     wanted = {item["question"] for item in items}
# #     rows = [
# #         row for row in load_progress()
# #         if row["question"] in wanted and "best_chunk_sim" in row
# #     ]
# #     done = {row["question"] for row in rows}

# #     if done:
# #         print(f"Resuming: {len(done)} questions already finished; skipping them.")

# #     failed = []

# #     for number, item in enumerate(items, start=1):
# #         if item["question"] in done:
# #             continue

# #         print(f"[{number}/{len(items)}] {item['question']}")

# #         try:
# #             row = evaluate_one(client, model, gemini, item, answer_vectors)
# #         except Exception as error:
# #             print(f"  FAILED: {error}")
# #             failed.append(item["question"])
# #             continue

# #         rows.append(row)
# #         save_progress(row)
# #         time.sleep(config.EVAL_DELAY)

# #     if failed:
# #         print(f"\n{len(failed)} questions failed (Gemini busy or network problem).")
# #         print("Run the evaluation again. Finished questions will be skipped.")

# #     return pd.DataFrame(rows)


# # # ---------- Results: only the five requested metrics ----------
# # def _rate(series):
# #     """Mean of a True/False series, ignoring missing values."""
# #     values = series.dropna().astype(float)
# #     return float(values.mean()) if len(values) else None


# # def calculate_metrics(df):
# #     """Calculate only the five requested evaluation metrics."""
# #     faithfulness = _rate(df["faithful"])
# #     source_hit_rate = _rate(df["retrieval_hit"])

# #     # Correct refusal rate applies only to questions whose correct answer is
# #     # that the information is not found in the provided documents.
# #     unanswerable = df[df["should_refuse"]]
# #     correct_refusal_rate = _rate(unanswerable["did_refuse"])

# #     return {
# #         "Total questions tested": len(df),
# #         "Retrieval hit rate": (
# #             round(source_hit_rate * 100, 2)
# #             if source_hit_rate is not None else None
# #         ),
# #         "Answer accuracy": (
# #             round(_rate(df["correct"]) * 100, 2)
# #             if _rate(df["correct"]) is not None else None
# #         ),
# #         "Correct refusal rate": (
# #             round(correct_refusal_rate * 100, 2)
# #             if correct_refusal_rate is not None else None
# #         ),
# #         "Hallucination rate": (
# #             round((1 - faithfulness) * 100, 2)
# #             if faithfulness is not None else None
# #         ),
# #     }


# # def print_summary(metrics):
# #     print("\n===== Evaluation Results =====")

# #     for name, value in metrics.items():
# #         if value is None:
# #             print(f"{name:27s}: Not available")
# #         elif name == "Total questions tested":
# #             print(f"{name:27s}: {value}")
# #         else:
# #             print(f"{name:27s}: {value:.2f}%")


# # def save_results(df, metrics, csv_path="evaluation_results1.csv",
# #                  json_path="evaluation_metrics1.json"):
# #     # Keep the detailed per-question CSV for debugging and analysis.
# #     # The JSON summary contains only the five requested metrics.
# #     df.to_csv(csv_path, index=False)

# #     with open(json_path, "w", encoding="utf-8") as f:
# #         json.dump(metrics, f, indent=2)

# #     print("Saved:", csv_path, "and", json_path)


# # if __name__ == "__main__":
# #     import sys
# #     from embedding_store import load_embed_model, get_qdrant_client
# #     from generation import get_gemini_client
# #     from server import answer_question

# #     json_path = sys.argv[1] if len(sys.argv) > 1 else config.EVAL_FILE

# #     print("Loading models...")
# #     model = load_embed_model()
# #     client = get_qdrant_client()
# #     gemini = get_gemini_client()

# #     df = run_evaluation(client, model, gemini, json_path)

# #     if df.empty:
# #         print("No question finished, nothing to calculate.")
# #         sys.exit(1)

# #     metrics = calculate_metrics(df)
# #     print_summary(metrics)
# #     save_results(df, metrics)

# #     if len(df) == len(load_eval_file(json_path)):
# #         clear_progress()
# #     else:
# #         print("\nSome questions are missing, so these numbers are partial.")

# # evaluation.py
# # Tests the RAG pipeline with a JSON file of questions and correct answers.
# #
# # JSON format (list):
# # [
# #   {"question": "...", "answer": "..."},
# #   {"question": "Who won the IPL in 2020?", "answer": "Information not found in the provided documents."}
# # ]
# # (A "source" key is allowed but IGNORED. Retrieval is judged by content only.)
# #
# # For every question two checks are made:
# #   a) retrieval check : is any retrieved chunk similar to the correct answer?
# #   b) answer check    : is the generated answer similar to the correct answer?
# #
# # Finished questions are saved in PROGRESS_FILE. If the run stops, run again and
# # finished questions are skipped.   Use  --fresh  to ignore old progress.
# #
# # Run:   python evaluation.py
# #        python evaluation.py my_questions.json
# #        python evaluation.py --fresh

# import json
# import os
# import sys
# import time

# import numpy as np
# import pandas as pd

# import config
# from generation import build_context, call_gemini, NOT_FOUND


# PROGRESS_FILE = "evaluation_progress2.jsonl"
# RESULTS_CSV = "evaluation_results2.csv"
# METRICS_JSON = "evaluation_metrics2.json"
# DIAGNOSTICS_JSON = "evaluation_diagnostics2.json"


# # ---------- Step 1: read the JSON ----------
# def load_eval_file(path=None):
#     path = path or config.EVAL_FILE

#     if not os.path.exists(path):
#         path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                             os.path.basename(path))
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Evaluation file not found: {path}")

#     with open(path, encoding="utf-8") as f:
#         items = json.load(f)

#     for number, item in enumerate(items, start=1):
#         if not str(item.get("question", "")).strip() or not str(item.get("answer", "")).strip():
#             raise ValueError(f"Item {number} has an empty question or answer.")
#         if "PUT THE CORRECT ANSWER" in item["answer"]:
#             raise ValueError("Please replace the 'PUT THE CORRECT ANSWER...' text "
#                              "in the JSON with the real answers first.")
#         item["question"] = item["question"].strip()

#     # same question with different answers is ambiguous for the system
#     seen = {}
#     for item in items:
#         seen.setdefault(item["question"].lower(), set()).add(item["answer"])
#     repeated = sum(1 for answers in seen.values() if len(answers) > 1)
#     if repeated:
#         print(f"WARNING: {repeated} questions appear more than once with different "
#               f"answers. Add the scheme/document name to the question text.")

#     return items


# def is_refusal(text):
#     """True if the text is the 'not found' message."""
#     return NOT_FOUND.lower() in str(text).lower()


# def item_key(question, answer):
#     """A question is identified by its text AND its correct answer."""
#     return (question, answer)


# # ---------- Step 2: embed the correct answers and store them ----------
# def build_answer_embeddings(model, items):
#     """
#     Embeds every correct answer once (same model as the chunks) and caches the
#     vectors in a file.  Returns a dict:  answer text -> vector
#     """
#     stored = {}
#     if os.path.exists(config.ANSWER_EMB_FILE):
#         data = np.load(config.ANSWER_EMB_FILE, allow_pickle=False)
#         stored = dict(zip(data["texts"].tolist(), data["vectors"]))

#     new_texts = [it["answer"] for it in items if not is_refusal(it["answer"])]
#     new_texts = [t for t in dict.fromkeys(new_texts) if t not in stored]

#     if new_texts:
#         print(f"Embedding {len(new_texts)} correct answers...")
#         vectors = model.encode(new_texts, batch_size=16, show_progress_bar=True,
#                                normalize_embeddings=True, convert_to_numpy=True)
#         for text, vec in zip(new_texts, vectors):
#             stored[text] = vec
#         np.savez(config.ANSWER_EMB_FILE,
#                  texts=np.array(list(stored.keys())),
#                  vectors=np.array(list(stored.values())))
#         print("Saved:", config.ANSWER_EMB_FILE)
#     else:
#         print("Correct answer embeddings loaded from file.")

#     return stored


# # ---------- Step 3: the two checks ----------
# def chunk_vectors(model, hits):
#     """
#     Returns a (num_hits, dim) array of normalized chunk vectors.
#     Uses the vectors returned by Qdrant when available, otherwise
#     re-embeds the chunk text (so it works even without with_vectors=True).
#     """
#     vectors = [getattr(h, "vector", None) for h in hits]
#     usable = all(v is not None and not isinstance(v, dict) for v in vectors)

#     if usable:
#         array = np.array(vectors, dtype=float)
#     else:
#         texts = [(h.payload or {}).get("text", "") for h in hits]
#         array = np.array(model.encode(texts, normalize_embeddings=True,
#                                       convert_to_numpy=True), dtype=float)

#     norms = np.linalg.norm(array, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return array / norms


# def best_chunk_match(answer_vec, hits, chunk_vecs):
#     """Finds the retrieved chunk closest to the correct answer (cosine)."""
#     if not hits:
#         return 0.0, None
#     sims = chunk_vecs @ np.asarray(answer_vec, dtype=float)
#     best = int(np.argmax(sims))
#     return float(sims[best]), hits[best]


# def check_faithfulness(gemini, question, answer, hits):
#     """Ask Gemini: is every fact in the answer supported by the context? YES/NO"""
#     prompt = f"""You are checking an answer for hallucination.

# Context:
# {build_context(hits)}

# Question: {question}
# Answer: {answer}

# Is every fact in the Answer supported by the Context?
# Reply with only one word: YES or NO."""
#     return call_gemini(gemini, prompt).upper().startswith("YES")


# def source_name(hit):
#     """File name only (no folders), for the CSV."""
#     raw = str((hit.payload or {}).get("source", "")).replace("\\", "/")
#     return os.path.basename(raw)


# def evaluate_one(client, model, gemini, item, answer_vectors):
#     """Runs one question through the pipeline and checks it against the correct answer."""
#     from server import answer_question      # imported here so this file also works as a module

#     question = item["question"]
#     truth = item["answer"]

#     start = time.time()
#     out = answer_question(client, model, gemini, question, with_vectors=True)
#     latency = time.time() - start

#     generated, hits = out["answer"], out["hits"]

#     should_refuse = is_refusal(truth)       # correct answer is "not found"
#     did_refuse = is_refusal(generated)      # system said "not found"

#     best_sim = None
#     best_hit = None
#     retrieval_match = None

#     if should_refuse:
#         similarity = 1.0 if did_refuse else 0.0
#         correct = did_refuse
#         problem = "ok" if did_refuse else "should_have_refused"
#     else:
#         truth_vec = answer_vectors[truth]

#         # a) retrieval check: best retrieved chunk vs correct answer
#         if hits:
#             best_sim, best_hit = best_chunk_match(truth_vec, hits, chunk_vectors(model, hits))
#             retrieval_match = best_sim >= config.RETRIEVAL_MATCH_THRESHOLD
#         else:
#             best_sim, retrieval_match = 0.0, False

#         # b) answer check: generated answer vs correct answer
#         if did_refuse:
#             similarity = 0.0
#         else:
#             gen_vec = model.encode([generated], normalize_embeddings=True,
#                                    convert_to_numpy=True)[0]
#             similarity = float(np.dot(gen_vec, truth_vec))
#         correct = similarity >= config.SEMANTIC_THRESHOLD

#         if correct:
#             problem = "ok"
#         elif not retrieval_match:
#             problem = "retrieval"       # the right chunk was not found
#         else:
#             problem = "generation"      # right chunk found, answer wrong or refused

#     # hallucination check (only when the system gave an answer)
#     faithful = None
#     if config.RUN_FAITHFULNESS and not did_refuse:
#         try:
#             faithful = check_faithfulness(gemini, question, generated, hits)
#         except Exception as error:
#             print(f"  faithfulness check failed: {error}")

#     return {
#         "question": question,
#         "correct_answer": truth,
#         "generated_answer": generated,
#         "similarity": round(similarity, 3),
#         "correct": bool(correct),
#         "best_chunk_sim": round(best_sim, 3) if best_sim is not None else None,
#         "retrieval_match": retrieval_match,
#         "best_chunk_source": source_name(best_hit) if best_hit else None,
#         "best_chunk_page": (best_hit.payload or {}).get("page_number") if best_hit else None,
#         "problem": problem,
#         "reason": out.get("reason"),        # why the pipeline refused (None if it answered)
#         "should_refuse": should_refuse,
#         "did_refuse": did_refuse,
#         "faithful": faithful,
#         "top_score": round(float(out["top_score"]), 3),
#         "latency_sec": round(latency, 2),
#     }


# # ---------- progress file (for resume) ----------
# def load_progress():
#     rows = []
#     if os.path.exists(PROGRESS_FILE):
#         with open(PROGRESS_FILE, encoding="utf-8") as f:
#             for line in f:
#                 if line.strip():
#                     rows.append(json.loads(line))
#     return rows


# def save_progress(row):
#     with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
#         f.write(json.dumps(row, ensure_ascii=False) + "\n")


# def clear_progress():
#     if os.path.exists(PROGRESS_FILE):
#         os.remove(PROGRESS_FILE)


# # ---------- Step 4: run all questions one by one ----------
# def run_evaluation(client, model, gemini, json_path=None):
#     """Runs every question. A failed question does not stop the whole run."""
#     items = load_eval_file(json_path)
#     answer_vectors = build_answer_embeddings(model, items)

#     wanted = {item_key(it["question"], it["answer"]) for it in items}
#     rows = [r for r in load_progress()
#             if item_key(r.get("question"), r.get("correct_answer")) in wanted
#             and "retrieval_match" in r]
#     done = {item_key(r["question"], r["correct_answer"]) for r in rows}
#     if done:
#         print(f"Resuming: {len(done)} questions already finished, skipping them.")

#     failed = []
#     for number, item in enumerate(items, start=1):
#         if item_key(item["question"], item["answer"]) in done:
#             continue
#         print(f"[{number}/{len(items)}] {item['question']}")

#         try:
#             row = evaluate_one(client, model, gemini, item, answer_vectors)
#         except Exception as error:
#             print(f"  FAILED: {error}")
#             failed.append(item["question"])
#             continue

#         rows.append(row)
#         save_progress(row)
#         time.sleep(config.EVAL_DELAY)

#     if failed:
#         print(f"\n{len(failed)} questions failed (Gemini busy or network problem).")
#         print("Run the evaluation again. Finished questions will be skipped.")

#     return pd.DataFrame(rows)


# # ---------- results ----------
# def _rate(series):
#     """Mean of a True/False column, ignoring empty (None) values."""
#     values = series.dropna().astype(float)
#     return float(values.mean()) if len(values) else None


# def _pct(value):
#     return round(value * 100, 2) if value is not None else None


# def calculate_metrics(df):
#     """The five report metrics (percent)."""
#     answerable = df[~df["should_refuse"]]
#     unanswerable = df[df["should_refuse"]]
#     faithfulness = _rate(df["faithful"])

#     return {
#         "Total questions tested": len(df),
#         "Retrieval hit rate": _pct(_rate(answerable["retrieval_match"])),
#         "Answer accuracy": _pct(_rate(df["correct"])),
#         "Correct refusal rate": _pct(_rate(unanswerable["did_refuse"])),
#         "Hallucination rate": _pct(None if faithfulness is None else 1 - faithfulness),
#     }


# def calculate_diagnostics(df):
#     """Extra numbers that explain WHY the main metrics look the way they do."""
#     answerable = df[~df["should_refuse"]]
#     answered = answerable[~answerable["did_refuse"]]

#     codes = df["reason"].dropna().astype(str).str.split(":").str[0]

#     return {
#         "answerable_questions": len(answerable),
#         "out_of_scope_questions": int(df["should_refuse"].sum()),
#         "system_answered": len(answered),
#         "accuracy_when_answered": _pct(_rate(answered["correct"])),
#         "false_refusal_rate": _pct(_rate(answerable["did_refuse"])),
#         "avg_best_chunk_similarity": round(float(df["best_chunk_sim"].dropna().astype(float).mean()), 3)
#                                      if df["best_chunk_sim"].notna().any() else None,
#         "problems": {k: int(v) for k, v in df["problem"].value_counts().items()},
#         "refusal_reasons": {k: int(v) for k, v in codes.value_counts().items()},
#         "avg_latency_sec": round(float(df["latency_sec"].mean()), 2),
#     }


# def print_summary(metrics, diagnostics):
#     print("\n===== Evaluation Results =====")
#     for name, value in metrics.items():
#         if value is None:
#             note = ""
#             if name == "Correct refusal rate":
#                 note = "  (no out-of-scope questions in the JSON)"
#             print(f"{name:27s}: Not available{note}")
#         elif name == "Total questions tested":
#             print(f"{name:27s}: {value}")
#         else:
#             print(f"{name:27s}: {value:.2f}%")

#     print("\n----- Diagnostics -----")
#     for name, value in diagnostics.items():
#         print(f"{name:27s}: {value}")


# def save_results(df, metrics, diagnostics):
#     df.to_csv(RESULTS_CSV, index=False)
#     with open(METRICS_JSON, "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)
#     with open(DIAGNOSTICS_JSON, "w", encoding="utf-8") as f:
#         json.dump(diagnostics, f, indent=2)
#     print(f"\nSaved: {RESULTS_CSV}, {METRICS_JSON}, {DIAGNOSTICS_JSON}")


# if __name__ == "__main__":
#     from embedding_store import load_embed_model, get_qdrant_client
#     from generation import get_gemini_client

#     if "--fresh" in sys.argv:
#         clear_progress()
#         print("Old progress cleared.")

#     paths = [a for a in sys.argv[1:] if not a.startswith("--")]
#     json_path = paths[0] if paths else config.EVAL_FILE

#     print("Loading models...")
#     model = load_embed_model()
#     client = get_qdrant_client()
#     gemini = get_gemini_client()

#     df = run_evaluation(client, model, gemini, json_path)
#     if df.empty:
#         print("No question finished, nothing to calculate.")
#         sys.exit(1)

#     metrics = calculate_metrics(df)
#     diagnostics = calculate_diagnostics(df)
#     print_summary(metrics, diagnostics)
#     save_results(df, metrics, diagnostics)

#     # all questions done: remove the progress file so the next run starts fresh
#     if len(df) == len(load_eval_file(json_path)):
#         clear_progress()
#     else:
#         print("\nSome questions are missing, so these numbers are partial.")

# evaluation.py
# Tests the RAG pipeline with a JSON file of questions and correct answers.
#
# JSON format (list):
# [
#   {"question": "...", "answer": "..."},
#   {"question": "Who won the IPL in 2020?", "answer": "Information not found in the provided documents."}
# ]
# (A "source" key is allowed but IGNORED. Retrieval is judged by content only.)
#
# For every question two checks are made:
#   a) retrieval check : is any retrieved chunk similar to the correct answer?
#   b) answer check    : is the generated answer similar to the correct answer?
#
# Finished questions are saved in PROGRESS_FILE. If the run stops, run again and
# finished questions are skipped.   Use  --fresh  to ignore old progress.
#
# Run:   python evaluation.py
#        python evaluation.py my_questions.json
#        python evaluation.py --fresh

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
            problem = "retrieval"       # the right chunk was not found
        else:
            problem = "generation"      # right chunk found, answer wrong or refused

    # hallucination check (only when the system gave an answer)
    faithful = None
    if config.RUN_FAITHFULNESS and not did_refuse:
        # no try/except on purpose: if this fails (API limit), the question is marked
        # FAILED and is repeated on the next run, instead of being saved without a value
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

    # all questions done: remove the progress file so the next run starts fresh
    if complete:
        clear_progress()