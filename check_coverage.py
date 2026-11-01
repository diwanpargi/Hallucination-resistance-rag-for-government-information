 
import json
import sys

import numpy as np

import config
from embedding_store import load_embed_model, get_qdrant_client
from evaluation import build_answer_embeddings, is_refusal, load_eval_file


def main(json_path=None):
    items = load_eval_file(json_path)
    answerable = [it for it in items if not is_refusal(it["answer"])]

    model = load_embed_model()
    client = get_qdrant_client()
    vectors = build_answer_embeddings(model, answerable)

    scores = []
    for item in answerable:
        vec = np.asarray(vectors[item["answer"]], dtype=float)
        top = client.query_points(config.COLLECTION_NAME, query=vec.tolist(),
                                  limit=1, with_payload=False).points
        scores.append((top[0].score if top else 0.0, item["question"]))

    threshold = config.RETRIEVAL_MATCH_THRESHOLD
    found = [s for s, _ in scores if s >= threshold]

    print(f"\nAnswerable questions        : {len(answerable)}")
    print(f"Answer found in collection  : {len(found)} "
          f"({100 * len(found) / max(len(answerable), 1):.1f}%) at threshold {threshold}")

    for cut in (0.45, 0.50, 0.55, 0.60, 0.65):
        n = sum(1 for s, _ in scores if s >= cut)
        print(f"  similarity >= {cut:.2f}: {n}")

    print("\n20 questions with the weakest match (check these in the PDFs):")
    for score, question in sorted(scores)[:20]:
        print(f"  {score:.2f}  {question[:90]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)