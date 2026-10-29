 
# # import config
# # #retrievel

# # def retrieve(client, model, question, top_k=None, threshold=None):
# #     """
# #     Retrieves the top chunks and filters out those whose scores are below the threshold (this is the first step in preventing hallucinations).
# #     """
# #     top_k = top_k or config.TOP_K
# #     threshold = config.SCORE_THRESHOLD if threshold is None else threshold

# #     vector = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
# #     result = client.query_points(
# #         collection_name=config.COLLECTION_NAME,
# #         query=vector.tolist(),
# #         limit=top_k,
# #         with_payload=True,
# #     )

# #     points = result.points
# #     top_score = points[0].score if points else 0.0
# #     good_hits = [p for p in points if p.score >= threshold]
# #     return good_hits, top_score

# import config
# from sentence_transformers import CrossEncoder

# _reranker = None


# def _get_reranker():
#     """Reranker ek hi baar load hota hai (pehli call par)."""
#     global _reranker
#     if _reranker is None:
#         _reranker = CrossEncoder(config.RERANK_MODEL)
#     return _reranker


# def retrieve(client, model, question, top_k=None, threshold=None, with_vectors=False):
#     """
#     1. Qdrant se CANDIDATES chunks lo
#     2. threshold se neeche wale hata do
#     3. reranker se re-order karke top_k rakho
#     """
#     top_k = top_k or config.TOP_K
#     threshold = config.SCORE_THRESHOLD if threshold is None else threshold

#     vector = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
#     result = client.query_points(
#         collection_name=config.COLLECTION_NAME,
#         query=vector.tolist(),
#         limit=config.CANDIDATES,
#         with_payload=True,
#         with_vectors=with_vectors,
#     )

#     points = result.points
#     top_score = points[0].score if points else 0.0   # MIN_BEST_SCORE check ke liye vector score hi rakha
#     good_hits = [p for p in points if p.score >= threshold]

#     if good_hits:
#         scores = _get_reranker().predict(
#             [(question, p.payload.get("text", "")) for p in good_hits]
#         )
#         ranked = sorted(zip(scores, good_hits), key=lambda x: -x[0])
#         good_hits = [p for _, p in ranked][:top_k]

#     return good_hits, top_score

# retrieval.py
# 1. get CANDIDATES chunks from Qdrant
# 2. drop chunks below SCORE_THRESHOLD
# 3. re-order the rest with a reranker and keep the best TOP_K

import config

_reranker = None


def _get_reranker():
    """The reranker is loaded only once (on the first call)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(getattr(config, "RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    return _reranker


def retrieve(client, model, question, top_k=None, threshold=None, with_vectors=False):
    """
    Returns (hits, top_score).
      hits      : the best chunks after threshold + reranking
      top_score : best vector score before filtering (used by enough_context)
    """
    top_k = top_k or config.TOP_K
    threshold = config.SCORE_THRESHOLD if threshold is None else threshold
    candidates = getattr(config, "CANDIDATES", 20)
    use_reranker = getattr(config, "USE_RERANKER", True)

    vector = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
    result = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector.tolist(),
        limit=max(candidates, top_k),
        with_payload=True,
        with_vectors=with_vectors,
    )

    points = result.points
    top_score = points[0].score if points else 0.0
    good_hits = [p for p in points if p.score >= threshold]

    if good_hits and use_reranker:
        scores = _get_reranker().predict(
            [(question, (p.payload or {}).get("text", "")) for p in good_hits]
        )
        ranked = sorted(zip(scores, good_hits), key=lambda pair: -pair[0])
        good_hits = [p for _, p in ranked]

    return good_hits[:top_k], top_score