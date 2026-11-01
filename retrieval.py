 
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