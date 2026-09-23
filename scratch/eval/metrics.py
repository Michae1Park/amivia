import math


def hit_rate_at_k(recommended: list[str], relevant: dict[str, int], k: int) -> float:
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def recall_at_k(recommended: list[str], relevant: dict[str, int], k: int) -> float | None:
    if not relevant:
        return None
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def precision_at_k(recommended: list[str], relevant: dict[str, int], k: int) -> float | None:
    if not relevant:
        return None
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def dcg_at_k(recommended: list[str], relevant: dict[str, int], k: int) -> float:
    return sum(
        relevant.get(item, 0) / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
    )


def ndcg_at_k(recommended: list[str], relevant: dict[str, int], k: int) -> float | None:
    if not relevant:
        return None
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    if idcg == 0:
        return None
    return dcg_at_k(recommended, relevant, k) / idcg


def mrr_at_k(recommended: list[str], relevant: dict[str, int], k: int) -> float | None:
    """1 / rank of the first relevant item in the top-k, or 0 if none appear.
    None (not 0) only when there's no relevant item at all to find — same
    convention as recall/precision/ndcg above, so a query with an empty
    relevant set doesn't silently drag the mean down."""
    if not relevant:
        return None
    for rank, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (rank + 1)
    return 0.0
