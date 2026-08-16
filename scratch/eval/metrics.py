import math


def hit_rate_at_k(recommended: list[str], relevant: dict[str, int], k: int) -> float:
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def recall_at_k(recommended: list[str], relevant: dict[str, int], k: int) -> float | None:
    if not relevant:
        return None
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


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
