from rapidfuzz import fuzz

_FUZZY_THRESHOLD = 60


def _fuzzy_score(a: str, b: str, partial: bool = False) -> float:
    """Returns a score in [-1.0, 1.0] based on fuzzy similarity."""
    ratio = fuzz.partial_ratio(a, b) if partial else fuzz.WRatio(a, b)
    if ratio >= _FUZZY_THRESHOLD:
        return (ratio / 100.0) + 1
    return -1.0


def match_condition(value: object, cond: object) -> float:  # noqa: C901
    """Returns a score in [-1.0, 1.0]: positive = match, negative = mismatch, 0 = unknown."""
    if value is None:
        return 0.0

    if not isinstance(cond, dict):
        if isinstance(value, str) and isinstance(cond, str):
            return _fuzzy_score(value.lower(), cond.lower())
        return 2.0 if value == cond else -1.0

    for op, target in cond.items():
        if op == "$eq":
            if isinstance(value, str) and isinstance(target, str):
                return _fuzzy_score(value.lower(), target.lower())
            return 2.0 if value == target else -1.0
        if op == "$ne":
            return 2.0 if value != target else -1.0
        if op == "$lt":
            return 2.0 if value < target else -1.0
        if op == "$lte":
            return 2.0 if value <= target else -1.0
        if op == "$gt":
            return 2.0 if value > target else -1.0
        if op == "$gte":
            return 2.0 if value >= target else -1.0
        if op == "$in":
            return 2.0 if value in target else -1.0
        if op == "$nin":
            return 2.0 if value not in target else -1.0
        if op in ("$like", "$ilike"):
            if not isinstance(value, str):
                return 0.0
            a = value.lower() if op == "$ilike" else value
            b = target.lower() if op == "$ilike" else target
            return _fuzzy_score(a, b, partial=True)

    return 0.0


def score_document(doc, filters: dict) -> float:
    """Normalized score in [-1, 1]. Missing metadata fields are neutral (0.0)."""
    if not filters:
        return 0.0

    total = sum(match_condition(doc.metadata.get(field), cond) for field, cond in filters.items())
    return total
