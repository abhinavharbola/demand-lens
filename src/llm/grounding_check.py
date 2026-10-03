import re

from src.utils.config import CONFIG

NUMBER_PATTERN = re.compile(r"(?<![\w.])(\$?-?\d[\d,]*(?:\.\d+)?%?)(?![\w])")


def _normalize(token):
    kind = "number"
    t = token.strip()
    if t.startswith("$"):
        kind = "currency"
        t = t[1:]
    if t.endswith("%"):
        kind = "percent"
        t = t[:-1]
    t = t.replace(",", "")
    try:
        value = float(t)
    except ValueError:
        return None
    return value, kind


def _is_list_marker(text, match, raw):
    line_start = text.rfind("\n", 0, match.start()) + 1
    prefix = text[line_start:match.start()].strip(" #>*-")
    suffix = text[match.end():match.end() + 1]
    return not prefix and suffix in (".", ")") and "." not in raw


def extract_claims(text):
    claims = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(1).rstrip(",")
        if _is_list_marker(text, match, raw):
            continue
        parsed = _normalize(raw)
        if parsed is None:
            continue
        value, kind = parsed
        claims.append({"raw": raw, "value": value, "kind": kind, "span": match.span()})
    return claims


def _within_tolerance(claim_value, fact_value, relative_tolerance, absolute_tolerance):
    if fact_value == 0:
        return abs(claim_value) <= absolute_tolerance
    rel_ok = abs(claim_value - fact_value) / abs(fact_value) <= relative_tolerance
    abs_ok = abs(claim_value - fact_value) <= absolute_tolerance
    return rel_ok or abs_ok


def _candidate_values(kind, fact_items):
    for key, value in fact_items:
        is_currency = key.endswith("_usd")
        if kind == "currency":
            if is_currency:
                yield key, value
        elif is_currency:
            continue
        elif kind == "percent":
            yield key, value
            if abs(value) <= 1:
                yield key, value * 100
        else:
            yield key, value


def check_grounding(text, facts, config=CONFIG):
    gc_cfg = config["grounding_check"]
    rel_tol = gc_cfg["relative_tolerance"]
    currency_tol = gc_cfg["absolute_tolerance_currency"]
    default_tol = gc_cfg["absolute_tolerance_default"]

    fact_items = [
        (k, float(v)) for k, v in facts.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    claims = extract_claims(text)

    results = []
    for claim in claims:
        tol = currency_tol if claim["kind"] == "currency" else default_tol
        matches = [
            (key, candidate) for key, candidate in _candidate_values(claim["kind"], fact_items)
            if _within_tolerance(claim["value"], candidate, rel_tol, tol)
        ]
        matched_fact_key = None
        if matches:
            matched_fact_key = min(matches, key=lambda kc: abs(kc[1] - claim["value"]))[0]
        results.append({**claim, "grounded": bool(matches), "matched_fact_key": matched_fact_key})

    grounded_count = sum(1 for r in results if r["grounded"])
    total = len(results)
    return {
        "claims": results,
        "grounded_count": grounded_count,
        "total_claims": total,
        "grounded_ratio": grounded_count / total if total else 0.0,
        "verifiable": total > 0,
    }
