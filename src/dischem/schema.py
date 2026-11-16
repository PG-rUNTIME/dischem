"""Output schema: the ten Phase 1 fields, their empty values, normalisation and serialisation.

The shapes mirror data/submission_format.json. A blank CSV cell is read by the evaluator as the
field's empty value, so an "empty" field is always written as a blank cell, never as N/A.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

SUBMISSION_COLUMNS = [
    "article_id", "online_name", "short_description", "long_description",
    "ingredient_information", "storage_instructions", "variant_attributes",
    "product_specifications", "weight", "dimensions", "nutritional_information",
]
TEXT_FIELDS = ["online_name", "short_description", "long_description"]
STRUCT_FIELDS = [c for c in SUBMISSION_COLUMNS[1:] if c not in TEXT_FIELDS]
OUTPUT_FIELDS = TEXT_FIELDS + STRUCT_FIELDS

EMPTY: dict[str, Any] = {
    "online_name": "",
    "short_description": "",
    "long_description": "",
    "ingredient_information": [],
    "storage_instructions": [],
    "variant_attributes": {"shade": [], "flavour": [], "fragrance": [], "other": []},
    "product_specifications": {
        "pack_size": {"value": "", "unit": ""},
        "application_area": [],
        "packaging_type": "",
        "packaging_material": [],
        "other": [],
    },
    "weight": {"value": "", "unit": ""},
    "dimensions": {"height": "", "width": "", "depth": "", "unit": ""},
    "nutritional_information": {
        "serving_size": {"value": "", "unit": ""},
        "per_100g": {},
        "per_serving": {},
    },
}

# Phrases that must never be written as a value (guardrail section 5).
_PLACEHOLDERS = {
    "", "n/a", "na", "n.a.", "none", "null", "nil", "unknown", "not available", "not applicable",
    "not specified", "not stated", "not visible", "not provided", "unspecified", "-", "--", "tbc",
    "[masked]", "masked",
}


def empty_record() -> dict[str, Any]:
    return copy.deepcopy(EMPTY)


def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().strip(".") in _PLACEHOLDERS


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = format_number(value)
    text = re.sub(r"\s+", " ", str(value)).strip()
    if "[masked]" in text.lower():
        return ""
    return "" if is_placeholder(text) else text


def format_number(x: float | int) -> str:
    if isinstance(x, int) or float(x).is_integer():
        return str(int(x))
    return repr(float(x))


def clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out, seen = [], set()
    for v in values:
        s = clean_str(v)
        key = s.lower().rstrip(".")
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def to_number(value: Any) -> float | int | str | None:
    """Nutrition values are numeric in the reference; keep qualifiers such as '<0.1' as text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    s = clean_str(value).replace(",", ".")
    if not s:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        f = float(s)
        return int(f) if f.is_integer() and "." not in s else f
    return s


def nutrient_key(label: str) -> str:
    """'Of which saturated fat (g)' -> 'saturated_fat_g' (convention used in template.csv)."""
    s = label.strip().lower()
    s = re.sub(r"^of which\s+", "", s)
    s = re.sub(r"^total sodium", "sodium", s)
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def normalise(rec: dict[str, Any]) -> dict[str, Any]:
    """Coerce any dict into the exact schema shape, dropping placeholders and unknown keys."""
    out = empty_record()
    for f in TEXT_FIELDS:
        out[f] = clean_str(rec.get(f))
    out["ingredient_information"] = clean_list(rec.get("ingredient_information"))
    out["storage_instructions"] = clean_list(rec.get("storage_instructions"))

    va = rec.get("variant_attributes") or {}
    if isinstance(va, dict):
        for k in out["variant_attributes"]:
            out["variant_attributes"][k] = clean_list(va.get(k))

    ps = rec.get("product_specifications") or {}
    if isinstance(ps, dict):
        size = ps.get("pack_size") or {}
        if isinstance(size, dict):
            out["product_specifications"]["pack_size"] = {
                "value": clean_str(size.get("value")), "unit": clean_str(size.get("unit"))}
        out["product_specifications"]["application_area"] = clean_list(ps.get("application_area"))
        out["product_specifications"]["packaging_type"] = clean_str(ps.get("packaging_type"))
        out["product_specifications"]["packaging_material"] = clean_list(ps.get("packaging_material"))
        other, seen = [], set()
        for item in ps.get("other") or []:
            if not isinstance(item, dict):
                continue
            a, v = clean_str(item.get("attribute")), clean_str(item.get("value"))
            a = re.sub(r"[^a-z0-9]+", "_", a.lower()).strip("_")
            if a and v and (a, v.lower()) not in seen:
                seen.add((a, v.lower()))
                other.append({"attribute": a, "value": v})
        out["product_specifications"]["other"] = other
    # a unit on its own is not a pack size
    if not out["product_specifications"]["pack_size"]["value"]:
        out["product_specifications"]["pack_size"]["unit"] = ""

    w = rec.get("weight") or {}
    if isinstance(w, dict):
        out["weight"] = {"value": clean_str(w.get("value")), "unit": clean_str(w.get("unit"))}
    if not out["weight"]["value"]:      # guardrail: a unit by itself is not a product weight
        out["weight"]["unit"] = ""

    d = rec.get("dimensions") or {}
    if isinstance(d, dict):
        for k in out["dimensions"]:
            out["dimensions"][k] = clean_str(d.get(k))
    if not any(out["dimensions"][k] for k in ("height", "width", "depth")):
        out["dimensions"]["unit"] = ""

    n = rec.get("nutritional_information") or {}
    if isinstance(n, dict):
        ss = n.get("serving_size") or {}
        if isinstance(ss, dict):
            out["nutritional_information"]["serving_size"] = {
                "value": clean_str(ss.get("value")), "unit": clean_str(ss.get("unit"))}
        for basis in ("per_100g", "per_serving"):
            src = n.get(basis) or {}
            if isinstance(src, list):   # LLM form: [{"nutrient": ..., "value": ...}]
                src = {i.get("nutrient"): i.get("value") for i in src if isinstance(i, dict)}
            vals = {}
            for k, v in src.items():
                num = to_number(v)
                key = nutrient_key(str(k or ""))
                if key and num is not None:
                    vals[key] = num
            out["nutritional_information"][basis] = vals
        if not out["nutritional_information"]["serving_size"]["value"]:
            out["nutritional_information"]["serving_size"]["unit"] = ""
    return out


def is_empty(field: str, value: Any) -> bool:
    return value == EMPTY[field]


def to_cell(field: str, value: Any) -> str:
    """CSV cell for a field: plain text for text fields, compact JSON otherwise, blank if empty."""
    if is_empty(field, value):
        return ""
    if field in TEXT_FIELDS:
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def from_cell(field: str, cell: str) -> Any:
    if field in TEXT_FIELDS:
        return cell or ""
    if not (cell or "").strip():
        return copy.deepcopy(EMPTY[field])
    return json.loads(cell)


def leaves(field: str, value: Any) -> list[tuple[str, str]]:
    """Flatten a structured field to (path, value) pairs — the unit the F1 metric counts."""
    out: list[tuple[str, str]] = []

    def walk(path: str, v: Any) -> None:
        if isinstance(v, dict):
            if set(v) == {"attribute", "value"}:
                out.append((f"{path}.{v['attribute']}", str(v["value"])))
                return
            for k, sub in v.items():
                walk(f"{path}.{k}" if path else k, sub)
        elif isinstance(v, list):
            for sub in v:
                walk(path, sub)
        elif v not in ("", None):
            out.append((path, str(v)))

    walk("", value)
    return out
