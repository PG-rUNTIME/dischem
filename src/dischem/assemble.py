"""Merging model output with parsed facts, and the final guardrail checks."""
from __future__ import annotations

import re
from typing import Any

from . import parsers, schema


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def merge(facts: dict[str, Any], llm: dict[str, Any] | None) -> dict[str, Any]:
    """Parsed facts win leaf by leaf; the model fills what is still empty; lists are united."""
    base = schema.normalise({k: v for k, v in facts.items() if not k.startswith("_")})
    if not llm:
        return base
    add = schema.normalise(llm)
    out = base

    for f in ("short_description", "long_description"):
        out[f] = add[f] or base[f]
    out["online_name"] = base["online_name"] or add["online_name"]
    for f in ("ingredient_information", "storage_instructions"):
        out[f] = schema.clean_list(base[f] + add[f])
    for k in out["variant_attributes"]:
        out["variant_attributes"][k] = schema.clean_list(
            base["variant_attributes"][k] + add["variant_attributes"][k])

    ps, aps = out["product_specifications"], add["product_specifications"]
    if not ps["pack_size"]["value"]:
        ps["pack_size"] = aps["pack_size"]
    ps["application_area"] = schema.clean_list(ps["application_area"] + aps["application_area"])
    ps["packaging_type"] = ps["packaging_type"] or aps["packaging_type"]
    ps["packaging_material"] = schema.clean_list(ps["packaging_material"] + aps["packaging_material"])
    have = {(o["attribute"], _norm(o["value"])) for o in ps["other"]}
    have_vals = {_norm(o["value"]) for o in ps["other"]}
    for o in aps["other"]:
        if (o["attribute"], _norm(o["value"])) not in have and _norm(o["value"]) not in have_vals:
            ps["other"].append(o)

    if not out["weight"]["value"]:
        out["weight"] = add["weight"]
    for k in ("height", "width", "depth"):
        out["dimensions"][k] = out["dimensions"][k] or add["dimensions"][k]
    out["dimensions"]["unit"] = out["dimensions"]["unit"] or add["dimensions"]["unit"]

    nut, anut = out["nutritional_information"], add["nutritional_information"]
    if not nut["serving_size"]["value"]:
        nut["serving_size"] = anut["serving_size"]
    for basis in ("per_100g", "per_serving"):
        for k, v in anut[basis].items():
            nut[basis].setdefault(k, v)
    return schema.normalise(out)


def proposed_values(out: dict[str, Any], facts: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Structured values that came from the model (not from parsing) — what the verifier checks."""
    base = schema.normalise({k: v for k, v in facts.items() if not k.startswith("_")})
    rows = []
    for f in schema.STRUCT_FIELDS:
        known = {(p, _norm(v)) for p, v in schema.leaves(f, base[f])}
        for path, value in schema.leaves(f, out[f]):
            if (path, _norm(value)) not in known:
                rows.append((f, path, value))
    return rows


def remove_unsupported(out: dict[str, Any], facts: dict[str, Any],
                       unsupported: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Delete model-proposed values the verifier rejected. Parsed facts are never removed."""
    if not unsupported:
        return out, 0
    base = schema.normalise({k: v for k, v in facts.items() if not k.startswith("_")})
    protected = {f: {_norm(v) for _, v in schema.leaves(f, base[f])} for f in schema.STRUCT_FIELDS}
    drop: dict[str, set[str]] = {}
    for item in unsupported:
        f, v = item.get("field"), _norm(item.get("value", ""))
        if f in schema.STRUCT_FIELDS and v and v not in protected[f]:
            drop.setdefault(f, set()).add(v)
    removed = 0

    def keep(field: str, value: Any) -> bool:
        nonlocal removed
        if _norm(value) in drop.get(field, ()):
            removed += 1
            return False
        return True

    for f in ("ingredient_information", "storage_instructions"):
        out[f] = [v for v in out[f] if keep(f, v)]
    for k, vals in out["variant_attributes"].items():
        out["variant_attributes"][k] = [v for v in vals if keep("variant_attributes", v)]
    ps = out["product_specifications"]
    if ps["pack_size"]["value"] and not keep("product_specifications", ps["pack_size"]["value"]):
        ps["pack_size"] = {"value": "", "unit": ""}
    ps["application_area"] = [v for v in ps["application_area"] if keep("product_specifications", v)]
    if ps["packaging_type"] and not keep("product_specifications", ps["packaging_type"]):
        ps["packaging_type"] = ""
    ps["packaging_material"] = [v for v in ps["packaging_material"] if keep("product_specifications", v)]
    ps["other"] = [o for o in ps["other"] if keep("product_specifications", o["value"])]
    if out["weight"]["value"] and not keep("weight", out["weight"]["value"]):
        out["weight"] = {"value": "", "unit": ""}
    for k in ("height", "width", "depth"):
        if out["dimensions"][k] and not keep("dimensions", out["dimensions"][k]):
            out["dimensions"][k] = ""
    nut = out["nutritional_information"]
    if nut["serving_size"]["value"] and not keep("nutritional_information", nut["serving_size"]["value"]):
        nut["serving_size"] = {"value": "", "unit": ""}
    for basis in ("per_100g", "per_serving"):
        nut[basis] = {k: v for k, v in nut[basis].items() if keep("nutritional_information", v)}
    return schema.normalise(out), removed


def fallback_text(rec: dict[str, Any], out: dict[str, Any]) -> None:
    """Text fields are never left empty: a blank scores 0 and there is no penalty for trying.

    The fallback only restates what the record itself says: the supplied description when there
    is one, otherwise the product name and its catalogue category.
    """
    name = out["online_name"] or parsers.clean_name(rec.get("Product Name", ""), rec.get("T_BRAND", ""))
    out["online_name"] = name
    sections = parsers.split_description(rec.get("Product Description", ""))
    if not out["short_description"]:
        out["short_description"] = rec.get("Short Description", "") or _name_sentence(rec, name)
    if not out["long_description"]:
        body = sections.get("body", "")
        feats = sections.get("key features", "")
        text = " ".join(x for x in (body, _bullets_to_sentence(feats)) if x)
        out["long_description"] = text or _name_sentence(rec, name)
    for f in ("short_description", "long_description"):
        out[f] = schema.clean_str(out[f])


def _bullets_to_sentence(block: str) -> str:
    items = [re.sub(r"^[•\-*]\s*", "", ln).strip().rstrip(".") for ln in block.splitlines()]
    items = [i for i in items if i]
    return ("Key features: " + "; ".join(items) + ".") if items else ""


def _name_sentence(rec: dict[str, Any], name: str) -> str:
    category = (rec.get("T_CATEGORY") or rec.get("category_group") or "").strip().lower()
    brand = (rec.get("T_BRAND") or "").strip()
    if category and brand:
        return f"{name} is a {category} product from {brand}."
    if category:
        return f"{name} is a {category} product."
    return f"{name}."


def finalise(rec: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """Last step for every record: schema shape, do-not-predict blanking, text fallbacks."""
    out = schema.normalise(out)
    fallback_text(rec, out)
    if _norm(out["long_description"]) == _norm(out["short_description"]) and rec.get("Product Description"):
        sections = parsers.split_description(rec["Product Description"])
        alt = " ".join(x for x in (sections.get("body", ""), _bullets_to_sentence(sections.get("key features", ""))) if x)
        if alt and _norm(alt) != _norm(out["short_description"]):
            out["long_description"] = schema.clean_str(alt)
    for f in rec.get("_dnp", []):
        if f in schema.OUTPUT_FIELDS:
            out[f] = schema.empty_record()[f]
    return schema.normalise(out)
