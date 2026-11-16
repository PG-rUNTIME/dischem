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


# ---------------------------------------------------------------------------------------------
# Post-processing rules (applied in finalise, so they can be re-applied to saved records).
# ---------------------------------------------------------------------------------------------

PACKAGING_TYPES = ["Bottle", "Jar", "Tube", "Box", "Pouch", "Tin", "Can", "Bag",
                   "Blister pack", "Sachet", "Tub", "Carton"]
_PACKAGING_LOOKUP = {p.lower(): p for p in PACKAGING_TYPES}

SPEC_OTHER_ALLOWED = ["certification", "recyclable_packaging", "fragrance_type", "spf",
                      "fabric_composition", "colour", "model", "age_range"]
_SPEC_OTHER_ALIASES = {"certifications": "certification", "color": "colour"}

_CLAIM_RE = re.compile(r"\bvegan\b|\bvegetarian\b|\b[\w]+[\s-]free\b", re.I)
_STORAGE_KEYS = ("store", "storage", "keep", "refrigerat", "temperature", "dry", "cool",
                 "sunlight", "reseal")
_STORAGE_WARNING_RE = re.compile(r"out of (the )?reach|warning|caution", re.I)
_ING_CLAIM_RE = re.compile(r"%\s*natural|natural[\s-]*(plant\s+)?origin|origin\s+ingredients|"
                           r"free from|suitable for", re.I)
_NON_ENGLISH_RE = re.compile(
    r"[éèêëàâçôûùîïüöäßñœ]|\b(et|de|du|des|la|le|les|avec|pour|der|die|das|und|van|met|vir|"
    r"en|ein|eine|mit|für)\b|\bd'", re.I)
_META_RE = re.compile(r"photo evidence|printed text|in the source|not provided|as shown", re.I)

_WEIGHT_UNITS = {
    "g": "g", "gram": "g", "grams": "g", "gram (g)": "g", "gm": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "kilogram (kg)": "kg", "kgs": "kg",
    "ml": "ml", "millilitre": "ml", "milliliter": "ml", "millilitres": "ml",
    "milliliters": "ml", "millilitre (ml)": "ml",
}


def _fold(text: str) -> str:
    return _norm(parsers._fold(str(text)))


def _spec_text(rec: dict[str, Any]) -> str:
    return _fold(re.sub(r"<[^>]+>", " ", rec.get("Specifications", "") or ""))


def _tokens(text: str) -> list[str]:
    return [t for t in _fold(text).split() if t]


def fix_shade(rec: dict[str, Any], out: dict[str, Any]) -> None:
    """Keep a shade only if the name or spec table says it; cap at the colours the name mentions."""
    shades = out["variant_attributes"]["shade"]
    if not shades:
        return
    name_tokens = set(_tokens(rec.get("Product Name", "")) + _tokens(out["online_name"]))
    spec = " " + _spec_text(rec) + " "
    in_name, in_spec = [], []
    for s in shades:
        toks = _tokens(s)
        if not toks:
            continue
        if all(t in name_tokens for t in toks):
            in_name.append(s)
        elif f" {' '.join(toks)} " in spec:
            in_spec.append(s)
    out["variant_attributes"]["shade"] = in_name or in_spec


def fix_packaging_type(out: dict[str, Any]) -> None:
    ps = out["product_specifications"]
    ps["packaging_type"] = _PACKAGING_LOOKUP.get(ps["packaging_type"].strip().lower(), "")


def fix_spec_other(out: dict[str, Any]) -> None:
    keep, seen = [], set()
    for o in out["product_specifications"]["other"]:
        attr = _SPEC_OTHER_ALIASES.get(o["attribute"].lower(), o["attribute"].lower())
        key = (attr, _norm(o["value"]))
        if attr in SPEC_OTHER_ALLOWED and key not in seen:
            seen.add(key)
            keep.append({"attribute": attr, "value": o["value"]})
    out["product_specifications"]["other"] = keep


def fix_variant_other(out: dict[str, Any]) -> None:
    """Claim words in variant 'other' are certifications, not variants."""
    other, moved = [], []
    for v in out["variant_attributes"]["other"]:
        if len(v.split()) <= 4 and _CLAIM_RE.search(v):
            moved.append(v)
        else:
            other.append(v)
    out["variant_attributes"]["other"] = other
    have = {_norm(o["value"]) for o in out["product_specifications"]["other"]
            if o["attribute"] == "certification"}
    for v in moved:
        if _norm(v) not in have:
            have.add(_norm(v))
            out["product_specifications"]["other"].append({"attribute": "certification", "value": v})


def fix_storage(out: dict[str, Any]) -> None:
    out["storage_instructions"] = [
        s for s in out["storage_instructions"]
        if any(k in s.lower() for k in _STORAGE_KEYS) and not _STORAGE_WARNING_RE.search(s)]


def fix_ingredients(out: dict[str, Any]) -> None:
    out["ingredient_information"] = [
        s for s in out["ingredient_information"]
        if not _ING_CLAIM_RE.search(s) and not _NON_ENGLISH_RE.search(s)]


def fix_weight(out: dict[str, Any]) -> None:
    """Standardise the unit spelling. A weight without a unit is kept: on the validation set 125 of
    158 reference weights have an empty unit, and parsed unit-less weights were right 19 times in
    19, so blanking them only discarded correct values."""
    w = out["weight"]
    unit = _WEIGHT_UNITS.get(w["unit"].strip().lower())
    if unit:
        w["unit"] = unit


def strip_meta_sentences(out: dict[str, Any]) -> None:
    for f in ("short_description", "long_description"):
        sentences = re.split(r"(?<=[.!?])\s+", out[f])
        out[f] = " ".join(s for s in sentences if not _META_RE.search(s)).strip()


def fix_brand_case(rec: dict[str, Any], out: dict[str, Any]) -> None:
    """Short acronym brands (NPL, USN, NYX, OGX) are written in capitals.

    Applied to brands of up to three letters, and to four-letter brands with no vowel. Four-letter
    brands that read as words (Dove, Nuby, Joie, Avon) keep normal case.
    """
    brand = (rec.get("T_BRAND") or "").strip()
    letters = re.sub(r"[^A-Za-z]", "", brand)
    if not brand or " " in brand or not letters.isalpha() or len(letters) > 4:
        return
    if len(letters) == 4 and re.search(r"[aeiou]", letters, re.I):
        return
    pat = re.compile(rf"(?<![\w]){re.escape(brand)}(?![\w])", re.I)
    for f in schema.TEXT_FIELDS:
        out[f] = pat.sub(brand.upper(), out[f])


def postprocess(rec: dict[str, Any], out: dict[str, Any]) -> None:
    fix_shade(rec, out)
    fix_packaging_type(out)
    fix_variant_other(out)           # may add certifications ...
    fix_spec_other(out)              # ... so the allow-list runs after it
    fix_storage(out)
    fix_ingredients(out)
    fix_weight(out)
    strip_meta_sentences(out)


def finalise(rec: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """Last step for every record: schema shape, post-processing, do-not-predict blanking, text fallbacks."""
    out = schema.normalise(out)
    postprocess(rec, out)
    fallback_text(rec, out)
    fix_brand_case(rec, out)
    if _norm(out["long_description"]) == _norm(out["short_description"]) and rec.get("Product Description"):
        sections = parsers.split_description(rec["Product Description"])
        alt = " ".join(x for x in (sections.get("body", ""), _bullets_to_sentence(sections.get("key features", ""))) if x)
        if alt and _norm(alt) != _norm(out["short_description"]):
            out["long_description"] = schema.clean_str(alt)
    for f in rec.get("_dnp", []):
        if f in schema.OUTPUT_FIELDS:
            out[f] = schema.empty_record()[f]
    return schema.normalise(out)
