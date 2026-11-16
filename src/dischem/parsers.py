"""Deterministic extraction: everything that can be read from the record without a model.

Each function only reports values that are explicitly present. Cells containing [MASKED] are
treated as missing. Nothing here infers a unit, an axis or a value.
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from typing import Any

from . import schema

MASK = "[MASKED]"

_ROW = re.compile(r"<tr>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*</tr>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def parse_spec_table(spec_html: str) -> tuple[dict[str, str], list[str]]:
    """Return (unmasked key->value, list of masked keys) from the Specifications cell."""
    values: dict[str, str] = {}
    masked: list[str] = []
    for key, val in _ROW.findall(spec_html or ""):
        key = html.unescape(_TAG.sub("", key)).strip()
        val = html.unescape(_TAG.sub(" ", val))
        val = re.sub(r"\s+", " ", val).strip()
        if not key:
            continue
        if MASK in val or not val:
            if MASK in val:
                masked.append(key)
            continue
        values[key] = val
    return values, masked


_QTY = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(.*)$")


def split_quantity(text: str) -> tuple[str, str]:
    """'110 millilitre (ml)' -> ('110', 'millilitre (ml)'); '4,8g' -> ('4.8', 'g')."""
    m = _QTY.match(text or "")
    if not m:
        return "", ""
    value = m.group(1).replace(",", ".")
    unit = m.group(2).strip()
    if unit.count("(") > unit.count(")"):   # source truncation such as 'gram (g'
        unit += ")"
    return value, unit


def split_list(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\s*[,;]\s*", text or "") if p.strip()]


def snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


_QUANTITY_UNIT = re.compile(
    r"^(ml|millilit|l\b|lit|g\b|gram|kg|kilogram|mg|milligram|oz|caps|tablet|softgel|sachet|"
    r"piece|pcs|pack|wipe|unit|'?s\b|single|count|sheet|pad|bag)", re.I)

# Spec keys with a dedicated place in the output (lower-cased key -> handler name).
_PACK_KEYS = {"pack size", "size", "volume", "net weight", "net volume", "net content", "contents"}
_SHADE_KEYS = {"colour", "color", "shade", "colour/shade"}
_FLAVOUR_KEYS = {"flavour", "flavor"}
_FRAGRANCE_KEYS = {"fragrance", "scent", "fragrance notes"}
_SKIP_KEYS = {"brand", "exclusive brands", "unit of size", "unit of measure"}


def specs_to_fields(specs: dict[str, str]) -> dict[str, Any]:
    """Map an unmasked spec table to output fields, following the conventions in template.csv."""
    out = schema.empty_record()
    ps = out["product_specifications"]
    lower = {k.lower(): v for k, v in specs.items()}

    # 'Unit Of Size' + 'Unit Of Measure' is a second way the catalogue stores pack size
    if "unit of size" in lower and "unit of measure" in lower and "pack size" not in lower:
        ps["pack_size"] = {"value": lower["unit of size"].replace(",", "."), "unit": lower["unit of measure"]}

    for key, val in specs.items():
        k = key.lower().strip()
        if k in _SKIP_KEYS:
            continue
        if k in _PACK_KEYS:
            value, unit = split_quantity(val)
            # 'Size: 25cm' is a measurement, not a sold quantity: only mass, volume and counts
            # are pack sizes unless the row is literally called 'Pack Size'.
            is_quantity = k == "pack size" or bool(_QUANTITY_UNIT.match(unit))
            if value and is_quantity and not ps["pack_size"]["value"]:
                ps["pack_size"] = {"value": value, "unit": unit}
            elif not (value and is_quantity):
                ps["other"].append({"attribute": snake(key), "value": val})
        elif k == "weight":
            value, unit = split_quantity(val)
            if value:
                out["weight"] = {"value": value, "unit": unit}
        elif k in ("height", "width", "depth"):
            value, unit = split_quantity(val)
            if value:
                out["dimensions"][k] = value
                if unit and not out["dimensions"]["unit"]:
                    out["dimensions"]["unit"] = unit
        elif k == "application area":
            ps["application_area"] = split_list(val)
        elif k == "packaging type":
            ps["packaging_type"] = val
        elif k == "packaging material":
            ps["packaging_material"] = split_list(val)
        elif k in _SHADE_KEYS:
            out["variant_attributes"]["shade"] = split_list(val)
        elif k in _FLAVOUR_KEYS:
            out["variant_attributes"]["flavour"] = split_list(val)
        elif k in _FRAGRANCE_KEYS:
            out["variant_attributes"]["fragrance"] = split_list(val)
        elif k in ("certifications", "certification"):
            for item in split_list(val):
                ps["other"].append({"attribute": "certification", "value": item})
        elif k == "serving size":
            value, unit = split_quantity(val)
            if value:
                out["nutritional_information"]["serving_size"] = {"value": value, "unit": unit}
        else:
            ps["other"].append({"attribute": snake(key), "value": val})
    return schema.normalise(out)


def parse_nutrition(cell: str) -> tuple[dict[str, Any], int]:
    """Return (nutritional_information built from unmasked values, number of masked values)."""
    out = schema.empty_record()["nutritional_information"]
    masked = 0
    text = (cell or "").strip()
    if not text or text == "{}":
        return out, 0
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return out, 0
    for label, bases in (data.get("nutrients") or {}).items():
        if not isinstance(bases, dict):
            continue
        key = schema.nutrient_key(label)
        for basis in ("per_100g", "per_serving"):
            raw = bases.get(basis)
            if raw is None:
                continue
            if MASK in str(raw):
                masked += 1
                continue
            num = schema.to_number(raw)
            if num is not None and key:
                out[basis][key] = num
    serving = data.get("serving_size")
    if isinstance(serving, str) and MASK not in serving:
        value, unit = split_quantity(serving)
        if value:
            out["serving_size"] = {"value": value, "unit": unit}
    elif isinstance(serving, str):
        masked += 1
    return out, masked


_SECTION = re.compile(
    r"^(Key Features|Usage Instructions|Warnings|Storage|Ingredients|Specifications)\s*:\s*(.*)$", re.I)


def split_description(text: str) -> dict[str, str]:
    """Split a Product Description into its labelled sections ('body' holds the lead text)."""
    sections: dict[str, list[str]] = {"body": []}
    current = "body"
    for line in (text or "").splitlines():
        m = _SECTION.match(line.strip())
        if m:
            current = m.group(1).lower()
            sections.setdefault(current, [])
            if m.group(2).strip():
                sections[current].append(m.group(2).strip())
        elif not line.strip():
            current = "body"            # a blank line ends a labelled section
        else:
            sections[current].append(line.strip())
    return {k: "\n".join(x for x in v if x).strip() for k, v in sections.items() if any(v)}


def storage_from_description(text: str) -> list[str]:
    """Explicit 'Storage:' section only. Bullets become separate instructions."""
    block = split_description(text).get("storage", "")
    if not block:
        return []
    items = [re.sub(r"^[•\-*]\s*", "", ln).strip() for ln in block.splitlines()]
    return [i for i in items if i and MASK not in i]


def clean_name(name: str, brand: str = "") -> str:
    """Customer-facing name: the supplied name, with an ALL-CAPS brand token put in normal case."""
    name = re.sub(r"\s+", " ", name or "").strip()
    if not name:
        return ""
    brand = (brand or "").strip()
    if brand:
        b_tokens = brand.split()
        n_tokens = name.split()
        head = n_tokens[: len(b_tokens)]
        if (len(head) == len(b_tokens) and " ".join(head).isupper() and len(" ".join(head)) > 2
                and _fold(" ".join(head)) == _fold(brand)):
            cased = [t if any(c.islower() for c in t) else t.capitalize() for t in b_tokens]
            cased = [re.sub(r"'S\b", "'s", t) for t in cased]
            name = " ".join(cased + n_tokens[len(b_tokens):])
    return name


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


_NAME_SIZE = re.compile(
    r"(?:(\d+)\s*[xX]\s*)?(\d+(?:[.,]\d+)?)\s?(ml|l|kg|g|mg|cm|mm|m)\b", re.I)


def size_tokens(name: str) -> list[str]:
    """Size expressions found in a product name, e.g. ['110ml'] or ['6 x 25ml']. For prompts only."""
    out = []
    for m in _NAME_SIZE.finditer(name or ""):
        qty = m.group(2).replace(",", ".") + m.group(3)
        out.append(f"{m.group(1)} x {qty}" if m.group(1) else qty)
    return out


def deterministic_fields(rec: dict[str, Any]) -> dict[str, Any]:
    """All model-free facts for a record, already in output shape."""
    specs, masked_keys = parse_spec_table(rec.get("Specifications", ""))
    out = specs_to_fields(specs)
    nutrition, masked_n = parse_nutrition(rec.get("Nutritional Information", ""))
    if nutrition["per_100g"] or nutrition["per_serving"]:
        if out["nutritional_information"]["serving_size"]["value"] and not nutrition["serving_size"]["value"]:
            nutrition["serving_size"] = out["nutritional_information"]["serving_size"]
        out["nutritional_information"] = nutrition
    elif nutrition["serving_size"]["value"]:
        out["nutritional_information"]["serving_size"] = nutrition["serving_size"]
    out["storage_instructions"] = storage_from_description(rec.get("Product Description", ""))
    out["online_name"] = clean_name(rec.get("Product Name", ""), rec.get("T_BRAND", ""))
    out = schema.normalise(out)
    out["_meta"] = {"spec_keys": sorted(specs), "masked_spec_keys": masked_keys,
                    "masked_nutrition_values": masked_n}
    return out
