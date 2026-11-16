"""Prompts and JSON schemas for the three model stages.

The rules are condensed from data/product_enrichment_guardrail.pdf and the Data page. The schemas
use only fixed keys, strings and arrays so that constrained decoding is reliable on any backend.
"""
from __future__ import annotations

import json
from typing import Any

from . import schema

S = {"type": "string"}
SL = {"type": "array", "items": S}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


_NUTRIENT_LIST = {"type": "array", "items": _obj({"nutrient": S, "value": S})}

# ---------------------------------------------------------------- stage A: image evidence
EVIDENCE_SCHEMA = _obj({
    "matches_product": {"type": "string", "enum": ["yes", "no", "unsure"]},
    "printed_text": SL,
    "product_form": S,
    "packaging_type": S,
    "packaging_materials": SL,
    "colours": SL,
    "pack_size": S,
    "claims": SL,
    "ingredients": SL,
    "directions": SL,
    "storage_instructions": SL,
    "certifications": SL,
    "nutrition_panel": S,
})

EVIDENCE_SYSTEM = """You read retail product photos and report only what is visibly there.
You are transcribing evidence for a product catalogue. Rules:
- Report printed text exactly as printed. Do not translate, complete or correct it.
- Never use outside knowledge about the brand or product. If something is not visible, leave the
  field as an empty string or empty list.
- matches_product: "yes" if the brand and product on the pack agree with the product name given,
  "no" if the photo clearly shows a different product or variant, otherwise "unsure".
- packaging_type: the visible container only (for example Bottle, Jar, Tube, Box, Pouch, Tin,
  Blister pack). Leave empty if the product is shown unpackaged.
- packaging_materials: only if printed on the pack or unmistakable (for example a glass jar).
- colours: the colour or shade of the product itself, not of the background.
- pack_size: net quantity or count as printed, for example "250 g", "30 pcs", "6 x 25 ml".
- claims: short factual claims printed on the pack, each as printed.
- ingredients, directions, storage_instructions, certifications: only if printed and legible.
- nutrition_panel: transcribe the table row by row with its column headings if one is legible,
  otherwise an empty string.
Return JSON only."""


def evidence_user(rec: dict[str, Any], n_images: int) -> str:
    return (f"Product name: {rec['Product Name']}\nBrand: {rec.get('T_BRAND', '')}\n"
            f"Category: {rec.get('category_group', '')} / {rec.get('T_CATEGORY', '')}\n"
            f"There {'is 1 photo' if n_images == 1 else f'are {n_images} photos'} of this record. "
            "Report what is visible across all of them.")


# ---------------------------------------------------------------- stage B: composition
COMPOSE_SCHEMA = _obj({
    "short_description": S,
    "long_description": S,
    "ingredient_information": SL,
    "storage_instructions": SL,
    "variant_attributes": _obj({"shade": SL, "flavour": SL, "fragrance": SL, "other": SL}),
    "product_specifications": _obj({
        "pack_size": _obj({"value": S, "unit": S}),
        "application_area": SL,
        "packaging_type": S,
        "packaging_material": SL,
        "other": {"type": "array", "items": _obj({"attribute": S, "value": S})},
    }),
    "weight": _obj({"value": S, "unit": S}),
    "dimensions": _obj({"height": S, "width": S, "depth": S, "unit": S}),
    "nutritional_information": _obj({
        "serving_size": _obj({"value": S, "unit": S}),
        "per_100g": _NUTRIENT_LIST,
        "per_serving": _NUTRIENT_LIST,
    }),
})

COMPOSE_SYSTEM = """You enrich pharmacy and health retail product records for Dis-Chem.
You are given one product record (some source fields are blank or marked [MASKED]), facts already
parsed from it, and sometimes evidence transcribed from product photos. Produce the output fields
as JSON.

Evidence rules — these decide the score:
- Use ONLY the record, the parsed facts and the photo evidence. Never use outside knowledge about
  the brand, the product or similar products. An accurate omission is better than a plausible
  invention.
- Every value must apply to this exact product and variant: check pack size, flavour, shade,
  weight and units.
- If a detail is not supported, leave it empty ("" or []). Never write N/A, none, unknown or
  [MASKED].
- Keep the source's qualifiers and strength ("may help" stays "may help"). Do not add safety,
  efficacy, suitability or free-from claims.
- Ignore photo evidence marked as not matching the product.

Field rules:
- short_description: one or two sentences saying what the product is or does. No bullets. If the
  record supplies a short description, stay close to its meaning. If there is no description and
  no photo evidence, write one plain sentence from the product name and category only.
- long_description: start with the full product name and what it is or is for, then the supported
  details (key features, how to use). Do not open with the same sentence as the short description.
  UK English. No invented claims.
- ingredient_information: ingredient names in source order and wording, including named active or
  key ingredients and any "may contain" statement. Empty if none are stated.
- storage_instructions: storage instructions exactly as supported by the sources.
- variant_attributes: shade (colour or shade), flavour, fragrance, other (other variant
  identifiers such as a range or theme name). Each is a list of strings.
- product_specifications.pack_size: the sold quantity, value and unit separately, for example
  {"value": "110", "unit": "ml"}; application_area: where it is applied (Face, Hair, Body...);
  packaging_type: Bottle, Jar, Tube, Box, Pouch...; packaging_material: only if stated or
  unmistakable in the photo evidence; other: further stated attributes as
  {"attribute": "snake_case_name", "value": "..."} (for example certification, fragrance_type,
  product_type, colour, material, spf).
- weight: only an explicit product weight with its value. A weight unit alone is not a weight.
- dimensions: only explicit height, width and depth. Do not relabel length as depth. The VOLUME
  column is not a dimension.
- nutritional_information: only values present in the sources, on their printed basis. Do not
  convert between per 100 g and per serving. Each entry is {"nutrient": name, "value": number}.
- Fields listed under "Do not fill" must be left empty.
Return JSON only."""

_SOURCE_KEYS = [
    ("Product name", "Product Name"), ("Brand", "T_BRAND"), ("Category group", "category_group"),
    ("Department", "T_DEPARTMENT"), ("Category", "T_CATEGORY"), ("Sub-category", "T_BMC"),
    ("Article type", "T_ARTICLE_CATEGORY"),
]


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def record_block(rec: dict[str, Any], facts: dict[str, Any], evidence: dict[str, Any] | None,
                 max_chars: int, siblings: dict[str, str] | None = None) -> str:
    lines = [f"{label}: {rec.get(col, '')}" for label, col in _SOURCE_KEYS if rec.get(col)]
    lines.append("Short description (source): " + (_clip(rec.get("Short Description", ""), 900) or "(blank)"))
    lines.append("Product description (source): " + (_clip(rec.get("Product Description", ""), max_chars) or "(blank)"))
    if rec.get("Specifications"):
        lines.append("Specifications (source): " + _clip(rec["Specifications"], 1500))
    nut = rec.get("Nutritional Information", "")
    if nut and nut != "{}":
        lines.append("Nutritional information (source): " + _clip(nut, 2500))
    parsed = {k: v for k, v in facts.items()
              if not k.startswith("_") and k not in schema.TEXT_FIELDS and not schema.is_empty(k, v)}
    if parsed:
        lines.append("Parsed facts (already verified, keep them): " + json.dumps(parsed, ensure_ascii=False))
    masked = (facts.get("_meta") or {}).get("masked_spec_keys") or []
    if masked:
        lines.append("Masked specification rows (fill only if another source supports them): " + ", ".join(masked))
    if evidence:
        ev = {k: v for k, v in evidence.items() if v not in ("", [], None) and k != "matches_product"}
        lines.append(f"Photo evidence (product match: {evidence.get('matches_product', 'unsure')}): "
                     + json.dumps(ev, ensure_ascii=False))
    else:
        lines.append("Photo evidence: none")
    if siblings:
        lines.append("Catalogue lookup (same brand, line and pack size; values that all matching "
                     "products share): " + json.dumps(siblings, ensure_ascii=False))
    dnp = [f for f in rec.get("_dnp", []) if f in schema.OUTPUT_FIELDS]
    if dnp:
        lines.append("Do not fill: " + ", ".join(dnp))
    return "\n".join(lines)


def to_llm_shape(out: dict[str, Any]) -> dict[str, Any]:
    """Convert a normalised output record to the composition schema (nutrients as a list)."""
    shaped = {k: out[k] for k in COMPOSE_SCHEMA["properties"]}
    nut = dict(shaped["nutritional_information"])
    for basis in ("per_100g", "per_serving"):
        nut[basis] = [{"nutrient": k, "value": str(v)} for k, v in nut[basis].items()]
    shaped["nutritional_information"] = nut
    return shaped


def compose_user(rec: dict[str, Any], facts: dict[str, Any], evidence: dict[str, Any] | None,
                 examples: list[dict[str, Any]], max_chars: int,
                 siblings: dict[str, str] | None = None) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"### Example {i}\n" + record_block(ex["input"], ex["facts"], None, 1200)
                     + "\nOutput: " + json.dumps(to_llm_shape(ex["output"]), ensure_ascii=False))
    parts.append("### Product to enrich\n" + record_block(rec, facts, evidence, max_chars, siblings)
                 + "\nOutput:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- stage C: verification
VERIFY_SCHEMA = _obj({
    "unsupported": {"type": "array", "items": _obj({
        "field": {"type": "string", "enum": schema.STRUCT_FIELDS},
        "value": S,
        "reason": S,
    })},
})

VERIFY_SYSTEM = """You are a strict reviewer of product catalogue data.
You are given the sources for one product and a list of proposed values. List every proposed
value that is NOT supported by the sources, or that belongs to a different product or variant, or
whose number or unit differs from the sources. Rules:
- Supported means the sources state it, or it is a faithful restatement (for example "236ml" in
  the product name supports a pack size of 236 ml).
- Outside knowledge does not count as support.
- Do not list values that are supported. Do not suggest new values.
- Copy the unsupported value exactly as proposed.
Return JSON only: {"unsupported": [{"field": ..., "value": ..., "reason": ...}]}"""


def verify_user(rec: dict[str, Any], facts: dict[str, Any], evidence: dict[str, Any] | None,
                proposed: list[tuple[str, str, str]], max_chars: int,
                siblings: dict[str, str] | None = None) -> str:
    listing = "\n".join(f"- {field} | {path} | {value}" for field, path, value in proposed)
    return ("### Sources\n" + record_block(rec, facts, evidence, max_chars, siblings)
            + "\n\n### Proposed values (field | path | value)\n" + listing)
