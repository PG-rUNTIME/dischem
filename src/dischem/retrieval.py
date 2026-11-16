"""Few-shot example selection and the optional catalogue (sibling) lookup."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from . import parsers, schema


def _tokens(text: str) -> set[str]:
    # digits and letters are separate tokens so that "4.8" matches "4,8g" and "236" matches "236ml"
    return set(re.findall(r"\d+|[a-z]+", (text or "").lower()))


def _source_text(rec: dict[str, Any]) -> str:
    return " ".join(str(rec.get(k, "")) for k in
                    ("Product Name", "Short Description", "Product Description",
                     "Specifications", "Nutritional Information")).lower()


def ground_example(example: dict[str, Any]) -> dict[str, Any]:
    """Keep only the structured values of a worked example that its own input supports.

    The official examples contain facts taken from sources the input does not show (the masked
    originals). Showing those to the model would teach it to invent, so each structured value is
    kept only if most of its words occur in the example's input.
    """
    rec, out = example["input"], example["output"]
    src = _source_text(rec)
    src_tokens = _tokens(src)

    def supported(value: Any) -> bool:
        toks = _tokens(str(value))
        if not toks:
            return False
        return len(toks & src_tokens) / len(toks) >= 0.75

    g = schema.empty_record()
    for f in schema.TEXT_FIELDS:
        g[f] = out[f]
    g["ingredient_information"] = [v for v in out["ingredient_information"] if supported(v)]
    g["storage_instructions"] = [v for v in out["storage_instructions"] if supported(v)]
    for k, vals in out["variant_attributes"].items():
        g["variant_attributes"][k] = [v for v in vals if supported(v)]
    ps, gps = out["product_specifications"], g["product_specifications"]
    if supported(ps["pack_size"]["value"]):
        gps["pack_size"] = dict(ps["pack_size"])
    gps["application_area"] = [v for v in ps["application_area"] if supported(v)]
    gps["packaging_type"] = ps["packaging_type"] if supported(ps["packaging_type"]) else ""
    gps["packaging_material"] = [v for v in ps["packaging_material"] if supported(v)]
    gps["other"] = [o for o in ps["other"] if supported(o["value"])]
    if supported(out["weight"]["value"]):
        g["weight"] = dict(out["weight"])
    for k in ("height", "width", "depth"):
        if supported(out["dimensions"][k]):
            g["dimensions"][k] = out["dimensions"][k]
    if any(g["dimensions"][k] for k in ("height", "width", "depth")):
        g["dimensions"]["unit"] = out["dimensions"]["unit"]
    nut = out["nutritional_information"]
    for basis in ("per_100g", "per_serving"):
        g["nutritional_information"][basis] = {
            k: v for k, v in nut[basis].items() if str(v).rstrip("0").rstrip(".") in src or str(v) in src}
    if supported(nut["serving_size"]["value"]):
        g["nutritional_information"]["serving_size"] = dict(nut["serving_size"])
    for f in rec.get("_dnp", []):
        if f in schema.OUTPUT_FIELDS:
            g[f] = schema.empty_record()[f]
    return {"input": rec, "output": schema.normalise(g),
            "facts": parsers.deterministic_fields(rec)}


class FewShot:
    """Picks worked examples for a record: same category group first, then closest category."""

    def __init__(self, examples: list[dict[str, Any]]):
        # Examples whose input has no description cannot be answered from their input, so they
        # would only demonstrate invention. Keep them out of the prompt.
        self.examples = [ground_example(e) for e in examples
                         if e["input"].get("Product Description")]

    def select(self, rec: dict[str, Any], k: int, k_weak: int | None = None,
               weak_overlap: float = 0.34) -> list[dict[str, Any]]:
        """k examples; k_weak (if given) when even the best example is a weak category match."""
        if k <= 0 or not self.examples:
            return []
        mine = _tokens(" ".join(rec.get(c, "") for c in ("category_group", "T_DEPARTMENT", "T_CATEGORY")))

        def score(ex: dict[str, Any]) -> tuple:
            e = ex["input"]
            if e["record_id"] == rec["record_id"]:
                return (-1, 0, "")
            same_group = e.get("category_group") == rec.get("category_group")
            theirs = _tokens(" ".join(e.get(c, "") for c in ("category_group", "T_DEPARTMENT", "T_CATEGORY")))
            overlap = len(mine & theirs) / max(1, len(mine | theirs))
            return (int(same_group), overlap, e["record_id"])

        ranked = [e for e in sorted(self.examples, key=score, reverse=True)
                  if e["input"]["record_id"] != rec["record_id"]]
        if k_weak and ranked:
            same_group, overlap, _ = score(ranked[0])
            if not same_group and overlap < weak_overlap:
                k = k_weak
        return ranked[:k]


_SIZE = re.compile(r"(\d+(?:[.,]\d+)?)\s?(ml|l|kg|g|mg|s)\b")


def _size_key(name: str) -> tuple | None:
    found = _SIZE.findall((name or "").lower())
    return found[-1] if found else None


class SiblingIndex:
    """Optional lookup of spec values shared by sibling products in unlabelled_data.csv.

    A sibling has the same brand, the same T_BMC, the same pack size in its name and a similar
    name. A value is returned only when at least `min_siblings` siblings state it and they all
    agree. Measured on unlabelled_data.csv this is right about 95–99% of the time for the default
    fields. It is OFF by default because the organisers say the unlabelled file is not a verified
    answer source for a test record; see the README before enabling it.
    """

    def __init__(self, records: list[dict[str, Any]], cfg: dict[str, Any]):
        self.cfg = cfg
        self.index: dict[tuple, list[tuple[set[str], tuple | None, dict[str, str]]]] = defaultdict(list)
        for r in records:
            specs, _ = parsers.parse_spec_table(r.get("Specifications", ""))
            if specs:
                key = (r.get("T_BRAND", "").lower(), r.get("T_BMC", ""))
                self.index[key].append((_tokens(r["Product Name"]), _size_key(r["Product Name"]), specs))

    def lookup(self, rec: dict[str, Any]) -> dict[str, str]:
        mine, size = _tokens(rec["Product Name"]), _size_key(rec["Product Name"])
        own, _ = parsers.parse_spec_table(rec.get("Specifications", ""))
        sibs = []
        for toks, s, specs in self.index.get((rec.get("T_BRAND", "").lower(), rec.get("T_BMC", "")), []):
            if s != size:
                continue
            if len(toks & mine) / max(1, len(toks | mine)) < float(self.cfg["min_name_jaccard"]):
                continue
            # a sibling that contradicts a value this record does show is not a sibling
            if any(k in specs and specs[k].lower() != v.lower() for k, v in own.items()
                   if k in self.cfg["fields"]):
                continue
            sibs.append(specs)
        out: dict[str, str] = {}
        for field in self.cfg["fields"]:
            if field in own:
                continue
            vals = [s[field] for s in sibs if field in s]
            if len(vals) >= int(self.cfg["min_siblings"]) and len({v.lower() for v in vals}) == 1:
                out[field] = vals[0]
        return out
