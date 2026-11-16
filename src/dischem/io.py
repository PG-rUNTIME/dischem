"""Loading the challenge files and writing the submission. Identifiers are always text."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import schema

csv.field_size_limit(10**9)

INPUT_COLUMNS = [
    "record_id", "category_group", "Product Name", "Short Description", "Product Description",
    "Nutritional Information", "Specifications", "T_DEPARTMENT", "T_CATEGORY", "T_BMC", "T_BRAND",
    "T_ARTICLE_CATEGORY", "T_WEIGHT_UNIT", "VOLUME", "T_VOLUME_UNIT",
]


def read_csv(path: str | os.PathLike) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            r[k] = (v or "").strip() if k != "record_id" else (v or "").strip()
    return rows


def load_records(path: str | os.PathLike) -> list[dict[str, Any]]:
    rows = read_csv(path)
    missing = [c for c in INPUT_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing input columns {missing}")
    for r in rows:
        raw = r.get("do_not_predict_fields") or "[]"
        try:
            r["_dnp"] = [str(x) for x in json.loads(raw)]
        except json.JSONDecodeError:
            r["_dnp"] = []
    return rows


def load_image_mapping(path: str | os.PathLike) -> dict[str, list[str]]:
    """record_id -> image file names. Use the CSV: the xlsx stores ids as numbers."""
    mapping: dict[str, list[str]] = defaultdict(list)
    if not Path(path).exists():
        return mapping
    for r in read_csv(path):
        rid = r["record_id"]
        mapping[rid]
        name = r.get("img_filename", "")
        if name:
            mapping[rid].append(Path(name).name)
    return mapping


def load_examples(data_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """The 14 worked examples: input row joined to its expected output."""
    d = Path(data_dir)
    if not (d / "template_input.csv").exists() or not (d / "template.csv").exists():
        return []
    outputs = {r["article_id"]: r for r in read_csv(d / "template.csv")}
    examples = []
    for rec in load_records(d / "template_input.csv"):
        out = outputs.get(rec["record_id"])
        if not out:
            continue
        parsed = {f: schema.from_cell(f, out.get(f, "")) for f in schema.OUTPUT_FIELDS}
        examples.append({"input": rec, "output": schema.normalise(parsed)})
    return examples


def write_submission(path: str | os.PathLike, records: list[dict[str, Any]],
                     results: dict[str, dict[str, Any]]) -> None:
    """One row per input record, in input order, with the exact SampleSubmission columns."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(schema.SUBMISSION_COLUMNS)
        for rec in records:
            out = results.get(rec["record_id"]) or schema.empty_record()
            w.writerow([rec["record_id"]] + [schema.to_cell(f, out[f]) for f in schema.OUTPUT_FIELDS])
    os.replace(tmp, path)


def check_submission(path: str | os.PathLike, records: list[dict[str, Any]],
                     sample_path: str | os.PathLike | None = None) -> list[str]:
    """Return a list of problems; empty means the file is safe to upload."""
    problems: list[str] = []
    rows = read_csv(path)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    if header != schema.SUBMISSION_COLUMNS:
        problems.append(f"header differs from required columns: {header}")
    ids = [r["article_id"] for r in rows]
    want = [r["record_id"] for r in records]
    if ids != want:
        problems.append(f"article_id list differs from input ({len(ids)} rows vs {len(want)})")
    if sample_path and Path(sample_path).exists():
        sample_ids = [r["article_id"] for r in read_csv(sample_path)]
        if sorted(sample_ids) != sorted(ids):
            problems.append("article_id set differs from SampleSubmission.csv")
    dnp = {r["record_id"]: set(r.get("_dnp", [])) for r in records}
    for r in rows:
        rid = r["article_id"]
        for f in schema.OUTPUT_FIELDS:
            cell = r.get(f, "")
            if f in dnp.get(rid, ()) and cell:
                problems.append(f"{rid}: {f} is in do_not_predict_fields but is filled")
            if "[MASKED]" in cell:
                problems.append(f"{rid}: {f} contains [MASKED]")
            if f in schema.TEXT_FIELDS:
                if schema.is_placeholder(cell) and cell:
                    problems.append(f"{rid}: {f} is a placeholder value")
                continue
            if not cell:
                continue
            try:
                value = json.loads(cell)
            except json.JSONDecodeError as exc:
                problems.append(f"{rid}: {f} is not valid JSON ({exc})")
                continue
            renorm = schema.normalise({f: value})[f]
            if renorm != value:
                problems.append(f"{rid}: {f} does not match the schema shape")
    return problems[:200]
