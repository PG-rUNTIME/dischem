"""Local scoring, so the 20 leaderboard submissions are not the only feedback.

    python -m dischem.evaluate score   --pred outputs/submission.csv --ref data/template.csv
    python -m dischem.evaluate make-val --n 300        # masked copy of unlabelled records

This is a PROXY for the official LLM judge, not a copy of it. It follows the published rules
(0–1 text score in five bands; value-level TP/FP/FN pooled per structured field; plain mean over
fields; fields with an empty reference are skipped) but matches by words, not by meaning. Use it
to compare two versions of the pipeline, not to predict the leaderboard number.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import io, parsers, schema
from .config import load_config

_STOP = set("a an the and or of for to in on with by is are it its this that from as at be".split())


def _tok(text: Any) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in _STOP]


def text_score(pred: str, ref: str) -> float:
    """Token F1 mapped to the judge's five bands."""
    p, r = _tok(pred), _tok(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(r))
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    f1 = 2 * precision * recall / (precision + recall)
    for threshold, band in ((0.80, 1.0), (0.60, 0.75), (0.40, 0.5), (0.20, 0.25)):
        if f1 >= threshold:
            return band
    return 0.0


def _same(a: str, b: str) -> bool:
    ta, tb = set(_tok(a)), set(_tok(b))
    if not ta or not tb:
        return False
    try:
        if abs(float(str(a).replace(",", ".")) - float(str(b).replace(",", "."))) < 1e-9:
            return True
    except ValueError:
        pass
    return ta == tb or len(ta & tb) / len(ta | tb) >= 0.6


def struct_counts(field: str, pred: Any, ref: Any) -> tuple[int, int, int]:
    """(TP, FP, FN). A predicted value matching nothing in the same slot counts as FP."""
    pl, rl = schema.leaves(field, pred), schema.leaves(field, ref)
    # units are judged with their value, not on their own
    pl = [(p, v) for p, v in pl if not p.endswith("unit")]
    rl = [(p, v) for p, v in rl if not p.endswith("unit")]
    used: set[int] = set()
    tp = 0
    for path, value in pl:
        for i, (rpath, rvalue) in enumerate(rl):
            if i in used:
                continue
            slot = path.split(".")[0] == rpath.split(".")[0]
            if slot and _same(value, rvalue):
                used.add(i)
                tp += 1
                break
    return tp, len(pl) - tp, len(rl) - len(used)


def score(pred_rows: list[dict[str, str]], ref_rows: list[dict[str, str]]) -> dict[str, Any]:
    preds = {r["article_id"]: r for r in pred_rows}
    text: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for ref in ref_rows:
        pred = preds.get(ref["article_id"], {})
        for f in schema.TEXT_FIELDS:
            if ref.get(f, "").strip():
                text[f].append(text_score(pred.get(f, ""), ref[f]))
        for f in schema.STRUCT_FIELDS:
            rv = schema.normalise({f: schema.from_cell(f, ref.get(f, ""))})[f]
            if schema.is_empty(f, rv):
                continue
            try:
                pv = schema.normalise({f: schema.from_cell(f, pred.get(f, ""))})[f]
            except json.JSONDecodeError:
                pv = schema.empty_record()[f]
            tp, fp, fn = struct_counts(f, pv, rv)
            c = counts[f]
            c[0] += tp; c[1] += fp; c[2] += fn
    fields: dict[str, float] = {f: sum(v) / len(v) for f, v in text.items() if v}
    detail: dict[str, Any] = {}
    for f, (tp, fp, fn) in counts.items():
        if tp + fn:
            fields[f] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            detail[f] = {"tp": tp, "fp": fp, "fn": fn}
    overall = sum(fields.values()) / len(fields) if fields else 0.0
    return {"overall": round(overall, 4), "fields": {k: round(v, 4) for k, v in fields.items()},
            "counts": detail, "records": len(ref_rows)}


def make_validation(cfg: dict[str, Any], n: int, seed: int) -> tuple[Path, Path]:
    """Mask unlabelled records the way the test set is masked; the originals become the reference.

    Rates are taken from Test_Masked.csv: descriptions removed for ~41% of records, the spec
    table removed for ~83% and half of its cells masked otherwise.
    """
    rng = random.Random(seed)
    data_dir = Path(cfg["data_dir"])
    pool = [r for r in io.load_records(data_dir / "unlabelled_data.csv") if r.get("Specifications")]
    by_group: dict[str, list] = defaultdict(list)
    for r in pool:
        by_group[r["category_group"]].append(r)
    picked = []
    groups = sorted(by_group)
    while len(picked) < n and any(by_group.values()):
        for g in groups:
            if by_group[g] and len(picked) < n:
                picked.append(by_group[g].pop(rng.randrange(len(by_group[g]))))

    in_path, ref_path = data_dir / "val_input.csv", data_dir / "val_reference.csv"
    with open(in_path, "w", encoding="utf-8", newline="") as fi, \
            open(ref_path, "w", encoding="utf-8", newline="") as fr:
        wi = csv.DictWriter(fi, fieldnames=io.INPUT_COLUMNS + ["do_not_predict_fields"])
        wr = csv.writer(fr)
        wi.writeheader()
        wr.writerow(schema.SUBMISSION_COLUMNS)
        for r in picked:
            ref = parsers.deterministic_fields(r)
            ref["short_description"], ref["long_description"] = r["Short Description"], r["Product Description"]
            ref = schema.normalise(ref)
            wr.writerow([r["record_id"]] + [schema.to_cell(f, ref[f]) for f in schema.OUTPUT_FIELDS])

            m = {c: r[c] for c in io.INPUT_COLUMNS}
            if rng.random() < 0.41:
                m["Short Description"] = m["Product Description"] = ""
            if rng.random() < 0.83:
                m["Specifications"] = ""
            else:
                m["Specifications"] = re.sub(
                    r"(<tr><td>.*?</td><td>)(.*?)(</td></tr>)",
                    lambda x: x.group(1) + (parsers.MASK if rng.random() < 0.5 else x.group(2)) + x.group(3),
                    m["Specifications"], flags=re.S)
            m["do_not_predict_fields"] = "[]"
            wi.writerow(m)
    return in_path, ref_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dischem.evaluate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--ref", required=True)
    v = sub.add_parser("make-val")
    v.add_argument("--config", default=None)
    v.add_argument("--n", type=int, default=300)
    v.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.cmd == "score":
        print(json.dumps(score(io.read_csv(args.pred), io.read_csv(args.ref)), indent=2))
        return 0
    in_path, ref_path = make_validation(load_config(args.config), args.n, args.seed)
    print(f"wrote {in_path} and {ref_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
