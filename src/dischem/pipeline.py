"""End-to-end run: parse -> read photos -> compose -> verify -> validate -> submission.csv

Usage:
    python -m dischem.pipeline run   --config config/default.yaml
    python -m dischem.pipeline run   --no-llm                      # parser-only baseline
    python -m dischem.pipeline run   --limit 50                    # timing benchmark
    python -m dischem.pipeline check outputs/submission.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import assemble, io, parsers, prompts, schema
from .config import load_config
from .llm import ChatClient, encode_image
from .retrieval import FewShot, SiblingIndex


class Budget:
    """Wall-clock guard for the 3-hour limit. Past the deadline no new model call is started."""

    def __init__(self, cfg: dict[str, Any]):
        self.start = time.time()
        self.limit = float(cfg["max_runtime_minutes"]) * 60
        self.reserve = float(cfg["reserve_minutes"]) * 60

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def can_call(self) -> bool:
        return self.elapsed < self.limit - self.reserve

    def can_verify(self) -> bool:
        return self.elapsed < 0.8 * (self.limit - self.reserve)


class GpuMonitor:
    """Samples nvidia-smi so the run report can show the 24 GB cap was respected."""

    def __init__(self) -> None:
        self.peak_mib = 0
        self.total_mib = 0
        self.name = ""
        self._task: asyncio.Task | None = None

    def _sample(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            if out:
                name, used, total = [x.strip() for x in out[0].split(",")]
                self.name, self.total_mib = name, int(total)
                self.peak_mib = max(self.peak_mib, int(used))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    async def _loop(self) -> None:
        while True:
            self._sample()
            await asyncio.sleep(15)

    def start(self) -> None:
        self._sample()
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> dict[str, Any]:
        if self._task:
            self._task.cancel()
        self._sample()
        return {"gpu": self.name, "peak_memory_mib": self.peak_mib, "total_memory_mib": self.total_mib}


def find_images(rec_id: str, mapping: dict[str, list[str]], images_dir: Path, limit: int) -> list[Path]:
    paths = []
    for name in mapping.get(rec_id, []):
        p = images_dir / name
        if p.exists():
            paths.append(p)
    return paths[:limit]


async def enrich_record(rec: dict[str, Any], *, cfg: dict[str, Any], client: ChatClient | None,
                        budget: Budget, few: FewShot, mapping: dict[str, list[str]],
                        siblings: SiblingIndex | None, counters: Counter) -> dict[str, Any]:
    facts = parsers.deterministic_fields(rec)
    sib_values: dict[str, str] = {}
    if siblings:
        sib_values = siblings.lookup(rec)
        if sib_values:
            counters["records_with_sibling_values"] += 1
            meta = facts["_meta"]
            facts = assemble.merge(facts, parsers.specs_to_fields(sib_values))
            facts["_meta"] = meta

    if client is None:
        return assemble.finalise(rec, assemble.merge(facts, None))

    # Stage A — photo evidence
    evidence = None
    if cfg["images"]["enabled"] and budget.can_call():
        paths = find_images(rec["record_id"], mapping, Path(cfg["images_dir"]), int(cfg["images"]["max_per_record"]))
        if paths:
            urls = await asyncio.to_thread(
                lambda: [encode_image(p, int(cfg["images"]["max_side_px"]), int(cfg["images"]["jpeg_quality"]))
                         for p in paths])
            evidence = await client.json_chat(
                tag="photo_evidence", system=prompts.EVIDENCE_SYSTEM,
                user=prompts.evidence_user(rec, len(urls)), schema=prompts.EVIDENCE_SCHEMA,
                images=urls, max_tokens=int(cfg["images"]["max_tokens"]),
                validate=_require_dict)
            if evidence is None:
                counters["evidence_failed"] += 1
            elif evidence.get("matches_product") == "no":
                counters["evidence_rejected_mismatch"] += 1
                evidence = None
            else:
                counters["evidence_ok"] += 1

    # Stage B — composition
    draft = None
    if budget.can_call():
        examples = few.select(rec, int(cfg["compose"]["few_shot"]),
                             int(cfg["compose"].get("few_shot_weak") or 0) or None)
        draft = await client.json_chat(
            tag="enriched_product", system=prompts.COMPOSE_SYSTEM,
            user=prompts.compose_user(rec, facts, evidence, examples,
                                      int(cfg["compose"]["max_source_chars"]), sib_values),
            schema=prompts.COMPOSE_SCHEMA, max_tokens=int(cfg["compose"]["max_tokens"]),
            validate=_require_dict)
        counters["compose_ok" if draft else "compose_failed"] += 1
    else:
        counters["compose_skipped_budget"] += 1
    out = assemble.merge(facts, draft)
    for f in rec.get("_dnp", []):                 # do not spend verifier effort on blanked fields
        if f in schema.OUTPUT_FIELDS:
            out[f] = schema.empty_record()[f]

    # Stage C — verification (deletes unsupported model values; never adds)
    if draft and cfg["verify"]["enabled"] and budget.can_verify():
        proposed = assemble.proposed_values(out, facts)
        if proposed:
            verdict = await client.json_chat(
                tag="verification", system=prompts.VERIFY_SYSTEM,
                user=prompts.verify_user(rec, facts, evidence, proposed,
                                         int(cfg["compose"]["max_source_chars"]), sib_values),
                schema=prompts.VERIFY_SCHEMA, max_tokens=int(cfg["verify"]["max_tokens"]),
                validate=_require_dict)
            if verdict is None:
                counters["verify_failed"] += 1
            else:
                out, removed = assemble.remove_unsupported(out, facts, verdict.get("unsupported") or [])
                counters["verify_ok"] += 1
                counters["values_removed_by_verifier"] += removed
    return assemble.finalise(rec, out)


def _require_dict(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


async def run(cfg: dict[str, Any], *, use_llm: bool, limit: int | None, wait_server: bool) -> dict[str, Any]:
    random.seed(int(cfg["seed"]))
    budget = Budget(cfg["budget"])
    data_dir, out_dir = Path(cfg["data_dir"]), Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = io.load_records(data_dir / cfg["input_file"])
    todo = records[:limit] if limit else records
    mapping = io.load_image_mapping(data_dir / "image_mapping.csv")
    few = FewShot(io.load_examples(data_dir))
    siblings = None
    if cfg["siblings"]["enabled"]:
        siblings = SiblingIndex(io.load_records(data_dir / "unlabelled_data.csv"), cfg["siblings"])

    with_images = sum(1 for r in todo if find_images(r["record_id"], mapping, Path(cfg["images_dir"]), 1))
    print(f"[setup] {len(todo)} records, {with_images} with a photo on disk, "
          f"{len(few.examples)} few-shot examples, llm={'on' if use_llm else 'off'}", flush=True)
    if use_llm and cfg["images"]["enabled"] and with_images == 0:
        print("[warn] no photos found under images_dir — check that product_images.zip is unpacked",
              file=sys.stderr, flush=True)

    client = None
    gpu = GpuMonitor()
    if use_llm:
        llm_cfg = dict(cfg["llm"], seed=cfg["seed"])
        client = ChatClient(llm_cfg, Path(cfg["cache_dir"]) / "llm")
        if wait_server:
            await client.wait_ready()
        gpu.start()

    counters: Counter = Counter()
    results: dict[str, dict[str, Any]] = {}
    done = 0
    queue: asyncio.Queue = asyncio.Queue()
    for r in todo:
        queue.put_nowait(r)

    async def worker() -> None:
        nonlocal done
        while True:
            try:
                rec = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results[rec["record_id"]] = await enrich_record(
                    rec, cfg=cfg, client=client, budget=budget, few=few, mapping=mapping,
                    siblings=siblings, counters=counters)
            except Exception as exc:  # one bad record must not lose the run
                counters["record_errors"] += 1
                print(f"[error] {rec['record_id']}: {exc!r}", file=sys.stderr, flush=True)
                results[rec["record_id"]] = assemble.finalise(
                    rec, assemble.merge(parsers.deterministic_fields(rec), None))
            done += 1
            if done % 50 == 0 or done == len(todo):
                rate = done / max(budget.elapsed, 1e-6)
                print(f"[progress] {done}/{len(todo)}  {budget.elapsed/60:.1f} min  "
                      f"{rate*60:.0f} records/min", flush=True)
                io.write_submission(out_dir / "submission.partial.csv", todo, results)

    n_workers = max(1, int(cfg["llm"]["concurrency"]) * 2) if use_llm else 8
    await asyncio.gather(*(worker() for _ in range(n_workers)))

    sub_path = out_dir / "submission.csv"
    io.write_submission(sub_path, todo, results)
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as fh:
        for r in todo:
            fh.write(json.dumps({"article_id": r["record_id"], **results[r["record_id"]]}, ensure_ascii=False) + "\n")
    is_test = cfg["input_file"] == "Test_Masked.csv" and not limit
    problems = io.check_submission(sub_path, todo, data_dir / "SampleSubmission.csv" if is_test else None)

    filled = Counter()
    for out in results.values():
        for f in schema.OUTPUT_FIELDS:
            filled[f] += not schema.is_empty(f, out[f])
    elapsed = budget.elapsed
    report = {
        "records": len(todo), "elapsed_minutes": round(elapsed / 60, 2),
        "projected_full_run_minutes": round(elapsed / 60 * len(records) / max(1, len(todo)), 1),
        "model": cfg["llm"]["model"] if use_llm else None, "seed": cfg["seed"],
        "llm_stats": client.stats if client else {}, "stage_counters": dict(counters),
        "fields_filled": dict(filled), "gpu": gpu.stop() if use_llm else {},
        "submission": str(sub_path), "submission_problems": problems,
    }
    if client:
        await client.aclose()
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def rebuild(cfg: dict[str, Any], records_path: str, out_path: str) -> int:
    """Re-run the post-processing on saved model output; no model server needed."""
    data_dir = Path(cfg["data_dir"])
    records = io.load_records(data_dir / cfg["input_file"])
    saved = {}
    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            saved[str(row.pop("article_id"))] = row
    results = {r["record_id"]: assemble.finalise(r, saved.get(r["record_id"]) or schema.empty_record())
               for r in records}
    io.write_submission(out_path, records, results)
    problems = io.check_submission(out_path, records, data_dir / "SampleSubmission.csv")
    print(f"wrote {out_path}: " + ("OK, valid" if not problems else "\n".join(problems)))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dischem", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline and write outputs/submission.csv")
    r.add_argument("--config", default=None)
    r.add_argument("--set", action="append", default=[], metavar="key=value",
                   help="override a config value, e.g. --set llm.concurrency=8")
    r.add_argument("--no-llm", action="store_true", help="parser-only baseline, no model server needed")
    r.add_argument("--limit", type=int, default=None, help="only the first N records (benchmark)")
    r.add_argument("--no-wait", action="store_true", help="do not wait for the model server to come up")
    c = sub.add_parser("check", help="validate a submission file")
    c.add_argument("submission")
    c.add_argument("--config", default=None)
    c.add_argument("--set", action="append", default=[])
    b = sub.add_parser("rebuild", help="re-apply finalise() to a saved records.jsonl and write a submission")
    b.add_argument("records_jsonl")
    b.add_argument("--out", default="outputs/submission_rebuilt.csv")
    b.add_argument("--config", default=None)
    b.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.cmd == "rebuild":
        return rebuild(cfg, args.records_jsonl, args.out)
    if args.cmd == "run":
        report = asyncio.run(run(cfg, use_llm=not args.no_llm, limit=args.limit,
                                 wait_server=not args.no_wait))
        return 1 if report["submission_problems"] else 0
    records = io.load_records(Path(cfg["data_dir"]) / cfg["input_file"])
    problems = io.check_submission(args.submission, records,
                                   Path(cfg["data_dir"]) / "SampleSubmission.csv")
    print("OK: submission is valid" if not problems else "\n".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
