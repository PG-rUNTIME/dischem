# Dis-Chem Product Data Enrichment — Phase 1 pipeline

End-to-end solution for the Zindi *Dis-Chem Product Data Enrichment Challenge*: reads
`Test_Masked.csv` and the product photos, and writes `outputs/submission.csv` in the required
format. Python only, open-source packages, one open-weight model, one 24 GB GPU.

> **Keep this repository private.** The challenge rules do not allow sharing the data, and code may
> only be shared inside your team. `data/` is git-ignored — do not force-add it.

## How it works

| Stage | What it does | Model? |
|---|---|---|
| 0. Load | Reads records with IDs as text, the image mapping and the 14 worked examples. | no |
| 1. Parse | Reads every unmasked spec-table row, nutrition value and `Storage:` line with plain code. `[MASKED]` cells are treated as missing. | no |
| 2. Photo evidence | For records with photos, the model transcribes what is printed on the pack (claims, pack size, packaging, colour) and says whether the photo matches the product. Mismatches are discarded. | yes |
| 3. Compose | One call per record turns the record, the parsed facts and the photo evidence into the ten output fields, as schema-constrained JSON. | yes |
| 4. Verify | A reviewer call lists model-proposed values the sources do not support; those are deleted. It can only delete, never add, and can never delete a parsed fact. | yes |
| 5. Validate | Exact schema shape, placeholders such as "N/A" removed, do-not-predict fields blanked, lonely units dropped, text fields never left empty, submission checked against `SampleSubmission.csv`. | no |

Design choices and where they come from:

- **Analyse, then check.** NVIDIA's Retail Catalog Enrichment Blueprint pairs a vision-language
  analysis step with a model-based quality check to catch hallucinations. Stages 2–4 follow that
  shape with one open model instead of NVIDIA's hosted microservices (paid services are not allowed).
- **One model for vision and text.** `Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0, open weights on
  Hugging Face). No model swap inside the 3-hour window.
- **vLLM** as the server, using the standard `response_format: json_schema` request so every reply
  is valid JSON for the schema.
- **Precision first.** The guardrail PDF's rule is "an accurate omission is better than a plausible
  invention", and wrong structured values count as false positives.

## What the data allows (measured on the downloaded files)

- 840 of 1,157 test records have at least one photo; 317 have none; 141 have neither a photo nor a
  description.
- The photos checked so far (30) are front-of-pack marketing shots, 1200 × 1200. They show product
  type, claims, pack size, packaging and colour. They rarely show ingredient lists or nutrition tables.
- The `Specifications` column is blank for 964 test records and partly masked for the other 193.
  Weights and dimensions that were removed cannot be recovered from the photos; they stay blank.
- The reference answers in `template.csv` contain facts that are not in the matching input. The
  few-shot examples shown to the model are therefore filtered to values their own input supports.

## Run it on the GPU pod

```bash
# 1. Code
git clone <your-private-repo-url> dischem-enrich && cd dischem-enrich

# 2. Data — copy the Zindi files into ./data (from your computer; replace host and port)
#    Needed: Test_Masked.csv, template_input.csv, template.csv, SampleSubmission.csv,
#            submission_format.json, product_images.zip   (unlabelled_data.csv is optional)
rsync -avP -e "ssh -p <PORT>" /path/to/DisChem-Zindi-Hackathon/data/ root@<POD_HOST>:~/dischem-enrich/data/

# 3. One-time setup: installs packages, unpacks the photos, downloads the model, runs the tests
bash scripts/setup_pod.sh

# 4. Benchmark on 50 records — prints "projected_full_run_minutes"
bash scripts/run_all.sh --limit 50

# 5. Full timed run -> outputs/submission.csv, outputs/run_report.json, logs/run_timing.txt
POD_USD_PER_HOUR=0.35 bash scripts/run_all.sh
```

Then copy `outputs/submission.csv` back and upload it to Zindi.

If the pod restarts mid-run, run step 5 again: finished model calls are read from `cache/` and the
run continues from where it stopped. Delete `cache/` to force a clean run (do this for the final
timed run you will report).

### No GPU needed

```bash
pip install httpx pyyaml pillow pytest && pip install -e .
python -m pytest -q                                   # 15 tests, includes a fake model server
python -m dischem.pipeline run --no-llm               # parser-only baseline submission
python -m dischem.pipeline check outputs/submission.csv
```

The parser-only baseline is a valid submission on its own and a sensible first upload.

## Staying inside the Phase 1 limits

| Limit | How the pipeline handles it |
|---|---|
| One GPU, ≤ 24 GB VRAM | `serve_vllm.sh` caps vLLM at 21.6 GB even on a larger card and pins one GPU. `run_report.json` records the GPU name and peak memory. |
| 3 hours end to end | The clock starts before the model loads. No new model call starts after minute 160; remaining records fall back to the parsed baseline, so a valid file is always written. |
| US$20 | `run_all.sh` prints minutes × your pod's hourly rate. |
| One retry per failed call | `llm.max_retries: 1`. |
| Reproducibility | `temperature 0`, fixed seed, pinned versions in `requirements.lock.txt` (written by setup). |

## Local scoring

```bash
python -m dischem.evaluate score --pred outputs/submission.csv --ref data/template.csv
python -m dischem.evaluate make-val --n 300       # needs data/unlabelled_data.csv
python -m dischem.pipeline run --set input_file=val_input.csv --set output_dir=outputs/val
python -m dischem.evaluate score --pred outputs/val/submission.csv --ref data/val_reference.csv
```

This scorer follows the published rules but matches by words, not meaning. Use it to compare two
versions of the pipeline; it will not predict the leaderboard number.

## Settings worth testing with your submissions

- `verify.enabled` — off gives more values and more risk of false positives.
- `images.max_per_record`, `images.max_side_px` — more pixels help small print and cost time.
- `compose.few_shot` — 0 to 3.
- `llm.concurrency` — raise it if the benchmark shows spare time.
- `siblings.enabled` — fills a few spec rows (packaging type, application area, recyclable
  packaging, certifications) from products in `unlabelled_data.csv` with the same brand, product
  line and pack size, only when at least two agree. On the unlabelled file this was right 95–99% of
  the time, and it applies to 29 test records. It is **off by default** because the Data page says
  the unlabelled file is not a verified answer source for a test record. Ask the organisers before
  using it in a final submission.
- A larger model: set `MODEL=...` for the scripts and `llm.model` in the config. A 4-bit build of
  `Qwen/Qwen3.8-27B` fits in 24 GB but is slower; benchmark it first, and add
  `llm.extra_body: {chat_template_kwargs: {enable_thinking: false}}` for models with a thinking mode.

## What has and has not been tested

Tested without a GPU: all parsers on the real files; a full 1,157-record run against a fake model
server (prompt building, photo encoding, merging, verification, retry, cache, submission check);
the parser-only baseline, which passes the submission check against `SampleSubmission.csv`.

Not yet tested: anything that needs the real model — the vLLM launch flags on your pod, actual
speed, peak memory, and output quality. Step 4 (the 50-record benchmark) is where those show up.

## Layout

```
config/default.yaml      settings
scripts/                 setup_pod.sh, serve_vllm.sh, run_all.sh
src/dischem/
  parsers.py             spec table, nutrition, storage, product name
  schema.py              output shape, empty values, normalisation
  prompts.py             prompts and JSON schemas for the three model stages
  llm.py                 async client, one retry, disk cache
  retrieval.py           few-shot selection, optional sibling lookup
  assemble.py            merging, verifier deletions, final checks
  pipeline.py            the run
  evaluate.py            local scorer and validation-set builder
tests/                   parser tests and an end-to-end test with a fake server
```
