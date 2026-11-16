"""End-to-end run against a fake OpenAI-compatible server (no GPU needed).

The fake server answers like vLLM would: one failure first to exercise the single retry, a photo
evidence reply, a composition that includes one invented value, and a verifier that rejects it.
"""
import asyncio
import csv
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest
from PIL import Image

from dischem import io, pipeline, schema
from dischem.config import load_config
from dischem.llm import ChatClient

SPEC = "<table><tr><td>Pack Size</td><td>[MASKED]</td></tr><tr><td>Packaging Type</td><td>Tube</td></tr></table>"


def _write_inputs(d: Path) -> None:
    cols = io.INPUT_COLUMNS + ["do_not_predict_fields"]
    rows = [
        dict(record_id="00123", category_group="PERSONAL CARE", **{
            "Product Name": "LIRENE Stop Roughness Foot Cream 75ml", "Short Description": "",
            "Product Description": "", "Nutritional Information": "{}", "Specifications": SPEC,
            "T_DEPARTMENT": "FOOT CARE", "T_CATEGORY": "FOOT CREAMS", "T_BMC": "FOOT", "T_BRAND": "Lirene",
            "T_ARTICLE_CATEGORY": "Single article", "T_WEIGHT_UNIT": "Kilogram", "VOLUME": "0.1",
            "T_VOLUME_UNIT": "Cubic decimeter",
            "do_not_predict_fields": '["nutritional_information","category_specific_fields"]'}),
        dict(record_id="456", category_group="BABY TOYS", **{
            "Product Name": "Winfun Jungle Pals Playmat", "Short Description": "A playmat.",
            "Product Description": "A colourful playmat.\n\nStorage: store in a cool, dry place.",
            "Nutritional Information": "{}", "Specifications": "", "T_DEPARTMENT": "TOYS",
            "T_CATEGORY": "PLAYMATS", "T_BMC": "MATS", "T_BRAND": "Winfun",
            "T_ARTICLE_CATEGORY": "Single article", "T_WEIGHT_UNIT": "Kilogram", "VOLUME": "",
            "T_VOLUME_UNIT": "", "do_not_predict_fields": '["ingredient_information"]'}),
    ]
    with open(d / "Test_Masked.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (d / "image_mapping.csv").write_text("record_id,img_filename\n00123,images/img_1.jpg\n456,\n")
    (d / "images").mkdir()
    Image.new("RGB", (1600, 1200), "white").save(d / "images" / "img_1.jpg")


class FakeVLLM:
    def __init__(self):
        self.calls = Counter()
        self.failed_once = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        body = json.loads(request.content)
        tag = body["response_format"]["json_schema"]["name"]
        assert body["temperature"] == 0.0 and body["seed"] == 42
        self.calls[tag] += 1
        if tag == "enriched_product" and not self.failed_once:
            self.failed_once = True
            return httpx.Response(500, text="boom")
        user = body["messages"][1]["content"]
        text = user[-1]["text"]
        if tag == "photo_evidence":
            assert user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            reply = {"matches_product": "yes", "printed_text": ["20% UREA", "with babassu oil"],
                     "product_form": "cream", "packaging_type": "Tube", "packaging_materials": [],
                     "colours": [], "pack_size": "75 ml", "claims": ["Intensely smoothing"],
                     "ingredients": ["urea", "babassu oil"], "directions": [], "storage_instructions": [],
                     "certifications": [], "nutrition_panel": ""}
        elif tag == "enriched_product":
            lirene = "Product name: LIRENE" in text.split("### Product to enrich")[-1]
            reply = {
                "short_description": "An intensely smoothing foot cream." if lirene else "A playmat for babies.",
                "long_description": "Lirene Stop Roughness Foot Cream 75ml is a foot cream-concentrate with 20% urea."
                if lirene else "Winfun Jungle Pals Playmat is a colourful playmat.",
                "ingredient_information": ["urea", "babassu oil"] if lirene else ["plastic"],
                "storage_instructions": [],
                "variant_attributes": {"shade": [], "flavour": [], "fragrance": [], "other": []},
                "product_specifications": {"pack_size": {"value": "75", "unit": "ml"} if lirene else {"value": "", "unit": ""},
                                           "application_area": ["Feet"] if lirene else [],
                                           "packaging_type": "Jar", "packaging_material": ["Plastic"] if lirene else [],
                                           "other": []},
                "weight": {"value": "N/A", "unit": "Kilogram"},
                "dimensions": {"height": "", "width": "", "depth": "", "unit": ""},
                "nutritional_information": {"serving_size": {"value": "", "unit": ""},
                                            "per_100g": [{"nutrient": "energy", "value": "5"}], "per_serving": []},
            }
        else:
            assert "Proposed values" in text
            reply = {"unsupported": [{"field": "product_specifications", "value": "Plastic", "reason": "not stated"},
                                     {"field": "product_specifications", "value": "Tube", "reason": "wrong"}]}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "```json\n" + json.dumps(reply) + "\n```"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}})


@pytest.fixture()
def workspace(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_inputs(data)
    cfg = load_config(None, [f"data_dir={data}", f"images_dir={data / 'images'}",
                             f"output_dir={tmp_path / 'out'}", f"cache_dir={tmp_path / 'cache'}",
                             "llm.concurrency=2"])
    return cfg, tmp_path


def _run(cfg, fake, monkeypatch):
    real = ChatClient.__init__

    def patched(self, c, cache_dir, transport=None):
        real(self, c, cache_dir, transport=httpx.MockTransport(fake))

    monkeypatch.setattr(ChatClient, "__init__", patched)
    return asyncio.run(pipeline.run(cfg, use_llm=True, limit=None, wait_server=True))


def test_end_to_end_with_fake_server(workspace, monkeypatch):
    cfg, tmp = workspace
    fake = FakeVLLM()
    report = _run(cfg, fake, monkeypatch)
    assert report["submission_problems"] == []
    assert report["llm_stats"]["retries"] == 1 and report["llm_stats"]["failed"] == 0
    assert fake.calls["photo_evidence"] == 1          # only the record that has a photo

    rows = {r["article_id"]: r for r in io.read_csv(tmp / "out" / "submission.csv")}
    assert list(rows) == ["00123", "456"]             # ids kept as text, in input order
    a = rows["00123"]
    assert a["online_name"] == "Lirene Stop Roughness Foot Cream 75ml"
    spec = json.loads(a["product_specifications"])
    assert spec["packaging_type"] == "Tube"           # parsed fact beats the model and the verifier
    assert spec["packaging_material"] == []           # invented value removed by the verifier
    assert spec["pack_size"] == {"value": "75", "unit": "ml"}
    assert a["nutritional_information"] == ""         # do_not_predict
    assert a["weight"] == ""                          # 'N/A' plus a lonely unit is dropped
    assert json.loads(a["ingredient_information"]) == ["urea", "babassu oil"]
    b = rows["456"]
    assert b["ingredient_information"] == ""          # do_not_predict
    assert json.loads(b["storage_instructions"]) == ["store in a cool, dry place."]


def test_second_run_is_served_from_cache(workspace, monkeypatch):
    cfg, tmp = workspace
    first = FakeVLLM()
    _run(cfg, first, monkeypatch)
    before = (tmp / "out" / "submission.csv").read_text()
    second = FakeVLLM()
    second.failed_once = True
    report = _run(cfg, second, monkeypatch)
    assert sum(second.calls.values()) == 0 and report["llm_stats"]["cached"] >= 4
    assert (tmp / "out" / "submission.csv").read_text() == before     # reproducible


def test_server_down_falls_back_to_parsed_baseline(workspace, monkeypatch):
    cfg, tmp = workspace

    def dead(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(503, text="down")

    report = _run(cfg, dead, monkeypatch)
    assert report["submission_problems"] == []
    assert report["llm_stats"]["calls"] == 2 * report["llm_stats"]["failed"]   # one retry each, no more
    rows = {r["article_id"]: r for r in io.read_csv(tmp / "out" / "submission.csv")}
    assert rows["00123"]["short_description"] and rows["456"]["short_description"] == "A playmat."


def test_no_llm_baseline_has_exact_columns(workspace):
    cfg, tmp = workspace
    report = asyncio.run(pipeline.run(cfg, use_llm=False, limit=None, wait_server=False))
    assert report["submission_problems"] == []
    with open(tmp / "out" / "submission.csv", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == schema.SUBMISSION_COLUMNS
