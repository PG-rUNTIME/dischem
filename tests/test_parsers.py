import json

from dischem import assemble, parsers, schema

SPEC = ("<table><tr><td>Pack Size</td><td>[MASKED]</td></tr><tr><td>Weight</td><td>19</td></tr>"
        "<tr><td>Height</td><td>108</td></tr><tr><td>Depth</td><td>18</td></tr>"
        "<tr><td>Width</td><td>69</td></tr><tr><td>Application Area</td><td>Lip, Cheeks</td></tr>"
        "<tr><td>Packaging Material</td><td>[MASKED]</td></tr><tr><td>Packaging Type</td><td>Box</td></tr>"
        "<tr><td>Recyclable Packaging</td><td>Yes</td></tr><tr><td>Certifications</td><td>Halaal, Kosher</td></tr></table>")


def test_spec_table_skips_masked_cells():
    values, masked = parsers.parse_spec_table(SPEC)
    assert masked == ["Pack Size", "Packaging Material"]
    assert values["Weight"] == "19" and "Pack Size" not in values


def test_specs_follow_template_conventions():
    out = parsers.specs_to_fields(parsers.parse_spec_table(SPEC)[0])
    assert out["weight"] == {"value": "19", "unit": ""}
    assert out["dimensions"] == {"height": "108", "width": "69", "depth": "18", "unit": ""}
    ps = out["product_specifications"]
    assert ps["application_area"] == ["Lip", "Cheeks"] and ps["packaging_type"] == "Box"
    assert ps["pack_size"] == {"value": "", "unit": ""}
    assert {"attribute": "recyclable_packaging", "value": "Yes"} in ps["other"]
    assert [o["value"] for o in ps["other"] if o["attribute"] == "certification"] == ["Halaal", "Kosher"]


def test_pack_size_units():
    assert parsers.split_quantity("110 millilitre (ml)") == ("110", "millilitre (ml)")
    assert parsers.split_quantity("4,8g") == ("4.8", "g")
    assert parsers.split_quantity("115gram (g") == ("115", "gram (g)")
    out = parsers.specs_to_fields({"Pack Size": "110 millilitre (ml)"})
    assert out["product_specifications"]["pack_size"] == {"value": "110", "unit": "millilitre (ml)"}


def test_size_in_centimetres_is_not_a_pack_size():
    out = parsers.specs_to_fields({"Size": "25cm", "Colour": "Blue And White"})
    assert out["product_specifications"]["pack_size"]["value"] == ""
    assert {"attribute": "size", "value": "25cm"} in out["product_specifications"]["other"]
    assert out["variant_attributes"]["shade"] == ["Blue And White"]


def test_nutrition_keeps_basis_and_drops_masked():
    cell = json.dumps({"nutrients": {
        "Energy (kJ)": {"per_100g": "[MASKED]", "per_serving": "298", "percent_nrv": None},
        "Of which saturated fat (g)": {"per_100g": "0.5", "per_serving": "[MASKED]"},
        "Total sodium (mg)": {"per_100g": "<1", "per_serving": None}},
        "serving_size": "115gram (g"})
    nut, masked = parsers.parse_nutrition(cell)
    assert masked == 2
    assert nut["per_serving"] == {"energy_kj": 298}
    assert nut["per_100g"] == {"saturated_fat_g": 0.5, "sodium_mg": "<1"}
    assert nut["serving_size"] == {"value": "115", "unit": "gram (g)"}
    assert parsers.parse_nutrition("{}")[0] == schema.empty_record()["nutritional_information"]


def test_storage_section_stops_at_blank_line():
    text = "Lead text.\n\nKey Features:\n• A\n• B\n\nStorage: store in a cool, dry place.\n\n\n• Dermatologically approved."
    assert parsers.storage_from_description(text) == ["store in a cool, dry place."]
    assert parsers.storage_from_description("No sections here.") == []


def test_clean_name_fixes_all_caps_brand_only():
    assert parsers.clean_name("PURITY From 6 Months Yoghurt 110ml", "Purity").startswith("Purity From")
    assert parsers.clean_name("CeraVe Foaming Cleanser 236ml", "Cerave") == "CeraVe Foaming Cleanser 236ml"
    assert parsers.clean_name("NATURE'S CHOICE Soya 450g", "Nature'S Choice") == "Nature's Choice Soya 450g"


def test_normalise_removes_placeholders_and_lonely_units():
    out = schema.normalise({
        "short_description": "N/A", "ingredient_information": ["Milk", "milk.", "unknown", "[MASKED]"],
        "weight": {"value": "", "unit": "Kilogram"},
        "dimensions": {"height": "", "width": "", "depth": "", "unit": "cm"},
        "product_specifications": {"pack_size": {"value": "", "unit": "ml"},
                                   "other": [{"attribute": "Recyclable Packaging", "value": "Yes"},
                                             {"attribute": "x", "value": "none"}]},
        "nutritional_information": {"per_100g": [{"nutrient": "Energy (kJ)", "value": "288"},
                                                 {"nutrient": "Protein (g)", "value": "not visible"}]},
    })
    assert out["short_description"] == ""
    assert out["ingredient_information"] == ["Milk"]
    assert out["weight"] == {"value": "", "unit": ""}
    assert out["dimensions"]["unit"] == ""
    assert out["product_specifications"]["pack_size"] == {"value": "", "unit": ""}
    assert out["product_specifications"]["other"] == [{"attribute": "recyclable_packaging", "value": "Yes"}]
    assert out["nutritional_information"]["per_100g"] == {"energy_kj": 288}


def test_blank_cell_round_trip():
    empty = schema.empty_record()
    for f in schema.OUTPUT_FIELDS:
        assert schema.to_cell(f, empty[f]) == ""
        assert schema.from_cell(f, "") == empty[f]


def test_merge_keeps_parsed_facts_and_verifier_cannot_delete_them():
    rec = {"record_id": "1", "Product Name": "Labello Rose 4,8g", "T_BRAND": "Labello",
           "Specifications": SPEC, "Nutritional Information": "{}", "Product Description": "",
           "Short Description": "", "_dnp": ["nutritional_information"]}
    facts = parsers.deterministic_fields(rec)
    llm = {"short_description": "A lip balm.", "long_description": "Labello Rose is a lip balm.",
           "weight": {"value": "25", "unit": "g"},
           "product_specifications": {"pack_size": {"value": "4.8", "unit": "g"}, "packaging_type": "Tube",
                                      "packaging_material": ["Paper"]},
           "nutritional_information": {"per_100g": [{"nutrient": "energy", "value": "1"}]}}
    out = assemble.merge(facts, llm)
    assert out["weight"]["value"] == "19"                       # parsed fact wins
    assert out["product_specifications"]["packaging_type"] == "Box"
    assert out["product_specifications"]["pack_size"]["value"] == "4.8"
    proposed = assemble.proposed_values(out, facts)
    assert ("product_specifications", "packaging_material", "Paper") in proposed
    assert all(v != "19" for _, _, v in proposed)
    out, removed = assemble.remove_unsupported(out, facts, [
        {"field": "product_specifications", "value": "Paper", "reason": ""},
        {"field": "weight", "value": "19", "reason": ""}])
    assert removed == 1 and out["product_specifications"]["packaging_material"] == []
    assert out["weight"]["value"] == "19"
    final = assemble.finalise(rec, out)
    assert final["nutritional_information"] == schema.empty_record()["nutritional_information"]


def test_text_fields_never_blank():
    rec = {"record_id": "2", "Product Name": "Kappa Men Black EDT 60ml", "T_BRAND": "Kappa",
           "T_CATEGORY": "MENS FRAGRANCE", "Short Description": "", "Product Description": "", "_dnp": []}
    out = assemble.finalise(rec, schema.empty_record())
    assert out["online_name"] == "Kappa Men Black EDT 60ml"
    assert "Kappa Men Black EDT 60ml" in out["short_description"] and out["long_description"]
