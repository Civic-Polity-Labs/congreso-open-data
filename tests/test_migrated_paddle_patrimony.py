"""Regression tests owned by the public acquisition package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from congreso_open_data.paddle_patrimony import normalize_paddle_result, paddle_numeric_candidates


def _source(tmp_path: Path) -> tuple[Path, str]:
    image = tmp_path / "page.png"
    image.write_bytes(b"fixture-image")
    return image, hashlib.sha256(image.read_bytes()).hexdigest()


def test_normalize_preserves_boxes_and_excludes_layout_noise(tmp_path: Path):
    image, source_sha = _source(tmp_path)
    raw = {
        "width": 1000,
        "height": 2000,
        "model_settings": {"use_table_recognition": True},
        "overall_ocr_res": {
            "rec_texts": ["37.842,15 €", "2"],
            "rec_scores": [0.99, 0.98],
            "rec_boxes": [[100, 200, 300, 240], [950, 1900, 970, 1940]],
        },
        "parsing_res_list": [
            {"block_label": "table", "block_content": "income", "block_bbox": [90, 180, 900, 1500]},
            {
                "block_label": "aside_text",
                "block_content": "stamp",
                "block_bbox": [0, 500, 50, 1500],
            },
        ],
        "table_res_list": [
            {"pred_html": "<table></table>", "cell_box_list": [[90, 180, 900, 250]]}
        ],
    }
    result = normalize_paddle_result(raw, source_sha256=source_sha, page_number=1, image_path=image)
    assert result["ocr_lines"][0]["bbox_norm"] == [0.1, 0.1, 0.3, 0.12]
    assert result["blocks"][0]["field_eligible"] is True
    assert result["blocks"][1]["field_eligible"] is False
    assert result["ocr_lines"][0]["field_eligible"] is True
    assert result["ocr_lines"][1]["field_eligible"] is False
    candidates = paddle_numeric_candidates(result)
    assert candidates[0]["candidate_status"] == "review_required"
    assert candidates[0]["cell_index"] == 0
    assert candidates[0]["raw_text"] == "37.842,15 €"
    assert result["tables"][0]["cells"][0]["bbox_px"] == [90.0, 180.0, 900.0, 250.0]
    assert result["tables"][0]["cells"][0]["bbox_valid"] is True
    assert result["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


def test_normalize_rejects_mismatched_ocr_arrays(tmp_path: Path):
    image, source_sha = _source(tmp_path)
    raw = {
        "width": 10,
        "height": 10,
        "overall_ocr_res": {"rec_texts": ["x"], "rec_scores": [], "rec_boxes": []},
    }
    with pytest.raises(ValueError, match="different lengths"):
        normalize_paddle_result(raw, source_sha256=source_sha, page_number=1, image_path=image)


def test_json_serialization_is_stable(tmp_path: Path):
    image, source_sha = _source(tmp_path)
    raw = {
        "width": 1,
        "height": 1,
        "overall_ocr_res": {"rec_texts": [], "rec_scores": [], "rec_boxes": []},
    }
    result = normalize_paddle_result(raw, source_sha256=source_sha, page_number=1, image_path=image)
    assert json.dumps(result, ensure_ascii=False, sort_keys=True)


def test_numeric_candidates_quarantine_low_score(tmp_path: Path):
    image, source_sha = _source(tmp_path)
    raw = {
        "width": 100,
        "height": 100,
        "overall_ocr_res": {
            "rec_texts": ["19.208,79"],
            "rec_scores": [0.4],
            "rec_boxes": [[10, 10, 50, 20]],
        },
        "parsing_res_list": [
            {"block_label": "table", "block_content": "", "block_bbox": [0, 0, 100, 100]}
        ],
        "table_res_list": [{"pred_html": "", "cell_box_list": [[0, 0, 100, 100]]}],
    }
    result = normalize_paddle_result(raw, source_sha256=source_sha, page_number=1, image_path=image)
    candidate = paddle_numeric_candidates(result)[0]
    assert candidate["candidate_status"] == "quarantined"
    assert candidate["review_reason"] == "ocr_score_below_gate"


def test_out_of_bounds_table_cell_is_retained_and_quarantined(tmp_path: Path):
    image, source_sha = _source(tmp_path)
    raw = {
        "width": 100,
        "height": 100,
        "overall_ocr_res": {
            "rec_texts": ["19.208,79"],
            "rec_scores": [0.99],
            "rec_boxes": [[10, 10, 50, 20]],
        },
        "parsing_res_list": [
            {"block_label": "table", "block_content": "", "block_bbox": [0, 0, 100, 100]}
        ],
        "table_res_list": [{"pred_html": "", "cell_box_list": [[0, 0, 150, 100]]}],
    }
    result = normalize_paddle_result(raw, source_sha256=source_sha, page_number=1, image_path=image)
    assert result["tables"][0]["cells"][0]["bbox_valid"] is False
    assert paddle_numeric_candidates(result)[0]["review_reason"] == "invalid_table_cell_bbox"


def test_bare_numeric_without_financial_context_is_not_financial_prediction(tmp_path: Path):
    image, source_sha = _source(tmp_path)
    raw = {
        "width": 100,
        "height": 100,
        "overall_ocr_res": {
            "rec_texts": ["2023"],
            "rec_scores": [0.99],
            "rec_boxes": [[10, 10, 50, 20]],
        },
        "parsing_res_list": [
            {"block_label": "table", "block_content": "", "block_bbox": [0, 0, 100, 100]}
        ],
        "table_res_list": [{"pred_html": "", "cell_box_list": [[0, 0, 100, 100]]}],
    }
    result = normalize_paddle_result(raw, source_sha256=source_sha, page_number=1, image_path=image)
    candidate = paddle_numeric_candidates(result)[0]
    assert candidate["prediction_class"] == "non_financial_numeric"
    assert candidate["candidate_status"] == "quarantined"
    assert candidate["review_reason"] == "non_financial_numeric_context"


def test_bare_numeric_with_financial_context_remains_review_candidate(tmp_path: Path):
    image, source_sha = _source(tmp_path)
    raw = {
        "width": 100,
        "height": 100,
        "overall_ocr_res": {
            "rec_texts": ["Cuenta corriente 200"],
            "rec_scores": [0.99],
            "rec_boxes": [[10, 10, 90, 20]],
        },
        "parsing_res_list": [
            {"block_label": "table", "block_content": "", "block_bbox": [0, 0, 100, 100]}
        ],
        "table_res_list": [{"pred_html": "", "cell_box_list": [[0, 0, 100, 100]]}],
    }
    result = normalize_paddle_result(raw, source_sha256=source_sha, page_number=1, image_path=image)
    candidate = paddle_numeric_candidates(result)[0]
    assert candidate["prediction_class"] == "financial_amount"
    assert candidate["candidate_status"] == "review_required"


@pytest.mark.parametrize(
    "text",
    [
        "Fecha del acuerdo plenario definitivo: 19.12.2023",
        "Superficie: 11,17 hectáreas",
    ],
)
def test_date_or_surface_decimal_is_not_financial_prediction(tmp_path: Path, text: str):
    image, source_sha = _source(tmp_path)
    raw = {
        "width": 100,
        "height": 100,
        "overall_ocr_res": {
            "rec_texts": [text],
            "rec_scores": [0.99],
            "rec_boxes": [[10, 10, 90, 20]],
        },
        "parsing_res_list": [
            {"block_label": "table", "block_content": "", "block_bbox": [0, 0, 100, 100]}
        ],
        "table_res_list": [{"pred_html": "", "cell_box_list": [[0, 0, 100, 100]]}],
    }
    result = normalize_paddle_result(raw, source_sha256=source_sha, page_number=1, image_path=image)
    candidates = paddle_numeric_candidates(result)
    assert candidates
    assert all(candidate["prediction_class"] == "non_financial_numeric" for candidate in candidates)
