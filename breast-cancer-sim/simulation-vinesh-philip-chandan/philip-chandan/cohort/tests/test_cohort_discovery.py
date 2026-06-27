"""Unit tests for cohort_discovery (mocked HTTP only)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cohort import COHORT_PATH
from cohort.cohort_discovery import (
    analyze_patient_imaging,
    audit_cohort,
    build_patient_report,
    fetch_pam50_subtype,
    find_longitudinal_patients,
    normalize_study_date,
    recommend_pair,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cohort_discovery"


def _load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _mock_fetcher(responses: dict[str, object]):
    def fetch(url: str) -> bytes:
        for key, payload in responses.items():
            if key in url:
                if isinstance(payload, bytes):
                    return payload
                return json.dumps(payload).encode("utf-8")
        raise AssertionError(f"Unexpected URL: {url}")

    return fetch


def test_normalize_study_date_formats() -> None:
    assert normalize_study_date("20020912") == "2002-09-12"
    assert normalize_study_date("2002-09-12") == "2002-09-12"
    assert normalize_study_date("") == "unknown-study"


def test_analyze_patient_longitudinal_luma() -> None:
    series = _load_fixture("series_a1ax.json")
    report = analyze_patient_imaging("TCGA-AR-A1AX", series_list=series)

    assert report["has_mri"] is True
    assert report["longitudinal"] is True
    assert report["study_dates"] == ["2002-09-12", "2003-09-24"]
    assert report["span_days"] == 377
    assert len(report["studies"]) == 2
    assert report["studies"][0]["best_series"]["SeriesDescription"] == "VIBRANT"


def test_analyze_patient_longitudinal_basal() -> None:
    series = _load_fixture("series_a1aq.json")
    report = analyze_patient_imaging("TCGA-AR-A1AQ", series_list=series)

    assert report["longitudinal"] is True
    assert report["study_dates"] == ["2001-11-21", "2003-05-07"]
    assert report["span_days"] == 532


def test_analyze_patient_no_mri() -> None:
    report = analyze_patient_imaging("TCGA-BH-A0BR", series_list=[])

    assert report["has_mri"] is False
    assert report["longitudinal"] is False
    assert report["series_count"] == 0


def test_fetch_pam50_subtype() -> None:
    clinical = _load_fixture("clinical_a1ax.json")
    fetcher = _mock_fetcher({"clinical-data": clinical})
    pam50 = fetch_pam50_subtype("TCGA-AR-A1AX", fetcher=fetcher, delay_seconds=0)

    assert pam50["pam50_raw"] == "BRCA_LumA"
    assert pam50["pam50_label"] == "Luminal A"


def test_build_patient_report_ok_primary() -> None:
    series = _load_fixture("series_a1ax.json")
    pam50 = {"pam50_raw": "BRCA_LumA", "pam50_label": "Luminal A"}
    report = build_patient_report(
        "TCGA-AR-A1AX",
        cohort_subtype="Luminal A",
        series_list=series,
        pam50=pam50,
        skip_cbio=True,
    )

    assert report["ok"] is True
    assert report["subtype_match"] is True
    assert report["issues"] == []


def test_build_patient_report_rev1_failures() -> None:
    report = build_patient_report(
        "TCGA-BH-A0BR",
        cohort_subtype="Luminal A",
        series_list=[],
        pam50={"pam50_raw": "BRCA_LumA", "pam50_label": "Luminal A"},
        skip_cbio=True,
    )

    assert report["ok"] is False
    assert "missing MRI on TCIA" in report["issues"]
    assert "not longitudinal" in report["issues"][1]


def test_audit_primary_rev2_passes() -> None:
    series_by_id = {
        "TCGA-AR-A1AX": _load_fixture("series_a1ax.json"),
        "TCGA-AR-A1AQ": _load_fixture("series_a1aq.json"),
    }
    clinical_by_id = {
        "TCGA-AR-A1AX": _load_fixture("clinical_a1ax.json"),
        "TCGA-AR-A1AQ": _load_fixture("clinical_a1aq.json"),
    }

    def fetcher(url: str) -> bytes:
        if "getSeries" in url and "TCGA-AR-A1AX" in url:
            return json.dumps(series_by_id["TCGA-AR-A1AX"]).encode("utf-8")
        if "getSeries" in url and "TCGA-AR-A1AQ" in url:
            return json.dumps(series_by_id["TCGA-AR-A1AQ"]).encode("utf-8")
        raise AssertionError(f"Unexpected URL: {url}")

    def cbio_fetcher(url: str) -> bytes:
        if "TCGA-AR-A1AX" in url:
            return json.dumps(clinical_by_id["TCGA-AR-A1AX"]).encode("utf-8")
        if "TCGA-AR-A1AQ" in url:
            return json.dumps(clinical_by_id["TCGA-AR-A1AQ"]).encode("utf-8")
        raise AssertionError(f"Unexpected URL: {url}")

    result = audit_cohort(COHORT_PATH, fetcher=fetcher, cbio_fetcher=cbio_fetcher)

    assert result["primary_ok"] is True
    assert len(result["reports"]) == 2


def test_find_longitudinal_luminal_a() -> None:
    patients = _load_fixture("patients.json")
    series = {
        "TCGA-AR-A1AX": _load_fixture("series_a1ax.json"),
        "TCGA-BH-A0BR": [],
        "TCGA-BH-A0BQ": _load_fixture("series_b0bq_single.json"),
    }
    clinical = {
        "TCGA-AR-A1AX": _load_fixture("clinical_a1ax.json"),
        "TCGA-BH-A0BQ": _load_fixture("clinical_b0bq.json"),
    }

    def fetcher(url: str) -> bytes:
        if "getPatient" in url:
            return json.dumps(patients).encode("utf-8")
        for tcga_id, payload in series.items():
            if "getSeries" in url and tcga_id in url:
                return json.dumps(payload).encode("utf-8")
        raise AssertionError(f"Unexpected URL: {url}")

    def cbio_fetcher(url: str) -> bytes:
        for tcga_id, payload in clinical.items():
            if tcga_id in url:
                return json.dumps(payload).encode("utf-8")
        return b"[]"

    matches = find_longitudinal_patients(
        "Luminal A",
        fetcher=fetcher,
        cbio_fetcher=cbio_fetcher,
    )

    assert [match["tcga_id"] for match in matches] == ["TCGA-AR-A1AX"]


def test_recommend_pair_selects_rev2_primaries() -> None:
    patients = _load_fixture("patients_recommend.json")
    series = {
        "TCGA-AR-A1AX": _load_fixture("series_a1ax.json"),
        "TCGA-OL-A66N": _load_fixture("series_ol_a66n.json"),
        "TCGA-AR-A1AQ": _load_fixture("series_a1aq.json"),
        "TCGA-A2-A04Q": _load_fixture("series_a2_a04q.json"),
    }
    clinical = {
        "TCGA-AR-A1AX": _load_fixture("clinical_a1ax.json"),
        "TCGA-OL-A66N": _load_fixture("clinical_ol_a66n.json"),
        "TCGA-AR-A1AQ": _load_fixture("clinical_a1aq.json"),
        "TCGA-A2-A04Q": _load_fixture("clinical_a2_a04q.json"),
    }

    def fetcher(url: str) -> bytes:
        if "getPatient" in url:
            return json.dumps(patients).encode("utf-8")
        for tcga_id, payload in series.items():
            if "getSeries" in url and tcga_id in url:
                return json.dumps(payload).encode("utf-8")
        raise AssertionError(f"Unexpected URL: {url}")

    def cbio_fetcher(url: str) -> bytes:
        for tcga_id, payload in clinical.items():
            if tcga_id in url:
                return json.dumps(payload).encode("utf-8")
        return b"[]"

    with patch("cohort.cohort_discovery.CBIO_DELAY_SECONDS", 0):
        result = recommend_pair(fetcher=fetcher, cbio_fetcher=cbio_fetcher)

    recommended = result["recommended"]
    assert recommended["Luminal A"]["tcga_id"] == "TCGA-AR-A1AX"
    assert recommended["Basal-like"]["tcga_id"] == "TCGA-AR-A1AQ"


def test_main_audit_exit_code_success() -> None:
    from cohort.cohort_discovery import main

    series_by_id = {
        "TCGA-AR-A1AX": _load_fixture("series_a1ax.json"),
        "TCGA-AR-A1AQ": _load_fixture("series_a1aq.json"),
    }

    def list_mr_series(tcga_id: str, collection: str = "TCGA-BRCA"):
        return series_by_id[tcga_id]

    with patch("cohort.cohort_discovery.list_mr_series", side_effect=list_mr_series), patch(
        "cohort.cohort_discovery.fetch_pam50_subtype",
        side_effect=lambda tcga_id, **kwargs: {
            "pam50_raw": "BRCA_LumA" if tcga_id == "TCGA-AR-A1AX" else "BRCA_Basal",
            "pam50_label": "Luminal A" if tcga_id == "TCGA-AR-A1AX" else "Basal-like",
        },
    ):
        assert main(["audit", "--json"]) == 0


def test_main_show_json(capsys: pytest.CaptureFixture[str]) -> None:
    from cohort.cohort_discovery import main

    series = _load_fixture("series_a1ax.json")

    def fetcher(url: str) -> bytes:
        if "getSeries" in url:
            return json.dumps(series).encode("utf-8")
        raise AssertionError(url)

    with patch("cohort.cohort_discovery.list_mr_series", return_value=series), patch(
        "cohort.cohort_discovery.fetch_pam50_subtype",
        return_value={"pam50_raw": "BRCA_LumA", "pam50_label": "Luminal A"},
    ):
        assert main(["show", "TCGA-AR-A1AX", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tcga_id"] == "TCGA-AR-A1AX"
    assert payload["span_days"] == 377
