"""Unit tests for download_tcia series selection and date normalization."""

from __future__ import annotations

import pytest

from download_tcia import (
    DEFAULT_COLLECTION,
    group_series_by_study,
    list_mr_series,
    normalize_study_date,
    pick_series,
)


def test_normalize_study_date_formats() -> None:
    assert normalize_study_date("2002-09-12") == "2002-09-12"
    assert normalize_study_date("20030912") == "2003-09-12"
    assert normalize_study_date("09-24-2003") == "2003-09-24"
    assert normalize_study_date("") == "unknown-study"


def test_pick_series_prefers_contrast() -> None:
    series_list = [
        {
            "SeriesDescription": "LOCALIZER",
            "ImageCount": 10,
            "SeriesInstanceUID": "1",
            "StudyDate": "2002-09-12",
        },
        {
            "SeriesDescription": "VIBRANT",
            "ImageCount": 352,
            "SeriesInstanceUID": "2",
            "StudyDate": "2002-09-12",
        },
    ]

    chosen = pick_series(series_list, prefer_contrast=True)

    assert chosen is not None
    assert chosen["SeriesInstanceUID"] == "2"


def test_group_series_by_study_normalizes_dates() -> None:
    grouped = group_series_by_study(
        [
            {"StudyDate": "09-24-2003", "SeriesInstanceUID": "a"},
            {"StudyDate": "2003-09-24", "SeriesInstanceUID": "b"},
        ]
    )

    assert set(grouped) == {"2003-09-24"}
    assert len(grouped["2003-09-24"]) == 2


@pytest.mark.integration
def test_list_mr_series_primary_cohort_patient() -> None:
    series_list = list_mr_series("TCGA-AR-A1AX", collection=DEFAULT_COLLECTION)

    assert series_list
    study_dates = {normalize_study_date(str(entry["StudyDate"])) for entry in series_list}
    assert "2002-09-12" in study_dates
    assert "2003-09-24" in study_dates

    baseline = group_series_by_study(series_list)["2002-09-12"]
    chosen = pick_series(baseline, prefer_contrast=True)
    assert chosen is not None
    assert "VIBRANT" in str(chosen.get("SeriesDescription", "")).upper()
