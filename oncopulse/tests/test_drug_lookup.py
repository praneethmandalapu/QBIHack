import pytest
from unittest.mock import patch, MagicMock
import requests
from src.bio.drug_lookup import get_drug_interactions, query_dgidb_api

def test_query_dgidb_api_failure(mocker):
    # Mock requests.get to raise a Timeout or ConnectionError
    mocker.patch("requests.get", side_effect=requests.exceptions.Timeout("API Timeout"))
    
    # query should fail gracefully and return empty list
    results = query_dgidb_api("DNMT3B")
    assert results == []

def test_get_drug_interactions_cache_fallback(mocker):
    # Mock requests.get to fail so it has to fall back to local JSON cache
    mocker.patch("requests.get", side_effect=requests.exceptions.ConnectionError("API Down"))
    
    # We query for DNMT3B, which should fall back to cached drug database
    results = get_drug_interactions(["DNMT3B"])
    
    assert len(results) > 0
    assert any(d.drug_name == "Decitabine" for d in results)
    assert all(d.gene_symbol == "DNMT3B" for d in results)
