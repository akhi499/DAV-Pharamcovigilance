"""API tests against the synthetic pipeline output (NLP in lexical-fallback mode)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(pipeline_paths):
    from api import main
    from pvsys.nlp import run as nlp_run
    from pvsys.nlp.search import reset_index

    nlp_run.run(pipeline_paths)
    main._db = None
    main._cached.cache_clear()
    reset_index()
    return TestClient(main.app)


@pytest.mark.parametrize("url", [
    "/api/health", "/api/overview", "/api/quality", "/api/demographics", "/api/trend",
    "/api/drugs/top?limit=5", "/api/reactions/top", "/api/drugs/search?q=dru",
    "/api/reactions/search?q=nau", "/api/signals/summary", "/api/signals?only_signals=false",
    "/api/nlp/status", "/api/nlp/categories", "/api/nlp/predictions", "/api/nlp/reviews",
    "/api/nlp/drug-profile?drug=DRUGX",
])
def test_endpoints_ok(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text


def test_overview_numbers(client):
    k = client.get("/api/overview").json()["kpis"]
    assert k["cases"] == 28 and k["probable_duplicates"] == 1


def test_signals_filters_and_disclaimer(client):
    body = client.get("/api/signals", params={"drug": "drugx", "only_signals": False}).json()
    assert body["total"] >= 1 and body["rows"][0]["drug"] == "DRUGX"
    assert "causality" in body["disclaimer"]
    body = client.get("/api/signals", params={"only_signals": False, "exclude_indication_bias": True}).json()
    assert all(not r["possible_indication_bias"] for r in body["rows"])


@pytest.mark.parametrize("url,code", [
    ("/api/signals?sort=drop table", 422),
    ("/api/signals?limit=0", 422),
    ("/api/signals?min_cases=1", 422),
    ("/api/drugs/search?q=a", 422),
    ("/api/drugs/NOPE/profile", 404),
    ("/api/signals/stratified?drug=DRUGX&pt_key=nausea&by=bogus", 422),
    ("/api/nlp/predictions?min_confidence=2", 422),
])
def test_input_validation(client, url, code):
    assert client.get(url).status_code == code


def test_stratified(client):
    r = client.get("/api/signals/stratified", params={"drug": "DRUGX", "pt_key": "nausea", "by": "sex"})
    assert r.status_code == 200
    strata = {s["stratum"]: s for s in r.json()["strata"]}
    assert strata["M"]["a"] == 5  # 4 background male cases + case 1002


def test_search_reports_mode(client):
    r = client.get("/api/nlp/search", params={"q": "nausea", "k": 3}).json()
    assert r["mode"] == "lexical-fallback" and r["results"][0]["review_status"] in ("unreviewed", "accepted",
                                                                                   "corrected", "rejected", "uncertain")


def test_concept_signals(client):
    r = client.get("/api/nlp/concept-signals", params={"q": "nausea", "min_similarity": 0.5})
    assert r.status_code == 200 and "terms" in r.json()


def test_review_post_flow(client):
    ok = client.post("/api/nlp/review", json={"pt_key": "Rash", "decision": "accepted", "reviewer": "tester"})
    assert ok.status_code == 200 and ok.json()["pt_key"] == "rash"
    bad = client.post("/api/nlp/review", json={"pt_key": "rash", "decision": "corrected", "reviewer": "t",
                                               "corrected_category": "xyz"})
    assert bad.status_code == 422
    missing = client.post("/api/nlp/review", json={"pt_key": "unknown term", "decision": "accepted", "reviewer": "t"})
    assert missing.status_code == 404
    rows = client.get("/api/nlp/predictions", params={"review_status": "accepted"}).json()["rows"]
    assert any(r["pt_key"] == "rash" for r in rows)


def test_category_filter_on_signals(client):
    r = client.get("/api/signals", params={"category": "gastrointestinal", "only_signals": False})
    assert r.status_code == 200
    assert all(row["nlp_category"] == "gastrointestinal" for row in r.json()["rows"])


def test_exclude_category_on_signals(client):
    allrows = client.get("/api/signals", params={"only_signals": False, "limit": 500}).json()
    some = {r["nlp_category"] for r in allrows["rows"]} - {None}
    cat = sorted(some)[0]
    r = client.get("/api/signals", params={"only_signals": False, "exclude_category": cat, "limit": 500}).json()
    assert all(row["nlp_category"] != cat for row in r["rows"])
    assert r["total"] < allrows["total"]
