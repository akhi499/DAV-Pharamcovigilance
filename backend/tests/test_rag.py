"""Tests of the evidence-grounded question-answering (RAG) module.

They run on the synthetic FAERS quarter (lexical-fallback mode, no FDA PDFs, no CTCAE), so every expected
value is hand-checkable: DRUGX is co-reported with Nausea, DRUGZ with Rash, DRUGY with Headache.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from pvsys.nlp import generator as gen
from pvsys.nlp.kb import BM25, TfidfRetriever, split_faq, tokenize, top_k, window_chunks


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- knowledge base / retrieval units

def test_tokenize_stems_consistently():
    assert tokenize("Duplicates") == tokenize("duplicate") == tokenize("duplicated")
    assert tokenize("ages") == tokenize("age")
    assert "the" not in tokenize("the reports")
    assert "not" in tokenize("does not establish")  # negation is kept


def test_bm25_ranks_relevant_document_first():
    docs = ["patient age is reported in days by some manufacturers", "the drug dictionary is based on SPL",
            "case version numbers replace ISR numbers"]
    for cls in (BM25, TfidfRetriever):
        r = cls(docs)
        hits = top_k(r.scores("why is the age so large, reported in days?"), 2)
        assert hits[0].index == 0
        assert top_k(r.scores("zzzz qqqq"), 3) == []  # no shared term -> nothing retrieved


def test_split_faq_and_windows():
    pages = ["FAQ TITLE\n1. Why is A? \nANSWER: Because of B.\n2. What is C?\nANSWER: C is D.\n  1  ",
             "3. Does E exist?\nANSWER: Yes, E exists.\n17. Not a question number in sequence"]
    faq = split_faq(pages)
    assert [q["number"] for q in faq] == [1, 2, 3]
    assert faq[0]["question"] == "Why is A?" and faq[0]["answer"] == "Because of B."
    assert faq[2]["page"] == 2
    win = window_chunks([" ".join(f"w{i}" for i in range(300))], window=140, stride=100)
    assert len(win) == 3 and win[1]["text"].startswith("w100 ")


# --------------------------------------------------------------------------- generator verifier

EV = "[S1] drug=SEMAGLUTIDE, a=74, ror=12.5, ror_lower=9.80, ror_upper=15.9, ic025=-1.40\n[D1] note: not causality"


@pytest.mark.parametrize("text,ok", [
    ("SEMAGLUTIDE and pancreatitis were co-reported in 74 cases, ROR 12.5 (9.80-15.9) [S1].", True),
    ("This does not prove that semaglutide causes pancreatitis [S1, D1].", True),
    ("Semaglutide causes pancreatitis [S1].", False),        # causal claim
    ("There were 75 cases [S1].", False),                    # number not in evidence
    ("There were 74 cases [S7].", False),                    # unknown citation
    ("There were 74 cases.", False),                         # no citation
    ("A 5% risk of pancreatitis [S1].", False),              # invented incidence
    ("", False),
    ("Its definition is \u201cnot causality\u201d [D1].", True),          # verbatim quote is allowed
    ("FDA says \u201csemaglutide causes harm\u201d [D1].", False),        # invented quote is rejected
])
def test_verifier(text, ok):
    assert (gen.verify(text, EV, {"S1", "D1"}) == []) is ok


def test_generator_unavailable_without_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_GENERATOR_DIR", str(tmp_path / "none"))
    st = gen.status()
    assert st["available"] is False and st["weights_present"] is False
    assert gen.summarise("q", [{"id": "S1", "kind": "statistic", "a": 1}])["status"] == "unavailable"


def test_generator_mechanics_with_tiny_random_model(tmp_path, monkeypatch):
    """A randomly initialised tiny GPT-2 exercises loading, prompting, decoding and verification offline.

    Its output is gibberish by construction, so the verifier must reject it - which is the point."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    trainer = tokenizers.trainers.WordLevelTrainer(special_tokens=["[UNK]", "[EOS]"])
    tok.train_from_iterator(["evidence question summary cases ror drug S1 74 12.5 not causes the of"] * 5, trainer)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
    d = tmp_path / "generator"
    fast.save_pretrained(d)
    cfg = transformers.GPT2Config(vocab_size=fast.vocab_size, n_positions=2048, n_embd=16, n_layer=1, n_head=2,
                                  eos_token_id=fast.eos_token_id)
    transformers.GPT2LMHeadModel(cfg).save_pretrained(d)
    (d / "provenance.json").write_text(json.dumps({"model_id": "test/tiny-random-gpt2", "revision": "none"}))
    monkeypatch.setenv("PV_GENERATOR_DIR", str(d))
    gen.get_generator.cache_clear()
    try:
        out = gen.summarise("Is DRUGX associated with nausea?", [{"id": "S1", "kind": "statistic", "drug": "DRUGX", "a": 74}],
                            "reported together in 74 cases")
    finally:
        gen.get_generator.cache_clear()
    assert out["status"] in {"verified", "rejected"}
    assert out["model_id"] == "test/tiny-random-gpt2" and out["decoding"] == "greedy"
    if out["status"] == "rejected":
        assert out["text"] is None and out["problems"]


# --------------------------------------------------------------------------- engine on the synthetic pipeline

@pytest.fixture(scope="module")
def engine(pipeline_paths):
    from pvsys.nlp import kb, rag
    from pvsys.nlp import run as nlp_run
    from pvsys.nlp.search import reset_index

    before = _sha(pipeline_paths.processed / "signals.parquet")
    if not (pipeline_paths.nlp / "predictions.parquet").exists():
        nlp_run.run(pipeline_paths)
    manifest = kb.build(pipeline_paths)
    reset_index()
    rag.reset_engine()
    eng = rag.RAGEngine(pipeline_paths)
    return eng, manifest, before, pipeline_paths


def _text(res):
    return " ".join(f"{s['text']} [{', '.join(s['cites'])}]" if s["cites"] else s["text"] for s in res["answer"])


def test_kb_without_fda_documents_still_builds(engine):
    _, manifest, _, _ = engine
    assert manifest["by_source"] == {"PROJECT": manifest["n_chunks"]}  # synthetic zip has no PDFs, no CTCAE file
    assert len(manifest["content_sha256"]) == 16


def test_pair_question_returns_stored_statistics(engine):
    eng, _, _, paths = engine
    res = eng.answer("Is DRUGX associated with nausea?")
    assert res["status"] == "answered" and res["intent"] == "pair"
    u = res["understanding"]
    assert [d["drug"] for d in u["drugs"]] == ["DRUGX"] and u["drugs"][0]["match"] == "exact"
    assert u["reactions"][0]["pt_key"] == "nausea" and u["reactions"][0]["match"] == "exact"
    s1 = next(e for e in res["evidence"] if e["id"] == "S1")
    import duckdb

    row = duckdb.sql(f"SELECT a, ror FROM '{paths.processed / 'signals.parquet'}' "
                     "WHERE drug='DRUGX' AND pt_key='nausea'").fetchone()
    assert (s1["a"], s1["ror"]) == (row[0], pytest.approx(row[1]))
    assert any("does not show that the drug causes" in s["text"] for s in res["answer"])  # caveat always attached
    caveat = next(s for s in res["answer"] if "does not show" in s["text"])
    assert caveat["cites"], "caveat must cite its sources"
    # every number in the composed answer is traceable to the evidence
    assert gen.verify(_text(res), gen.evidence_block(res["evidence"]), {e["id"] for e in res["evidence"]},
                      {"2025", "95"}) == []
    assert res["provenance"]["reaction_linker"].startswith("TF-IDF")  # graceful fallback in tests


def test_typo_and_case_tolerant_drug_linking(engine):
    eng, _, _, _ = engine
    res = eng.answer("drugxx nausea")
    assert res["understanding"]["drugs"][0]["drug"] == "DRUGX"
    assert res["understanding"]["drugs"][0]["match"] == "fuzzy"


def test_untested_pair_reports_count_not_ratio(engine):
    eng, _, _, _ = engine
    res = eng.answer("Is DRUGY associated with rash?")
    s1 = res["evidence"][0]
    assert s1["tested"] is False and s1["a"] == 0
    assert "no case reports" in res["answer"][0]["text"] and "not evidence of safety" in res["answer"][0]["text"]


def test_drug_and_reaction_only_questions(engine):
    eng, _, _, _ = engine
    d = eng.answer("Tell me about DRUGZ")
    assert d["intent"] == "drug" and any(e["kind"] == "aggregate" for e in d["evidence"])
    r = eng.answer("Which drugs are reported with rash?")
    assert r["intent"] == "reaction" and r["understanding"]["reactions"][0]["pt_key"] == "rash"


def test_documentation_question_quotes_sources(engine):
    eng, _, _, _ = engine
    res = eng.answer("What does disproportionality mean? Does it prove causation?")
    assert res["intent"] == "documentation" and res["status"] == "answered"
    assert all(e["kind"] == "document" for e in res["evidence"])
    assert all(s["cites"] for s in res["answer"])


def test_abstains_when_nothing_relevant(engine):
    eng, _, _, _ = engine
    res = eng.answer("what is the weather today")
    assert res["status"] == "abstained" and res["evidence"] == []


def test_input_validation(engine):
    eng, _, _, _ = engine
    with pytest.raises(ValueError):
        eng.answer("   ")
    with pytest.raises(ValueError):
        eng.answer("x" * 301)


def test_rag_evaluation_and_independence(engine):
    from pvsys.nlp.rag_eval import evaluate_rag, ranking_metrics

    _, _, before, paths = engine
    m = ranking_metrics(np.array([1, 2, np.inf, 1.0]))
    assert m["recall_at_1"] == 0.5 and m["mrr"] == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    ev = evaluate_rag(paths, n_pairs=3)
    assert ev["faq_retrieval"]["status"] == "unavailable"     # no FDA FAQ in the synthetic zip
    assert ev["reaction_linking"]["status"] == "unavailable"  # no CTCAE file
    e2e = ev["end_to_end"]
    assert e2e["status"] == "ok" and e2e["questions"] >= 15
    assert e2e["exact_retrieval_generic_templates"] == 1.0    # tiny vocabulary: every templated pair is recovered
    assert e2e["deterministic_answers_passing_grounding_check"] == 1.0
    assert (paths.nlp / "evaluation_rag.json").exists()
    assert _sha(paths.processed / "signals.parquet") == before  # QA never alters statistical outputs


def test_api_rag_endpoints(engine):
    from fastapi.testclient import TestClient

    from api import main
    from pvsys.nlp.rag import reset_engine

    main._db = None
    main._cached.cache_clear()
    main._ask_cached.cache_clear()
    reset_engine()
    c = TestClient(main.app)
    r = c.post("/api/rag/ask", json={"question": "Is DRUGX associated with nausea?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered" and body["evidence"] and "disclaimer" in body
    assert body["generated_summary"]["status"] in {"unavailable", "not_requested"}
    assert c.post("/api/rag/ask", json={"question": "Is DRUGX associated with nausea?"}).json() == body  # cached
    assert c.post("/api/rag/ask", json={"question": "a"}).status_code == 422
    assert c.post("/api/rag/ask", json={"question": "12345 678"}).status_code == 422
    assert c.post("/api/rag/ask", json={"question": "q" * 301}).status_code == 422
    st = c.get("/api/rag/status").json()
    assert st["available"] and st["examples"] and st["generator"]["available"] is False
    assert c.get("/api/rag/evaluation").status_code == 200
    assert c.post("/api/admin/reload").status_code == 200
