"""Tests of external-evidence ingestion, chunking, indexing, retrieval, evaluation and the QA integration.

No network: a fake fetcher serves recorded-shape openFDA / DailyMed / FDA / PubMed payloads.
"""

from __future__ import annotations

import json
from urllib.parse import unquote

import numpy as np
import pytest

from pvsys.evidence import sources as src
from pvsys.evidence.chunking import split_words, tag_drugs

LABEL_DRUGX = {
    "set_id": "set-drugx-mfr", "id": "spl-1", "version": "7", "effective_time": "20260130",
    "openfda": {"brand_name": ["Drugxa"], "generic_name": ["DRUGX"], "manufacturer_name": ["Acme Pharma"]},
    "boxed_warning": ["WARNING: SEVERE LIVER INJURY. Drugxa can cause hepatitis; monitor liver tests."],
    "warnings_and_cautions": ["5.1 Hepatitis: Serious hepatitis has been reported. 5.2 Hypotension: monitor blood pressure."],
    "adverse_reactions": ["The most common adverse reactions (>=5%) were nausea, headache and rash."],
    "contraindications": ["Do not use in patients with known hypersensitivity to drugx."],
}
LABEL_REPACK = {**LABEL_DRUGX, "set_id": "set-drugx-repack", "version": "1",
                "openfda": {**LABEL_DRUGX["openfda"], "manufacturer_name": ["REMEDYREPACK INC."]}}
PUBMED_XML = """<?xml version="1.0"?><!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle//EN" "x.dtd">
<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><Journal><Title>Drug Safety</Title>
<JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal><ArticleTitle>Safety of drugx: a review</ArticleTitle>
<Abstract><AbstractText Label="RESULTS">Nausea was the most frequent adverse event of drugx.</AbstractText></Abstract>
<PublicationTypeList><PublicationType>Review</PublicationType></PublicationTypeList></Article></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
DSC_INDEX = '<a href="/drugs/drug-safety-communications/fda-warns-drugx-liver">x</a><a href="/drugs/drug-safety-and-availability/fda-drug-safety-podcasts">p</a>'
DSC_PAGE = """<meta property="article:published_time" content="Thu, 03/19/2026 - 09:05" />
<article id="main-content"><nav><ul><li>Menu item</li></ul></nav><h1>FDA warns about liver injury with Drugxa (drugx)</h1>
<p>FDA is warning that the medicine drugx can cause serious liver injury.</p><p>Patients should report symptoms.</p>
<p>Health care professionals should check liver tests before starting drugx and periodically during treatment, and stop the
medicine if liver injury is suspected. FDA reviewed reports submitted to the FDA Adverse Event Reporting System.</p>
Content current as of: 03/19/2026</article>"""


class FakeFetcher:
    def __init__(self, fail_pubmed: bool = False):
        self.n_requests = 0
        self.fail_pubmed = fail_pubmed

    def get(self, url: str):
        self.n_requests += 1
        u = unquote(url)
        if "api.fda.gov" in u:
            if '"DRUGX"' in u:
                return 200, json.dumps({"results": [LABEL_REPACK, LABEL_DRUGX]})
            return 404, json.dumps({"error": {"code": "NOT_FOUND"}})
        if "dailymed" in u:
            return 200, json.dumps({"data": [{"setid": "set-drugx-repack", "spl_version": 1, "title": "DRUGX [REMEDYREPACK INC.]"},
                                             {"setid": "set-drugx-mfr", "spl_version": 7, "title": "DRUGXA [ACME PHARMA]"}]})
        if "esearch" in u:
            if self.fail_pubmed:
                return 500, ""
            return 200, json.dumps({"esearchresult": {"idlist": ["123"] if "drugx" in u else []}})
        if "efetch" in u:
            return 200, PUBMED_XML
        if u.endswith("drug-safety-communications"):
            return 200, DSC_INDEX
        if "fda-warns-drugx-liver" in u:
            return 200, DSC_PAGE
        return 404, ""

    def json(self, url: str):
        code, text = self.get(url)
        try:
            return code, json.loads(text) if code == 200 else None
        except json.JSONDecodeError:
            return code, None


# --------------------------------------------------------------------------- parsing units

def test_parsers():
    recs = src.parse_pubmed_xml(PUBMED_XML)
    assert recs[0]["pmid"] == "123" and recs[0]["doi"] == "10.1/x" and recs[0]["year"] == "2024"
    assert recs[0]["abstract"].startswith("RESULTS: Nausea") and recs[0]["url"].endswith("/123/")
    page = src.parse_dsc_page(DSC_PAGE, "https://www.fda.gov/x")
    assert page["title"].startswith("FDA warns about liver injury") and "Menu item" not in page["text"]
    assert page["published"].startswith("Thu, 03/19/2026") and "Content current" not in page["text"]
    assert src.dsc_links(DSC_INDEX) == ["/drugs/drug-safety-communications/fda-warns-drugx-liver"]


def test_label_set_prefers_manufacturer_with_most_revisions():
    cands = [{"setid": "a", "spl_version": 30, "title": "X [REMEDYREPACK INC.]"}, {"setid": "b", "spl_version": 7, "title": "X [ACME]"},
             {"setid": "c", "spl_version": 3, "title": "X [OTHER]"}]
    assert src.choose_setid(cands, [{"set_id": "a"}, {"set_id": "b"}, {"set_id": "c"}]) == "b"


def test_chunking_and_drug_tagging():
    words = " ".join(f"Sentence {i} has several words in it." for i in range(80))
    chunks = split_words(words, size=60, overlap=10)
    assert len(chunks) > 3 and all(len(c.split()) <= 75 for c in chunks)
    assert tag_drugs("FDA warns about Drugxa use", {"DRUGXA": "DRUGX", "OTHER": "OTHER"}) == ["DRUGX"]
    assert tag_drugs("drugxanthine", {"DRUGX": "DRUGX"}) == []  # whole words only


# --------------------------------------------------------------------------- full offline build on the synthetic pipeline

@pytest.fixture(scope="module")
def evidence(pipeline_paths):
    from pvsys.evidence import index as ix
    from pvsys.nlp import kb, rag
    from pvsys.nlp import run as nlp_run
    from pvsys.nlp.search import reset_index

    if not (pipeline_paths.nlp / "predictions.parquet").exists():
        nlp_run.run(pipeline_paths)
    if not (pipeline_paths.nlp / "kb_chunks.parquet").exists():
        kb.build(pipeline_paths)
    sig_before = (pipeline_paths.processed / "signals.parquet").read_bytes()
    man = src.fetch_all(pipeline_paths, drugs=["DRUGX", "DRUGZ"], fetcher=FakeFetcher())
    build = ix.build_index(pipeline_paths)
    reset_index()
    rag.reset_engine()
    yield pipeline_paths, man, build, sig_before  # conftest points PV_EMBED_DIR at an empty folder: BM25-only
    ix.reset_index()


def test_fetch_manifest_and_documents(evidence):
    paths, man, _, _ = evidence
    assert man["stats"]["labels"] == {"ok": 1, "not_found": 1, "error": 0, "cached": 0}
    assert man["stats"]["dsc"]["ok"] == 1 and man["failures"] == []
    import pandas as pd

    docs = pd.read_parquet(src.evidence_dir(paths) / "documents.parquet")
    lab = docs[docs.source_type == "label"].iloc[0]
    assert lab["identifier"] == "set-drugx-mfr" and lab["version"] == "7" and lab["publisher"] == "Acme Pharma"
    assert "DRUGXA" in lab["aliases"] and lab["url"].endswith("setid=set-drugx-mfr")
    assert set(docs.source_type) == {"label", "literature", "regulatory"}
    again = src.fetch_all(paths, drugs=["DRUGX"], fetcher=FakeFetcher())  # cached: no label refetch
    assert again["stats"]["labels"]["cached"] == 1


def test_fetch_records_failures_without_aborting(pipeline_paths, tmp_path):
    import os

    old = os.environ["PV_DATA_DIR"]
    os.environ["PV_DATA_DIR"] = str(tmp_path)
    try:
        from pvsys.config import get_paths

        p = get_paths()
        (p.processed).mkdir(parents=True)
        man = src.fetch_all(p, drugs=["DRUGX"], fetcher=FakeFetcher(fail_pubmed=True))
        assert man["stats"]["pubmed"]["error"] == 1 and man["stats"]["labels"]["ok"] == 1
        assert any(f["source"] == "pubmed" for f in man["failures"])
    finally:
        os.environ["PV_DATA_DIR"] = old


def test_index_bm25_only_and_search(evidence):
    from pvsys.evidence.index import EvidenceIndex

    paths, _, build, _ = evidence
    assert build["retrieval_modes"] == ["bm25"] and build["embedding"]["status"] == "unavailable"
    idx = EvidenceIndex(paths)
    assert idx.mode == "bm25-only" and "DRUGX" in idx.indexed_drugs
    res = idx.search("What is the boxed warning for drugx?", k=3, drugs=["DRUGX"], mode="hybrid")
    assert res["mode"] == "bm25"  # graceful degradation reported
    assert res["hits"][0]["section_key"] == "boxed_warning"
    reg = idx.chunks[idx.chunks.source_type == "regulatory"]
    assert (reg["drugs"] == "DRUGX").all()  # communication tagged via brand alias / generic name
    m = idx.label_mentions("DRUGX", ["hepatitis"])
    assert m and m[0]["section_key"] == "boxed_warning"
    assert idx.search("nothing", k=3, drugs=["NOT A DRUG"])["hits"] == []
    with pytest.raises(ValueError):
        idx.search("  ")


def test_dense_index_with_tiny_random_model(evidence, tmp_path, monkeypatch):
    """Exercises the vector path offline with a randomly initialised tiny BERT (mechanics, not quality)."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from pvsys.evidence import embed
    from pvsys.evidence import index as ix

    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.train_from_iterator(["drugx hepatitis nausea warning liver label boxed"] * 3,
                            tokenizers.trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    d = tmp_path / "tiny-bert"
    fast.save_pretrained(d)
    transformers.BertModel(transformers.BertConfig(vocab_size=fast.vocab_size, hidden_size=16, num_hidden_layers=1,
                                                   num_attention_heads=2, intermediate_size=32)).save_pretrained(d)
    monkeypatch.setenv("PV_EMBED_DIR", str(d))
    embed.get_embedder.cache_clear()
    try:
        paths = evidence[0]
        build = ix.build_index(paths, rebuild_documents=False)
        assert build["retrieval_modes"] == ["hybrid", "dense", "bm25"] and build["embedding"]["dim"] == 16
        vec = np.load(src.evidence_dir(paths) / "embeddings.npy")
        assert vec.shape == (build["chunks"], 16)
        idx = ix.EvidenceIndex(paths)
        for mode in ("dense", "hybrid", "bm25"):
            r = idx.search("drugx liver warning", k=3, drugs=["DRUGX"], mode=mode)
            assert r["mode"] == mode and r["hits"]
        assert r["hits"][0]["bm25"] is not None
    finally:
        monkeypatch.setenv("PV_EMBED_DIR", str(tmp_path / "none"))
        embed.get_embedder.cache_clear()
        ix.build_index(evidence[0], rebuild_documents=False)  # back to the BM25-only index for later tests
        ix.reset_index()


def test_evaluation_runs_and_records_protocol(evidence):
    from pvsys.evidence.evaluate import classify_boxed, evaluate, sufficiency_threshold

    paths = evidence[0]
    r = evaluate(paths, skip_generation=True)
    assert r["drugs_with_label"] == 1 and r["retrieval"]["modes"] == ["bm25"]
    assert r["sufficiency"]["status"] == "unavailable"  # no vectors -> no calibration, default kept
    assert sufficiency_threshold(paths) == pytest.approx(0.60)
    assert r["generation"] == {"status": "skipped"}
    assert r["concordance"]["status"] == "ok"
    assert (src.evidence_dir(paths) / "evaluation_evidence.json").exists()
    assert classify_boxed("The label has a boxed warning about liver injury [L1].", "verified") == "yes"
    assert classify_boxed("Drugx does not have a boxed warning [L1].", "verified") == "no"
    assert classify_boxed(None, "insufficient") == "no_answer"


def test_qa_engine_adds_established_evidence(evidence):
    from pvsys.nlp.rag import RAGEngine

    paths, _, _, sig_before = evidence
    eng = RAGEngine(paths)
    res = eng.answer("Is DRUGX associated with nausea?")
    est = res["established"]
    assert est["status"] in ("answered", "partial")
    lm = est["label_mentions"][0]
    assert lm["drug"] == "DRUGX" and lm["found"] and "Adverse Reactions" in lm["sections"]
    ids = {e["id"] for e in est["evidence"]}
    assert all(c in ids for s in est["sentences"] for c in s["cites"])
    assert all(e["url"] and e["source_type"] in ("label", "literature", "regulatory") for e in est["evidence"])
    assert est["generated"]["status"] in ("not_requested", "unavailable")
    # FAERS part is unchanged and still carries the causality caveat
    assert any("does not show that the drug causes" in s["text"] for s in res["answer"])
    none = eng.answer("Is DRUGZ associated with rash?")["established"]
    assert none["status"] == "insufficient" and none["not_indexed"] == ["DRUGZ"]
    assert eng.answer("what is the weather today")["established"]["status"] == "not_applicable"
    assert (paths.processed / "signals.parquet").read_bytes() == sig_before  # evidence never touches statistics


def test_external_summary_unavailable_without_generator(monkeypatch, tmp_path):
    from pvsys.nlp import generator as g

    monkeypatch.setenv("PV_GENERATOR_DIR", str(tmp_path / "none"))
    assert g.summarise_external("q", [{"id": "L1", "kind": "external", "text": "x"}])["status"] == "unavailable"
    ev = [{"id": "L1", "kind": "external", "title": "Drugxa label", "section": "Warnings", "text": "Serious hepatitis has been reported."}]
    ok = g.sentence_support("The label warns that serious hepatitis has been reported [L1].", ev)
    bad = g.sentence_support("Pregnant women should avoid drugx [L1].", ev)
    assert ok[0]["support"] >= g.MIN_SENTENCE_SUPPORT > bad[0]["support"]


def test_api_evidence_endpoints(evidence):
    from fastapi.testclient import TestClient

    from api import main
    from pvsys.nlp.rag import reset_engine

    main._db = None
    main._ask_cached.cache_clear()
    reset_engine()
    c = TestClient(main.app)
    st = c.get("/api/evidence/status").json()
    assert st["available"] and st["index"]["documents"] == 3 and st["sufficiency_threshold"] == pytest.approx(0.6)
    docs = c.get("/api/evidence/documents", params={"drug": "drugx"}).json()
    assert docs["total"] == 3 and {r["source_type"] for r in docs["rows"]} == {"label", "literature", "regulatory"}
    assert c.get("/api/evidence/documents", params={"source_type": "nope"}).status_code == 422
    s = c.get("/api/evidence/search", params={"q": "boxed warning", "drug": "DRUGX"}).json()
    assert s["hits"] and s["index_mode"] == "bm25-only"
    assert c.get("/api/evidence/evaluation").status_code == 200
    body = c.post("/api/rag/ask", json={"question": "Is DRUGX associated with nausea?"}).json()
    assert body["established"]["label_mentions"][0]["found"] is True
    assert c.post("/api/rag/ask", json={"question": "Is DRUGX associated with nausea?", "external": False}).json()[
        "established"] == {"status": "not_requested"}


def test_mcnemar_and_wilson():
    from pvsys.evidence.compare_generators import mcnemar_exact, wilson

    r = mcnemar_exact([True] * 10, [False] * 10)
    assert r["only_first_correct"] == 10 and r["p_value"] == pytest.approx(2 / 2 ** 10)
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    lo, hi = wilson(39, 40)
    assert lo == pytest.approx(0.8712, abs=1e-3) and hi == pytest.approx(0.9956, abs=1e-3)


def test_generator_comparison_mechanics(evidence, tmp_path):
    """Runs the full comparison with a tiny random GPT-2 on the synthetic index (mechanics, not quality)."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from pvsys.evidence.compare_generators import compare

    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.train_from_iterator(["the label boxed warning drugx L1 yes no"] * 3,
                            tokenizers.trainers.WordLevelTrainer(special_tokens=["[UNK]", "[EOS]"]))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
    d = tmp_path / "generator-tiny"
    fast.save_pretrained(d)
    transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=fast.vocab_size, n_positions=2048, n_embd=16,
                                                         n_layer=1, n_head=2, eos_token_id=fast.eos_token_id)).save_pretrained(d)
    paths = evidence[0]
    r = compare([d], paths)
    m = r["models"][0]
    assert m["status"] == "ok" and m["parameters_millions"] >= 0
    assert r["task_a_gold_counts"] == {"yes": 1, "no": 0}  # DRUGX's label has a boxed warning
    for t in ("task_a", "task_b"):
        s = m[t]
        assert s["n"] == 1 and 0 <= s["system_accuracy"] <= 1 and "p_value" in s["mcnemar_extractive_vs_generative"]
    assert len(m["items_task_a"]) == 1 and r["inputs"]["task_a"][0]["chunk_ids"]
    assert (src.evidence_dir(paths) / "generator_comparison.json").exists()


def _tiny_bert(d, words: str, num_labels: int | None = None, id2label: dict | None = None):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.train_from_iterator([words] * 3, tokenizers.trainers.WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                                cls_token="[CLS]", sep_token="[SEP]")
    fast.save_pretrained(d)
    cfg = dict(vocab_size=fast.vocab_size, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
               intermediate_size=32)
    if num_labels is None:
        transformers.BertModel(transformers.BertConfig(**cfg)).save_pretrained(d)
    else:
        extra = {"id2label": id2label, "label2id": {v: k for k, v in id2label.items()}} if id2label else {}
        transformers.BertForSequenceClassification(transformers.BertConfig(num_labels=num_labels, **cfg, **extra)
                                                   ).save_pretrained(d)


def test_routing_and_reranking_modes(evidence, tmp_path, monkeypatch):
    """dense+route / rerank / rerank+route with tiny random models (mechanics, not quality)."""
    pytest.importorskip("torch")
    from pvsys.evidence import embed, rerank
    from pvsys.evidence import index as ix

    words = "drugx hepatitis nausea warning liver label boxed who should not take pregnancy"
    _tiny_bert(tmp_path / "emb", words)
    monkeypatch.setenv("PV_EMBED_DIR", str(tmp_path / "emb"))
    embed.get_embedder.cache_clear()
    rerank.get_reranker.cache_clear()
    try:
        paths = evidence[0]
        ix.build_index(paths, rebuild_documents=False)
        idx = ix.EvidenceIndex(paths)
        r = idx.search("Does drugx have a boxed warning?", k=3, drugs=["DRUGX"], mode="rerank")
        assert r["mode"] == "dense" and "re-ranker" in r["note"]  # no re-ranker model: honest fallback
        r = idx.search("Does drugx have a boxed warning?", k=3, drugs=["DRUGX"], mode="dense+route")
        assert r["routed_sections"] == ["boxed_warning"] and r["top_dense_similarity"] is not None
        idx.route_weight["dense+route"] = 10.0  # a large bonus puts the routed section first
        r = idx.search("Does drugx have a boxed warning?", k=3, drugs=["DRUGX"], mode="dense+route")
        assert r["hits"][0]["section_key"] == "boxed_warning"
        _tiny_bert(tmp_path / "rr", words, num_labels=1)
        monkeypatch.setenv("PV_RERANKER_DIR", str(tmp_path / "rr"))
        rerank.get_reranker.cache_clear()
        idx = ix.EvidenceIndex(paths)
        for mode in ("rerank", "rerank+route"):
            r = idx.search("drugx liver warning", k=3, drugs=["DRUGX"], mode=mode)
            assert r["mode"] == mode and r["hits"] and r["note"] is None
        with pytest.raises(ValueError):
            idx.search("drugx", mode="nonsense")
    finally:
        monkeypatch.setenv("PV_EMBED_DIR", str(tmp_path / "none"))
        monkeypatch.setenv("PV_RERANKER_DIR", str(tmp_path / "none"))
        embed.get_embedder.cache_clear()
        rerank.get_reranker.cache_clear()
        ix.build_index(evidence[0], rebuild_documents=False)
        ix.reset_index()


def test_route_sections():
    from pvsys.evidence.rerank import route_sections

    assert route_sections("Who should not take Humira?") == ("contraindications",)
    assert route_sections("Does X have a boxed warning?") == ("boxed_warning",)
    assert route_sections("Is it safe to take X while pregnant?") == ("pregnancy", "use_in_specific_populations")
    assert route_sections("What happens if someone takes too much X?") == ("overdosage",)
    assert route_sections("Can X be combined with other medicines?") == ("drug_interactions",)
    assert route_sections("What are the warnings for X?") == ("warnings_and_cautions", "warnings", "precautions")
    assert route_sections("Tell me about X") == ()


def test_structure_check():
    from pvsys.nlp.checks import boxed_polarity, structure_problems

    boxed = [{"id": "L1", "section_key": "boxed_warning"}]
    other = [{"id": "L1", "section_key": "adverse_reactions"}]
    deny = "No, there is no boxed warning for capecitabine [L1]."
    assert boxed_polarity(deny) == "no" and boxed_polarity("Yes, it has a boxed warning [L1].") == "yes"
    assert boxed_polarity("Nausea is common [L1].") is None
    assert structure_problems(deny, boxed) and not structure_problems(deny, other)
    assert structure_problems("Yes, X carries a boxed warning [L1].", other)
    assert not structure_problems("Nausea is common [L1].", boxed)


def test_entailment_check_with_fake_model():
    from pvsys.nlp.checks import entailment_check

    class Fake:
        def probs(self, premises, hypotheses):
            return [(0.9, 0.05) if h.startswith("No") else (0.02, 0.9) for h in hypotheses]

    items = [{"id": "L1", "section": "BOXED WARNING", "text": "WARNING: serious infections."}]
    r = entailment_check("No, there is no boxed warning [L1]. Serious infections occur [L1].", items, Fake())
    assert r["status"] == "ok" and len(r["problems"]) == 1 and r["sentences"][0]["contradiction"] == 0.9
    assert entailment_check("Uncited sentence.", items, Fake())["problems"] == []
    assert entailment_check("No warning [L1].", items)["status"] == "unavailable"  # no model in tests


def test_tiny_nli_model_loads(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from pvsys.nlp import checks

    _tiny_bert(tmp_path / "nli", "no boxed warning serious infections",
               num_labels=3, id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
    nli = checks.NLI(tmp_path / "nli")
    assert nli.device == "cpu" and (nli.i_con, nli.i_ent) == (0, 1)
    r = checks.entailment_check("no boxed warning [L1].", [{"id": "L1", "section": "x", "text": "serious infections"}], nli)
    assert r["status"] == "ok" and len(r["sentences"]) == 1


def test_verifier_ablation():
    from pvsys.evidence.compare_generators import ablation

    def row(gold, pred, v1=(), struct=(), ent=(), extr=True):
        return {"gold": gold, "pred": pred, "correct": gold == pred, "extractive_correct": extr, "verified": not v1,
                "problems": list(v1), "seconds": 1.0,
                "checks": {"generated": True, "v1": list(v1), "structure": list(struct), "entailment": list(ent)}}

    rows = [row("yes", "no", struct=["x"]), row("yes", "yes"), row("no", "yes", ent=["y"]), row("no", "no", v1=["z"])]
    a = ablation(rows, task_a=True)
    assert a["v1"]["verified_n"] == 3 and a["v1"]["verified_false_negatives"] == 1
    assert a["v1+structure"]["verified_false_negatives"] == 0 and a["v1+structure"]["verified_n"] == 2
    assert a["v1+structure+entailment"]["verified_n"] == 1 and a["v1+structure+entailment"]["system_accuracy"] == 1.0


def test_generator_quantization_switch(monkeypatch, tmp_path):
    from pvsys.nlp.generator import _wants_4bit

    monkeypatch.setenv("PV_GEN_QUANT", "none")
    assert _wants_4bit(tmp_path, None) is False
