"""External clinical evidence for the question-answering module.

Sources (all public, no API key needed):

* FDA drug labels (Structured Product Labeling) - section text from openFDA ``/drug/label``,
  label set chosen and linked through NLM DailyMed;
* FDA Drug Safety Communications - www.fda.gov;
* scientific literature - PubMed abstracts of reviews / meta-analyses via NCBI E-utilities.

Modules: ``sources`` (download + cache), ``chunking``, ``embed`` (BGE sentence embeddings),
``index`` (vector + BM25 hybrid retrieval), ``evaluate``. The statistical pipeline never imports
this package.
"""
