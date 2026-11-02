import { PageHeader } from "@/components/common";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const SECTIONS: { title: string; body: string[] }[] = [
  {
    title: "Data and case handling",
    body: [
      "Source: FDA Adverse Event Reporting System (FAERS, now published as AEMS) quarterly ASCII extract for 2025 Q1, loaded as text and stored as Parquet; DuckDB runs the SQL.",
      "Case versions follow the FDA README: cases on the quarter's delete list are dropped and, for each CASEID, the row with the latest FDA_DT (then highest PRIMARYID) is kept.",
      "Probable duplicates are reports with different CASEIDs but the same sex, age, event date, country, set of suspect drugs and set of reaction terms. They are kept in descriptive counts and excluded from signal detection.",
      "Age is converted to years and weight to kilograms; implausible values are set to missing, never imputed. Drug names use the active ingredient when given, otherwise a regex-cleaned verbatim name (no RxNorm mapping).",
    ],
  },
  {
    title: "Disproportionality analysis",
    body: [
      "For each suspect drug (primary or secondary suspect) and reaction term, a case-level 2×2 table is built over the analysis population.",
      "ROR, PRR (with 95% CIs), Yates chi-square, the BCPNN information component (IC025) and Benjamini–Hochberg q-values are reported. When any cell is zero, 0.5 is added to every cell (Haldane–Anscombe).",
      "Screening rules: ROR lower CI > 1; PRR ≥ 2 with χ² ≥ 4 (Evans); IC025 > 0; each with at least 3 cases. Pairs where the reaction is also the stated indication in ≥ 20% of cases are marked as possible indication bias.",
    ],
  },
  {
    title: "What the statistics cannot tell you",
    body: [
      "Disproportional reporting is not causality: a flagged pair is a hypothesis for clinical review.",
      "FAERS has no denominator of exposed patients, so incidence, prevalence and absolute risk cannot be computed, and ratios cannot be compared as relative risks.",
      "Reporting is voluntary and selective (under-reporting, stimulated reporting after publicity or litigation, notoriety bias, the Weber effect), and many fields are missing.",
      "With hundreds of thousands of reports, very small differences are statistically significant; effect sizes and confidence intervals matter more than p-values.",
    ],
  },
  {
    title: "The language model component",
    body: [
      "Model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext (BERT-base, 110M parameters, Apache-2.0), run locally on CPU. It is PubMedBERT further trained so synonyms of the same UMLS concept have similar embeddings.",
      "Each of 23 project-defined categories is described by 6–14 anchor phrases written by the project before the evaluation reference was inspected. A term is assigned to the category whose averaged anchor embedding is most similar; the model abstains when the similarity or the margin over the runner-up is below thresholds tuned on a development split.",
      "The same embeddings power search by meaning and the concept-to-signals explorer. The statistical pipeline never imports the model; if weights or torch are missing, the dashboard serves precomputed predictions and falls back to a lexical encoder, labelling the mode.",
      "Categories are not MedDRA System Organ Classes. Official MedDRA hierarchies require an MSSO licence and are not used. CTCAE v5.0 (NCI, public) is used as an external evaluation reference through a project-defined crosswalk, and its definitions are shown with attribution as a glossary.",
    ],
  },
  {
    title: "Question answering (Ask the data)",
    body: [
      "Retrieval-augmented: drug names are found in the FAERS vocabulary (plus brand names learnt from FAERS), reaction phrases are linked to MedDRA terms exactly or with SapBERT (accepted only at similarity ≥ 0.70), then the stored statistics and BM25-ranked passages from FDA's FAERS README, FAQ and file descriptions, CTCAE definitions and project method notes are retrieved.",
      "Answers are composed only from that evidence: every sentence cites it, documentation is quoted verbatim, the causality caveat is always attached, and the system abstains or asks for clarification instead of guessing. An optional local generative model's summary is shown only if a verifier finds every number, citation and quotation in the evidence.",
      "Evaluated against FDA's own FAQ answers, NCI definitions and templated questions over real signal pairs (see the page's evaluation tab). It cannot compare drugs, handle doses or time trends, and has not been evaluated on real user questions.",
    ],
  },
];

export default function Methods() {
  return (
    <>
      <PageHeader title="Methods and limitations" purpose="How the numbers on every page were produced, and the limits that apply to all of them." />
      <div className="grid gap-6 lg:grid-cols-2">
        {SECTIONS.map((s) => (
          <Card key={s.title}>
            <CardHeader><CardTitle>{s.title}</CardTitle></CardHeader>
            <CardContent>
              <ul className="list-disc space-y-2 pl-5 text-[15px] leading-relaxed">
                {s.body.map((b) => <li key={b}>{b}</li>)}
              </ul>
            </CardContent>
          </Card>
        ))}
      </div>
    </>
  );
}
