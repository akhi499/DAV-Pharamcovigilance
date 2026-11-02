import { useQuery } from "@tanstack/react-query";
import { Fragment } from "react";
import { CheckCircle2, CircleDashed } from "lucide-react";
import { Disclaimer, PageHeader, ProvisionalBadge, QueryState, SectionTitle } from "@/components/common";
import { Plot, palette } from "@/components/Plot";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TBody, TD, TH, THead, TR, Table } from "@/components/ui/table";
import { apiGet, type Metrics, type NlpStatus } from "@/lib/api";
import { categoryLabel, useCategories } from "@/lib/categories";
import { fmtInt, fmtNum, fmtPct } from "@/lib/utils";

function errorBars(...items: [number, [number, number] | null][]) {
  return {
    array: items.map(([v, c]) => (c ? (c[1] - v) * 100 : 0)),
    arrayminus: items.map(([v, c]) => (c ? (v - c[0]) * 100 : 0)),
  };
}

const ci = (c: [number, number] | null | undefined) => (c ? ` (95% CI ${fmtPct(c[0])}–${fmtPct(c[1])})` : "");

export default function ModelPage() {
  const q = useQuery({ queryKey: ["nlp-status"], queryFn: () => apiGet<NlpStatus>("/nlp/status") });
  const reviews = useQuery({ queryKey: ["nlp-reviews"], queryFn: () => apiGet<{ counts: Record<string, number>; rows: Record<string, string>[]; note: string }>("/nlp/reviews") });
  const cats = useCategories();
  return (
    <>
      <PageHeader title="Model and evaluation" purpose="Which language model produced the categories, exactly which version, and how well it agrees with an independent expert reference compared with simpler baselines." />
      <QueryState q={q} height="h-96">
        {(s) => {
          const ev = s.evaluation_reference;
          const neural = ev?.systems?.neural;
          const lexical = ev?.systems?.lexical;
          const major = ev?.systems?.majority_class;
          const p = s.provenance;
          return (
            <>
              <div className="grid gap-6 lg:grid-cols-2">
                <Card>
                  <CardHeader><CardTitle>Model provenance</CardTitle><CardDescription>Recorded with every prediction run.</CardDescription></CardHeader>
                  <CardContent>
                    <dl className="grid grid-cols-[160px_1fr] gap-x-3 gap-y-1.5 text-sm">
                      <dt className="text-muted-foreground">Status</dt>
                      <dd className="flex items-center gap-1.5">
                        {s.neural_encoder.available ? <CheckCircle2 className="h-4 w-4 text-primary" /> : <CircleDashed className="h-4 w-4 text-signal" />}
                        {s.neural_encoder.available ? "Weights loaded locally (CPU)" : `Unavailable: ${!s.neural_encoder.weights_present ? "weights missing" : "torch/transformers not installed"}. Precomputed predictions still shown; search uses the lexical fallback.`}
                      </dd>
                      {p?.model && Object.entries({
                        Model: p.model.model_id, Revision: p.model.model_revision, Licence: p.model.license, Weights: `${p.model.weights_file} (SHA-256 ${String(p.model.weights_sha256).slice(0, 16)}…)`,
                        Pooling: p.model.pooling, "Max tokens": p.model.max_length, Runtime: `torch ${p.model.torch}, transformers ${p.model.transformers}, ${p.model.device}`,
                      }).map(([k, v]) => (<Fragment key={k}><dt className="text-muted-foreground">{k}</dt><dd className="break-all">{String(v)}</dd></Fragment>))}
                      {p && (<>
                        <dt className="text-muted-foreground">Run</dt><dd>{p.run_id} at {p.created_at}</dd>
                        <dt className="text-muted-foreground">Taxonomy</dt><dd>{p.taxonomy.version} ({p.taxonomy.n_categories} categories, {p.taxonomy.n_anchors} anchor phrases)</dd>
                        <dt className="text-muted-foreground">Abstention rule</dt><dd>similarity &lt; {p.thresholds.min_score} or margin &lt; {p.thresholds.min_margin} ({p.thresholds.source})</dd>
                        <dt className="text-muted-foreground">Data</dt><dd className="break-all">{p.data_manifest?.source_zip} ({p.data_manifest?.source_sha256.slice(0, 16)}…)</dd>
                      </>)}
                    </dl>
                  </CardContent>
                </Card>
                {p && (
                  <div className="provisional-frame p-5">
                    <div className="mb-3 flex items-center justify-between"><h3 className="font-semibold">Run on FAERS 2025 Q1 terms</h3><ProvisionalBadge /></div>
                    <dl className="grid grid-cols-2 gap-4 text-sm">
                      <div><dt className="text-muted-foreground">Terms categorised</dt><dd className="text-2xl font-semibold tabular-nums">{fmtInt(p.n_predicted)}</dd><dd className="text-xs text-muted-foreground">of {fmtInt(p.n_terms)}</dd></div>
                      <div><dt className="text-muted-foreground">Abstained</dt><dd className="text-2xl font-semibold tabular-nums">{fmtInt(p.n_abstained)}</dd><dd className="text-xs text-muted-foreground">{Object.entries(p.abstain_reasons).map(([k, v]) => `${fmtInt(v)} ${k.replace("_", " ")}`).join(", ")}</dd></div>
                      <div><dt className="text-muted-foreground">Case mentions covered</dt><dd className="text-2xl font-semibold tabular-nums">{fmtPct(p.case_weighted_coverage)}</dd><dd className="text-xs text-muted-foreground">frequent terms are rarely ambiguous</dd></div>
                      <div><dt className="text-muted-foreground">Agrees with lexical baseline</dt><dd className="text-2xl font-semibold tabular-nums">{fmtPct(p.agreement_with_lexical_top1)}</dd><dd className="text-xs text-muted-foreground">top-1, all terms</dd></div>
                    </dl>
                    <Disclaimer>No accuracy is claimed for these FAERS predictions until blinded human annotations exist (see below). The reference evaluation measures agreement with NCI's grouping of a different, cleaner term list.</Disclaimer>
                  </div>
                )}
              </div>

              <SectionTitle>Evaluation against NCI CTCAE v5.0</SectionTitle>
              {!ev || ev.status === "unavailable" ? (
                <Alert variant="warning"><AlertTitle>No reference evaluation</AlertTitle><AlertDescription>{ev?.reason ?? "Place CTCAE_v5.0.xlsx in data/reference and run `python -m pvsys.pipeline evaluate`."}</AlertDescription></Alert>
              ) : (
                <>
                  <p className="mb-4 max-w-3xl text-sm leading-relaxed text-muted-foreground">
                    {fmtInt(ev.dataset.evaluation_terms)} CTCAE terms with an unambiguous organ-system heading ({fmtInt(ev.dataset.excluded_equal_to_anchor)} removed because they equal one of our anchor phrases).
                    Thresholds were tuned on a {fmtPct(ev.protocol.dev_fraction, 0)} development split ({fmtInt(ev.dataset.dev_terms)} terms, seed {ev.protocol.seed});
                    all figures below are on the untouched test split of {fmtInt(ev.dataset.test_terms)} terms, {fmtInt(ev.dataset.test_terms_in_faers_2025q1)} of which also appear in FAERS 2025 Q1.
                  </p>
                  <div className="grid gap-6 lg:grid-cols-2">
                    <Card>
                      <CardHeader><CardTitle>Accuracy compared with baselines</CardTitle><CardDescription>Top-1 accuracy if the model always answers, with bootstrap 95% CI.</CardDescription></CardHeader>
                      <CardContent>
                        <Plot
                          ariaLabel="Top-1 accuracy of SapBERT, lexical baseline and majority class"
                          height={260}
                          data={[{
                            type: "bar",
                            x: ["Majority class", "Lexical TF-IDF", "SapBERT"],
                            y: [major?.test_accuracy ?? 0, lexical?.test?.top1_accuracy_no_abstention ?? 0, neural?.test?.top1_accuracy_no_abstention ?? 0].map((v) => v * 100),
                            error_y: { type: "data", symmetric: false, ...errorBars([major?.test_accuracy ?? 0, null], [lexical?.test?.top1_accuracy_no_abstention ?? 0, lexical?.test?.top1_accuracy_95ci ?? null], [neural?.test?.top1_accuracy_no_abstention ?? 0, neural?.test?.top1_accuracy_95ci ?? null]) },
                            marker: { color: [palette.muted, palette.primary, palette.provisional] },
                            hovertemplate: "%{x}: %{y:.1f}%<extra></extra>",
                          }]}
                          layout={{ yaxis: { range: [0, 100], ticksuffix: "%" } }}
                        />
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader><CardTitle>Accuracy when allowed to abstain</CardTitle><CardDescription>Keep only the most confident share of predictions (test split).</CardDescription></CardHeader>
                      <CardContent>
                        <Plot
                          ariaLabel="Accuracy versus coverage"
                          height={260}
                          data={[
                            neural?.risk_coverage_test && { type: "scatter" as const, mode: "lines+markers" as const, name: "SapBERT", x: neural.risk_coverage_test.map((r) => r.coverage * 100), y: neural.risk_coverage_test.map((r) => r.accuracy * 100), line: { color: palette.provisional } },
                            lexical?.risk_coverage_test && { type: "scatter" as const, mode: "lines+markers" as const, name: "Lexical", x: lexical.risk_coverage_test.map((r) => r.coverage * 100), y: lexical.risk_coverage_test.map((r) => r.accuracy * 100), line: { color: palette.primary } },
                          ].filter(Boolean) as Plotly.Data[]}
                          layout={{ xaxis: { title: { text: "share of terms answered" }, ticksuffix: "%" }, yaxis: { title: { text: "accuracy" }, ticksuffix: "%", range: [0, 100] }, legend: { orientation: "h", x: 0, y: 1.12 }, margin: { l: 56, r: 16, t: 28, b: 48 } }}
                        />
                      </CardContent>
                    </Card>
                  </div>
                  <Card className="mt-6">
                    <Table>
                      <THead><TR><TH>System</TH><TH className="text-right">Top-1 accuracy</TH><TH className="text-right">Coverage with abstention</TH><TH className="text-right">Accuracy when answering</TH><TH className="text-right">Macro-F1 (answered)</TH><TH className="text-right">FAERS-overlap subset</TH></TR></THead>
                      <TBody>
                        {[["SapBERT (zero-shot prototypes)", neural?.test, neural?.test_in_faers], ["Lexical TF-IDF baseline", lexical?.test, lexical?.test_in_faers]].map(([name, m, mf]) => {
                          const mm = m as Metrics | undefined;
                          const mmf = mf as Metrics | undefined;
                          return mm ? (
                            <TR key={name as string}>
                              <TD className="font-medium">{name as string}</TD>
                              <TD className="text-right tabular-nums">{fmtPct(mm.top1_accuracy_no_abstention)}<span className="block text-xs text-muted-foreground">{ci(mm.top1_accuracy_95ci)}</span></TD>
                              <TD className="text-right tabular-nums">{fmtPct(mm.coverage)}</TD>
                              <TD className="text-right tabular-nums">{fmtPct(mm.selective_accuracy)}<span className="block text-xs text-muted-foreground">{ci(mm.selective_accuracy_95ci)}</span></TD>
                              <TD className="text-right tabular-nums">{fmtNum(mm.macro_f1_answered, 3)}</TD>
                              <TD className="text-right tabular-nums">{fmtPct(mmf?.top1_accuracy_no_abstention)} <span className="text-xs text-muted-foreground">n={mmf?.n}</span></TD>
                            </TR>
                          ) : null;
                        })}
                        {major && <TR><TD className="font-medium">Majority class ({categoryLabel(major.predicts, cats.data)})</TD><TD className="text-right tabular-nums">{fmtPct(major.test_accuracy)}</TD><TD colSpan={4} /></TR>}
                        {neural?.status === "unavailable" && <TR><TD colSpan={6} className="text-signal">SapBERT not evaluated: {neural.reason}</TD></TR>}
                      </TBody>
                    </Table>
                  </Card>
                  {neural?.test && <PerCategory m={neural.test} />}
                  {neural?.confusion_test && (
                    <Card className="mt-6">
                      <CardHeader><CardTitle>Where SapBERT goes wrong</CardTitle><CardDescription>Test-split confusion matrix: rows are the reference category, columns the model output (including abstentions).</CardDescription></CardHeader>
                      <CardContent>
                        <Plot
                          ariaLabel="Confusion matrix"
                          height={560}
                          data={[{
                            type: "heatmap", z: neural.confusion_test.matrix,
                            x: neural.confusion_test.labels_pred.map((l) => categoryLabel(l, cats.data)),
                            y: neural.confusion_test.labels_gold.map((l) => categoryLabel(l, cats.data)),
                            colorscale: [[0, "rgba(109,79,176,0.03)"], [1, palette.provisional]], hovertemplate: "reference %{y}<br>model %{x}<br>%{z} terms<extra></extra>",
                          }]}
                          layout={{ margin: { l: 190, r: 16, t: 8, b: 160 }, xaxis: { tickangle: -45 } }}
                        />
                      </CardContent>
                    </Card>
                  )}
                  <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                    {ev.caveats.map((c) => <li key={c}>{c}</li>)}
                  </ul>
                </>
              )}

              <SectionTitle>Human review</SectionTitle>
              <div className="grid gap-6 lg:grid-cols-2">
                <Card>
                  <CardHeader><CardTitle>Blinded annotation (ground truth)</CardTitle><CardDescription>The only labels used to score the model on FAERS terms.</CardDescription></CardHeader>
                  <CardContent className="text-sm leading-relaxed">
                    {s.evaluation_annotations?.status === "ok" ? (
                      <p>
                        {fmtInt(s.evaluation_annotations.n_gold_terms)} annotated terms. SapBERT top-1 accuracy {fmtPct(s.evaluation_annotations.neural?.top1_accuracy_no_abstention)};
                        inter-annotator κ = {fmtNum(s.evaluation_annotations.cohens_kappa_double_annotated ?? null, 2)}.
                      </p>
                    ) : (
                      <p className="mb-3"><Badge variant="muted">No annotations yet</Badge> No accuracy is claimed on FAERS terms.</p>
                    )}
                    <ol className="list-decimal space-y-1 pl-5">
                      <li>Export a stratified batch without model output: <code className="rounded bg-muted px-1">python -m pvsys.pipeline annotate-export --n 200</code></li>
                      <li>Two people label it independently using <code className="rounded bg-muted px-1">LABEL_GUIDE.md</code>.</li>
                      <li>Import each sheet: <code className="rounded bg-muted px-1">python -m pvsys.pipeline annotate-import data/reviews/annotation_batches/&lt;file&gt;.csv</code></li>
                      <li>Re-score: <code className="rounded bg-muted px-1">python -m pvsys.pipeline evaluate</code></li>
                    </ol>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>Dashboard reviews</CardTitle><CardDescription>Accept, correct or reject from the Term predictions table. Logged, not scored.</CardDescription></CardHeader>
                  <CardContent>
                    <QueryState q={reviews}>
                      {(r) => (
                        <>
                          <div className="mb-3 flex flex-wrap gap-2 text-sm">
                            {Object.keys(r.counts).length === 0 ? <span className="text-muted-foreground">No reviews recorded yet.</span> :
                              Object.entries(r.counts).map(([k, v]) => <Badge key={k} variant="outline">{k}: {v}</Badge>)}
                          </div>
                          <ul className="max-h-48 divide-y overflow-y-auto text-sm">
                            {r.rows.slice(0, 20).map((x, i) => (
                              <li key={i} className="py-1.5"><span className="font-medium">{x.pt}</span>: {x.decision}{x.corrected_category ? ` → ${categoryLabel(x.corrected_category, cats.data)}` : ""} <span className="text-xs text-muted-foreground">by {x.reviewer}, {x.reviewed_at}</span></li>
                            ))}
                          </ul>
                          <Disclaimer>{r.note}</Disclaimer>
                        </>
                      )}
                    </QueryState>
                  </CardContent>
                </Card>
              </div>
            </>
          );
        }}
      </QueryState>
    </>
  );
}

function PerCategory({ m }: { m: Metrics }) {
  const cats = useCategories();
  const rows = Object.entries(m.per_category).sort((a, b) => (b[1].recall ?? -1) - (a[1].recall ?? -1));
  return (
    <Card className="mt-6">
      <CardHeader><CardTitle>SapBERT by category (test split)</CardTitle><CardDescription>Small supports make individual figures unstable.</CardDescription></CardHeader>
      <Table>
        <THead><TR><TH>Reference category</TH><TH className="text-right">Terms</TH><TH className="text-right">Precision</TH><TH className="text-right">Recall</TH><TH className="text-right">Abstained</TH></TR></THead>
        <TBody>
          {rows.map(([k, v]) => (
            <TR key={k}>
              <TD>{categoryLabel(k, cats.data)}</TD><TD className="text-right tabular-nums">{v.support}</TD>
              <TD className="text-right tabular-nums">{fmtPct(v.precision, 0)}</TD><TD className="text-right tabular-nums">{fmtPct(v.recall, 0)}</TD>
              <TD className="text-right tabular-nums">{v.abstained}</TD>
            </TR>
          ))}
        </TBody>
      </Table>
    </Card>
  );
}
