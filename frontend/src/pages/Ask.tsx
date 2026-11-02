import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ArrowRight, BookOpen, CheckCircle2, CircleDashed, ExternalLink, FileText, HelpCircle, Landmark, MessageSquareText, Microscope, Sparkles } from "lucide-react";
import { Fragment, useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Disclaimer, PageHeader, ProvisionalBadge, QueryState, SectionTitle } from "@/components/common";
import { Plot, palette } from "@/components/Plot";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { TBody, TD, TH, THead, TR, Table } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  apiGet, apiPost, type AskResponse, type DocEvidence, type Established, type EvidenceDocs, type EvidenceEvaluation, type EvidenceStatus,
  type ExternalEvidence, type IrMetrics, type RagEvaluation, type RagStatus, type RankMetrics, type SourceType, type StatEvidence,
} from "@/lib/api";
import { cn, fmtInt, fmtNum, fmtPct } from "@/lib/utils";

const SOURCE_LABEL: Record<string, string> = {
  "FDA-README": "FDA FAERS README",
  "FDA-FAQ": "FDA FAERS FAQ",
  "FDA-ASC_NTS": "FDA file descriptions",
  PROJECT: "Project methods note",
  CTCAE: "NCI CTCAE v5.0",
};
const MATCH_LABEL: Record<string, string> = {
  exact: "exact name", alias: "brand/verbatim name in FAERS", prefix: "only product starting with this word", fuzzy: "spelling-tolerant match",
  semantic: "language-model link",
};

export default function Ask() {
  const [tab, setTab] = useState("ask");
  return (
    <>
      <PageHeader
        title="Ask the data"
        purpose="Ask a question in plain language. The system finds the drug and reaction you mean, retrieves what FAERS reports show and what FDA labels, FDA safety communications and published reviews state, and answers only from that evidence, with citations."
      />
      <Alert variant="provisional" className="mb-6">
        <Sparkles className="h-4 w-4 text-provisional" />
        <AlertTitle>Retrieval-augmented answers are provisional</AlertTitle>
        <AlertDescription>
          SapBERT links your words to MedDRA reaction terms; FAERS statistics come straight from DuckDB; clinical sources (FDA labels, FDA safety
          communications, PubMed reviews) are found by vector search (BGE embeddings) and quoted with their source details; an experimental local model
          (Qwen2.5) can be switched on to write that part, shown only if every sentence is supported by the passage it cites. Check the evidence: a reporting association in FAERS is not proof that a drug causes a reaction.
        </AlertDescription>
      </Alert>
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="ask">Ask</TabsTrigger>
          <TabsTrigger value="sources">Sources</TabsTrigger>
          <TabsTrigger value="evaluation">How well it works</TabsTrigger>
        </TabsList>
        <TabsContent value="ask"><AskPanel /></TabsContent>
        <TabsContent value="sources"><SourcesPanel /></TabsContent>
        <TabsContent value="evaluation"><EvaluationPanel /></TabsContent>
      </Tabs>
    </>
  );
}

function AskPanel() {
  const [params, setParams] = useSearchParams();
  const asked = params.get("q") ?? "";
  const generate = params.get("gen") === "1"; // opt-in: the evaluation found the small local model unreliable
  const [draft, setDraft] = useState(asked);
  const [wantGen, setWantGen] = useState(generate);
  useEffect(() => setDraft(asked), [asked]);
  const status = useQuery({ queryKey: ["rag-status"], queryFn: () => apiGet<RagStatus>("/rag/status"), staleTime: 60_000 });
  const answer = useQuery({
    queryKey: ["rag-ask", asked, generate],
    queryFn: () => apiPost<AskResponse>("/rag/ask", generate ? { question: asked, generate: true } : { question: asked }),
    enabled: asked.trim().length >= 3,
    staleTime: Infinity,
    retry: false,
  });
  const submit = (q: string, gen = wantGen) => {
    const v = q.trim();
    if (v.length < 3) return;
    const next = new URLSearchParams();
    next.set("q", v);
    if (gen) next.set("gen", "1");
    setParams(next);
  };
  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    submit(draft);
  };
  const genAvailable = status.data?.generator.available ?? false;
  return (
    <div className="mt-4">
      <form onSubmit={onSubmit} className="flex flex-col gap-3 rounded-lg border bg-card p-4">
        <Label htmlFor="rag-q">Your question</Label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input id="rag-q" value={draft} maxLength={300} onChange={(e) => setDraft(e.target.value)}
            placeholder="e.g. Is semaglutide associated with pancreatitis?" className="h-10 text-[15px]" />
          <Button type="submit" disabled={draft.trim().length < 3 || answer.isFetching} className="h-10">
            <MessageSquareText className="h-4 w-4" /> Ask
          </Button>
        </div>
        <label className={cn("flex items-center gap-2 text-sm", !genAvailable && "text-muted-foreground")}>
          <input type="checkbox" checked={wantGen && genAvailable} disabled={!genAvailable} onChange={(e) => setWantGen(e.target.checked)} />
          Experimental: let the local generative model (Qwen2.5-0.5B) write the clinical-evidence answer – shown only if every sentence is supported by its cited source; it passed these checks for 43% of evaluation questions
          {!genAvailable && <span className="text-xs">(not installed: see README, “Local models”)</span>}
        </label>
        {status.data && (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs text-muted-foreground">Try:</span>
            {status.data.examples.map((ex) => (
              <button key={ex} type="button" onClick={() => { setDraft(ex); submit(ex); }}
                className="rounded-full border bg-background px-3 py-1 text-xs hover:bg-accent">
                {ex}
              </button>
            ))}
          </div>
        )}
      </form>
      {status.data && !status.data.available && (
        <Alert variant="warning" className="mt-4">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Knowledge base not built</AlertTitle>
          <AlertDescription>Run <code>python -m pvsys.pipeline rag-index</code> in the backend folder, then reload.</AlertDescription>
        </Alert>
      )}
      {asked && (answer.isPending && answer.fetchStatus !== "idle" ? <Skeleton className="mt-6 h-64 w-full" /> : (
        <QueryState q={answer} height="h-64">{(r) => <AnswerView r={r} />}</QueryState>
      ))}
    </div>
  );
}

function CiteChip({ id }: { id: string }) {
  return (
    <a href={`#ev-${id}`} className="ml-1 inline-flex rounded border border-provisional/40 px-1 text-[11px] font-medium text-provisional no-underline hover:bg-provisional-soft"
      onClick={(e) => {
        e.preventDefault();
        const el = document.getElementById(`ev-${id}`);
        el?.scrollIntoView?.({ behavior: "smooth", block: "center" });
        el?.classList.add("ring-2", "ring-provisional");
        window.setTimeout(() => el?.classList.remove("ring-2", "ring-provisional"), 1600);
      }}>
      {id}
    </a>
  );
}

function AnswerView({ r }: { r: AskResponse }) {
  const stats = r.evidence.filter((e): e is StatEvidence => e.kind !== "document");
  const docs = r.evidence.filter((e): e is DocEvidence => e.kind === "document");
  const u = r.understanding;
  return (
    <div className="mt-6 grid gap-6 xl:grid-cols-5">
      <div className="flex flex-col gap-6 xl:col-span-3">
        <section className="provisional-frame p-5" aria-label="Answer">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-semibold">Observed in FAERS reports</h2>
            <div className="flex items-center gap-2">
              {r.status === "answered" ? <Badge variant="default"><CheckCircle2 className="h-3 w-3" /> Grounded in {r.evidence.length} evidence items</Badge>
                : r.status === "abstained" ? <Badge variant="muted"><CircleDashed className="h-3 w-3" /> Not answered</Badge>
                  : <Badge variant="signal"><HelpCircle className="h-3 w-3" /> Needs clarification</Badge>}
              <ProvisionalBadge />
            </div>
          </div>
          <p className="mb-3 text-sm text-muted-foreground">“{r.question}”</p>
          <ul className="flex flex-col gap-2.5 text-[15px] leading-relaxed">
            {r.answer.map((s, i) => (
              <li key={i} className={cn(s.text.startsWith("This is a reporting association") && "border-t pt-2.5 text-sm text-muted-foreground")}>
                {s.text}{s.cites.map((c) => <CiteChip key={c} id={c} />)}
              </li>
            ))}
          </ul>
          <GeneratedPanel r={r} />
          <p className="mt-3 text-xs text-muted-foreground">Counts of spontaneous reports and disproportionality statistics: what was reported, not what is proven.</p>
        </section>
        {r.established && <EstablishedPanel e={r.established} />}
        {(r.established?.evidence?.length ?? 0) > 0 && <ExternalSources items={r.established!.evidence!} />}
        {stats.length > 0 && <StatTable rows={stats} />}
        {docs.length > 0 && (
          <section aria-label="Documents">
            <SectionTitle>Documents retrieved</SectionTitle>
            <div className="flex flex-col gap-3">
              {docs.map((d) => (
                <div key={d.id} id={`ev-${d.id}`} className="rounded-lg border bg-card p-4 transition-shadow">
                  <div className="mb-1.5 flex flex-wrap items-center gap-2 text-sm">
                    <Badge variant="outline">{d.id}</Badge>
                    <Badge variant="muted"><BookOpen className="h-3 w-3" /> {SOURCE_LABEL[d.source] ?? d.source}</Badge>
                    <span className="font-medium">{d.title}</span>
                    {d.score != null && <span className="text-xs text-muted-foreground">BM25 {fmtNum(d.score, 1)}</span>}
                  </div>
                  <p className="text-sm leading-relaxed text-muted-foreground">{d.text.length > 700 ? `${d.text.slice(0, 700)}…` : d.text}</p>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
      <aside className="flex flex-col gap-6 xl:col-span-2">
        <Card>
          <CardHeader><CardTitle>How the question was read</CardTitle><CardDescription>Intent: {r.intent} · reaction linking: {u.link_mode === "neural" ? "SapBERT" : "spelling fallback"}</CardDescription></CardHeader>
          <CardContent className="flex flex-col gap-3 text-sm">
            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">Drugs</div>
              {u.drugs.length ? u.drugs.map((d) => (
                <div key={d.drug} className="flex flex-wrap items-baseline gap-1.5"><span className="font-medium">{d.drug}</span>
                  <span className="text-xs text-muted-foreground">from “{d.text}”, {MATCH_LABEL[d.match]}, {fmtInt(d.cases)} suspect cases</span></div>
              )) : <span className="text-muted-foreground">none recognised</span>}
              {u.ambiguous_drugs.map((a) => (
                <div key={a.text} className="mt-1 text-xs text-signal">“{a.text}” is ambiguous: {a.candidates.map((c) => c.drug).join(", ")}</div>
              ))}
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">Reaction terms (MedDRA PT)</div>
              {u.reactions.length ? u.reactions.map((x) => (
                <div key={x.pt_key} className="mb-1.5">
                  <div className="flex flex-wrap items-baseline gap-1.5"><span className="font-medium">{x.pt}</span>
                    <span className="text-xs text-muted-foreground">from “{x.phrase}”, {MATCH_LABEL[x.match]}{x.match === "semantic" && `, similarity ${x.similarity.toFixed(2)}`}</span></div>
                  {x.related.length > 0 && <div className="text-xs text-muted-foreground">related: {x.related.map((t) => t.pt).join(", ")}</div>}
                </div>
              )) : <span className="text-muted-foreground">none recognised</span>}
              {u.unlinked_phrases.filter((p) => p.best_candidate).map((p) => (
                <div key={p.phrase} className="text-xs text-muted-foreground">“{p.phrase}” not linked (best guess {p.best_candidate}, {fmtNum(p.best_similarity ?? 0)} &lt; {p.threshold})</div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Provenance</CardTitle><CardDescription>Answered in {fmtInt(r.elapsed_ms)} ms</CardDescription></CardHeader>
          <CardContent>
            <dl className="grid grid-cols-[130px_1fr] gap-x-3 gap-y-1 text-xs">
              {Object.entries(r.provenance).filter(([, v]) => v != null).map(([k, v]) => (
                <Fragment key={k}><dt className="text-muted-foreground">{k.replace(/_/g, " ")}</dt><dd className="break-all">{String(v)}</dd></Fragment>
              ))}
            </dl>
          </CardContent>
        </Card>
        <Disclaimer>{r.disclaimer}</Disclaimer>
      </aside>
    </div>
  );
}

function GeneratedPanel({ r }: { r: AskResponse }) {
  const g = r.generated_summary;
  if (g.status === "not_requested" || g.status === "unavailable" || g.status === "skipped") return null;
  return (
    <div className="mt-4 rounded-md border border-dashed border-provisional/50 bg-card p-3 text-sm">
      <div className="mb-1 flex items-center gap-2 font-medium">
        <Sparkles className="h-4 w-4 text-provisional" /> Generated summary ({g.model_id})
        {g.status === "verified" ? <Badge variant="default">Passed evidence check</Badge> : <Badge variant="danger">Discarded</Badge>}
      </div>
      {g.status === "verified" && <p>{g.text}</p>}
      {g.status !== "verified" && <p className="text-muted-foreground">The draft was discarded: {(g.problems ?? [g.reason]).join("; ")}.</p>}
    </div>
  );
}

function StatTable({ rows }: { rows: StatEvidence[] }) {
  const tested = rows.filter((r) => r.kind === "statistic");
  const aggregates = rows.filter((r) => r.kind === "aggregate");
  return (
    <section aria-label="Statistical evidence">
      <SectionTitle aside={<span className="text-xs text-muted-foreground">precomputed; the language model does not change them</span>}>Statistical evidence</SectionTitle>
      {aggregates.map((a) => (
        <div key={a.id} id={`ev-${a.id}`} className="mb-2 rounded-md border bg-card px-3 py-2 text-sm transition-shadow">
          <Badge variant="outline" className="mr-2">{a.id}</Badge>
          {a.drug ? `${a.drug}: ${fmtInt(a.suspect_cases)} suspect cases, ${fmtInt(a.pairs_tested)} tested reaction terms, ${fmtInt(a.pairs_flagged_all_three)} flagged by all three methods`
            : `${a.pt}: ${fmtInt(a.reaction_cases)} cases (${a.link === "semantic" ? `language-model link, similarity ${a.similarity}` : "exact term"})`}
        </div>
      ))}
      {tested.length > 0 && (
        <div className="overflow-x-auto rounded-lg border bg-card">
          <Table>
            <THead><TR><TH>Id</TH><TH>Drug</TH><TH>Reaction</TH><TH className="text-right">Cases (a)</TH><TH className="text-right">Expected</TH><TH className="text-right">ROR (95% CI)</TH><TH className="text-right">PRR</TH><TH className="text-right">IC025</TH><TH>Flags</TH><TH /></TR></THead>
            <TBody>
              {tested.map((s) => (
                <TR key={s.id} id={`ev-${s.id}`} className="transition-shadow">
                  <TD><Badge variant="outline">{s.id}</Badge></TD>
                  <TD className="font-medium">{s.drug}</TD>
                  <TD>{s.pt}</TD>
                  <TD className="text-right tabular-nums">{fmtInt(s.a)}</TD>
                  {s.tested === false ? (
                    <TD colSpan={4} className="text-muted-foreground">not tested: fewer than {s.min_cases} co-reports</TD>
                  ) : (<>
                    <TD className="text-right tabular-nums">{fmtNum(s.expected, 1)}</TD>
                    <TD className="whitespace-nowrap text-right tabular-nums">{fmtNum(s.ror)} ({fmtNum(s.ror_lower)}–{fmtNum(s.ror_upper)})</TD>
                    <TD className="text-right tabular-nums">{fmtNum(s.prr)}</TD>
                    <TD className="text-right tabular-nums">{fmtNum(s.ic025)}</TD>
                  </>)}
                  <TD className="whitespace-nowrap">
                    {s.methods_flagging != null && <Badge variant={s.methods_flagging === 3 ? "signal" : "muted"}>{s.methods_flagging}/3 methods</Badge>}
                    {s.possible_indication_bias && <Badge variant="outline" className="ml-1">Indication? {s.indication_overlap_pct}%</Badge>}
                    {s.haldane_applied && <Badge variant="outline" className="ml-1">Haldane</Badge>}
                  </TD>
                  <TD>
                    {s.drug && s.pt_key && (
                      <Link to={`/signals?drug=${encodeURIComponent(s.drug)}&pt=${encodeURIComponent(s.pt_key)}&min_cases=3&xib=0&method=any`}
                        className="inline-flex items-center gap-1 text-xs text-primary hover:underline" aria-label={`Open ${s.drug} and ${s.pt} in signal detection`}>
                        details <ArrowRight className="h-3 w-3" />
                      </Link>
                    )}
                  </TD>
                </TR>
              ))}
            </TBody>
          </Table>
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ evaluation tab

const pct = (x: number | null | undefined) => fmtPct(x ?? null);
const ci = (c: [number, number] | null | undefined) => (c ? `${fmtPct(c[0])}–${fmtPct(c[1])}` : "–");

function RankRow({ name, m, note }: { name: string; m: Partial<RankMetrics> | undefined; note?: string }) {
  if (!m) return null;
  return (
    <TR>
      <TD className="font-medium">{name}{note && <div className="text-xs font-normal text-muted-foreground">{note}</div>}</TD>
      <TD className="text-right tabular-nums">{pct(m.recall_at_1)}{m.recall_at_1_95ci && <div className="text-xs text-muted-foreground">{ci(m.recall_at_1_95ci)}</div>}</TD>
      <TD className="text-right tabular-nums">{pct(m.recall_at_3)}</TD>
      <TD className="text-right tabular-nums">{pct(m.recall_at_5)}</TD>
      <TD className="text-right tabular-nums">{pct(m.recall_at_10)}</TD>
      <TD className="text-right tabular-nums">{fmtNum(m.mrr, 3)}{m.mrr_95ci && <div className="text-xs text-muted-foreground">{fmtNum(m.mrr_95ci[0], 3)}–{fmtNum(m.mrr_95ci[1], 3)}</div>}</TD>
    </TR>
  );
}

function RankHead() {
  return <THead><TR><TH>System</TH><TH className="text-right">Top-1 (95% CI)</TH><TH className="text-right">Top-3</TH><TH className="text-right">Top-5</TH><TH className="text-right">Top-10</TH><TH className="text-right">MRR</TH></TR></THead>;
}

function EvaluationPanel() {
  const q = useQuery({ queryKey: ["rag-eval"], queryFn: () => apiGet<RagEvaluation>("/rag/evaluation") });
  return (
    <div className="mt-4">
      <QueryState q={q} height="h-96">
        {(ev) => {
          const faq = ev.faq_retrieval;
          const link = ev.reaction_linking;
          const e2e = ev.end_to_end;
          const sap = link.systems?.sapbert;
          const sys = link.systems ?? {};
          return (
            <div className="flex flex-col gap-6">
              <p className="max-w-3xl text-sm text-muted-foreground">
                Each part of the pipeline is scored against references written by someone other than this project: FDA's own FAQ answers, and NCI's
                CTCAE definitions. Templated questions built from real signal pairs are also checked. Run on {ev.created_at}.
              </p>
              <Card>
                <CardHeader>
                  <CardTitle>1 · Finding the right documentation passage</CardTitle>
                  <CardDescription>
                    {faq.status === "ok" ? `The ${faq.queries} questions in FDA's FAERS FAQ are used as queries over ${faq.corpus_size} passages (${faq.corpus}). The gold answer is the one FDA wrote.` : faq.reason}
                  </CardDescription>
                </CardHeader>
                {faq.systems && (
                  <CardContent>
                    <div className="overflow-x-auto"><Table><RankHead /><TBody>
                      <RankRow name="BM25 (used by the app)" m={faq.systems.bm25} note="textbook parameters, not tuned" />
                      <RankRow name="Word TF-IDF baseline" m={faq.systems.tfidf} />
                      <RankRow name="Random ranking" m={faq.systems.random} note="expected value" />
                    </TBody></Table></div>
                    {faq.paired_difference_tfidf_minus_bm25 && (
                      <Disclaimer>
                        TF-IDF minus BM25 (paired bootstrap): top-1 {fmtPct(faq.paired_difference_tfidf_minus_bm25.recall_at_1.mean)} (95% CI {ci(faq.paired_difference_tfidf_minus_bm25.recall_at_1["95ci"])}).
                        The retriever was fixed before this test, so the result is reported, not used to re-select on the same 52 questions.
                      </Disclaimer>
                    )}
                  </CardContent>
                )}
              </Card>
              <Card className="provisional-frame">
                <CardHeader>
                  <div className="flex items-center justify-between"><CardTitle>2 · Linking words to MedDRA reaction terms</CardTitle><ProvisionalBadge /></div>
                  <CardDescription>
                    {link.status === "ok" ? `For ${link.queries} CTCAE v5.0 terms that also occur in FAERS, NCI's written definition is the query and the gold answer is that term, among all ${fmtInt(link.candidates)} FAERS terms. The input variant was chosen on a ${link.dev}-term dev split; results are for the ${link.test}-term test split.` : link.reason}
                  </CardDescription>
                </CardHeader>
                {link.status === "ok" && (
                  <CardContent className="flex flex-col gap-4">
                    <div className="overflow-x-auto rounded-md bg-card"><Table><RankHead /><TBody>
                      <RankRow name="SapBERT (used by the app)" m={sap?.test} note={sap?.chosen_variant} />
                      <RankRow name="Character TF-IDF baseline" m={sys.tfidf_char?.test} note={sys.tfidf_char?.chosen_variant} />
                      <RankRow name="Word BM25 baseline" m={sys.bm25_word?.test} note={sys.bm25_word?.chosen_variant} />
                    </TBody></Table></div>
                    <Plot
                      ariaLabel="Top-k recall of reaction linking by system"
                      height={260}
                      data={[["SapBERT", sap?.test, palette.provisional], ["Char TF-IDF", sys.tfidf_char?.test, palette.muted], ["Word BM25", sys.bm25_word?.test, palette.signal]]
                        .filter(([, m]) => m).map(([name, m, color]) => ({
                          type: "bar", name: name as string, marker: { color: color as string },
                          x: ["Top-1", "Top-3", "Top-5", "Top-10"],
                          y: ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10"].map((k) => 100 * ((m as RankMetrics)[k as keyof RankMetrics] as number)),
                          hovertemplate: "%{x}: %{y:.1f}%<extra>" + name + "</extra>",
                        }))}
                      layout={{ barmode: "group", yaxis: { title: { text: "% of test definitions" }, range: [0, 100] } }}
                    />
                    {sap?.at_production_threshold && (
                      <p className="text-sm text-muted-foreground">
                        At the app's acceptance threshold (similarity ≥ {sap.at_production_threshold.threshold}), {pct(sap.at_production_threshold.test_coverage)} of test definitions
                        get a link, and {pct(sap.at_production_threshold.test_precision_at_1_when_accepted)} of those links are the gold term. {sap.at_production_threshold.note}
                      </p>
                    )}
                    {link.examples_test && (
                      <details className="text-sm">
                        <summary className="cursor-pointer text-muted-foreground">Examples from the test split</summary>
                        <div className="mt-2 overflow-x-auto"><Table><THead><TR><TH>NCI definition</TH><TH>Gold term</TH><TH>SapBERT top match</TH><TH className="text-right">Rank of gold</TH></TR></THead><TBody>
                          {link.examples_test.map((x) => (
                            <TR key={x.gold}><TD className="max-w-md text-xs">{x.definition}</TD><TD>{x.gold}</TD>
                              <TD className={x.rank === 1 ? "text-primary" : ""}>{x.sapbert_top1}</TD><TD className="text-right tabular-nums">{x.rank}</TD></TR>
                          ))}
                        </TBody></Table></div>
                      </details>
                    )}
                  </CardContent>
                )}
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>3 · End to end: does a question retrieve the right statistics?</CardTitle>
                  <CardDescription>{e2e.note}</CardDescription>
                </CardHeader>
                {e2e.status === "ok" && (
                  <CardContent className="grid gap-4 md:grid-cols-2">
                    <div>
                      <div className="text-3xl font-semibold tabular-nums">{pct(e2e.exact_retrieval_generic_templates)}</div>
                      <div className="text-sm text-muted-foreground">
                        of {fmtInt((e2e.pairs_sampled ?? 0) * 5)} templated questions ({e2e.pairs_sampled} random pairs × 5 phrasings) returned exactly the stored statistics
                        (95% CI {ci(e2e.exact_retrieval_95ci)}). Mean latency {fmtInt(e2e.mean_latency_ms)} ms on CPU.
                      </div>
                      <div className="mt-3 text-sm">
                        Grounding self-check: {pct(e2e.deterministic_answers_passing_grounding_check)} of answers have every number, citation and quotation traceable to their evidence.
                      </div>
                    </div>
                    <div className="text-sm">
                      <Table><THead><TR><TH>Phrasing</TH><TH className="text-right">Exact</TH></TR></THead><TBody>
                        {Object.entries(e2e.per_template ?? {}).map(([k, v]) => (
                          <TR key={k}><TD>{k.replace(/_/g, " ")}</TD><TD className="text-right tabular-nums">{pct(v.exact_retrieval)} <span className="text-xs text-muted-foreground">n={v.n}</span></TD></TR>
                        ))}
                      </TBody></Table>
                      {e2e.failure_reasons && Object.keys(e2e.failure_reasons).length > 0 && (
                        <p className="mt-2 text-xs text-muted-foreground">Failures: {Object.entries(e2e.failure_reasons).map(([k, v]) => `${k} (${v})`).join(", ")}</p>
                      )}
                    </div>
                  </CardContent>
                )}
              </Card>
              <SectionTitle>Clinical sources (labels, FDA communications, literature)</SectionTitle>
              <EvidenceEvaluationPanel />
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>What these numbers do not show</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc pl-4">
                    {ev.caveats.map((c) => <li key={c}>{c}</li>)}
                    <li>{ev.generator.reason}</li>
                  </ul>
                </AlertDescription>
              </Alert>
            </div>
          );
        }}
      </QueryState>
    </div>
  );
}

// ------------------------------------------------------------------ established clinical evidence

const TYPE_META: Record<SourceType, { label: string; icon: typeof FileText }> = {
  label: { label: "FDA label", icon: FileText },
  regulatory: { label: "FDA safety communication", icon: Landmark },
  literature: { label: "Published review", icon: Microscope },
};

function EstablishedPanel({ e }: { e: Established }) {
  if (e.status === "not_applicable" || e.status === "not_requested") return null;
  const g = e.generated;
  return (
    <section className="rounded-lg border-2 border-primary/30 bg-card p-5" aria-label="Established clinical evidence">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Established clinical evidence</h2>
        <div className="flex flex-wrap items-center gap-2">
          {e.status === "answered" && <Badge variant="default"><CheckCircle2 className="h-3 w-3" /> From {e.evidence?.length ?? 0} source passages</Badge>}
          {e.status === "partial" && <Badge variant="signal">Label check only</Badge>}
          {e.status === "insufficient" && <Badge variant="muted"><CircleDashed className="h-3 w-3" /> Insufficient evidence</Badge>}
          {e.status === "unavailable" && <Badge variant="muted">Sources not indexed</Badge>}
          {g?.status === "verified" && <Badge variant="provisional"><Sparkles className="h-3 w-3" /> AI-written, passed source checks</Badge>}
          {g && g.status === "rejected" && <Badge variant="outline">AI draft discarded; sources quoted</Badge>}
        </div>
      </div>
      {e.status === "unavailable" ? (
        <p className="text-sm text-muted-foreground">{e.reason}</p>
      ) : (
        <>
          {e.label_mentions && e.label_mentions.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {e.label_mentions.map((m) => (
                <span key={m.drug + m.pt} className={cn("rounded-md border px-2 py-1 text-xs", m.found ? "border-primary/40 bg-accent" : "bg-muted")}>
                  {m.pt} in {m.drug} label: <strong>{m.found ? `yes (${m.sections.join(", ")})` : m.label_indexed ? "not found" : "label not indexed"}</strong>
                </span>
              ))}
            </div>
          )}
          <ul className="flex flex-col gap-2.5 text-[15px] leading-relaxed">
            {(e.sentences ?? []).map((s, i) => (
              <li key={i} className={cn(s.generated && "rounded-md bg-provisional-soft p-2")}>
                {s.generated ? <CitedText text={s.text} /> : <>{s.text}{s.cites.map((c) => <CiteChip key={c} id={c} />)}</>}
              </li>
            ))}
          </ul>
          {g?.status === "rejected" && (
            <details className="mt-3 text-xs text-muted-foreground">
              <summary className="cursor-pointer">Why the AI-written answer was not shown</summary>
              <p className="mt-1">Draft: “{g.rejected_text}”</p>
              <p>Checks failed: {(g.problems ?? []).join("; ")}</p>
            </details>
          )}
          <p className="mt-3 text-xs text-muted-foreground">
            What FDA labeling, FDA safety communications and published reviews state. Retrieval: {e.retrieval_mode}
            {e.top_similarity != null && <> · best similarity {fmtNum(e.top_similarity)} (threshold {fmtNum(e.threshold)})</>}
            {g?.model_id && <> · generator {g.model_id}{g.seconds != null && `, ${fmtNum(g.seconds, 1)} s`}</>}
          </p>
        </>
      )}
    </section>
  );
}

/** Renders generated text, turning [L1]-style citations into clickable chips. */
function CitedText({ text }: { text: string }) {
  const parts = text.split(/(\[[A-Z]\d+(?:\s*[,;]\s*[A-Z]\d+)*\])/g);
  return (
    <>
      {parts.map((p, i) => {
        const m = p.match(/^\[(.+)\]$/);
        if (!m) return <Fragment key={i}>{p}</Fragment>;
        return <Fragment key={i}>{m[1].split(/[,;]/).map((id) => <CiteChip key={id.trim()} id={id.trim()} />)}</Fragment>;
      })}
    </>
  );
}

function ExternalSources({ items }: { items: ExternalEvidence[] }) {
  return (
    <section aria-label="Clinical sources">
      <SectionTitle aside={<span className="text-xs text-muted-foreground">retrieved passages with source details</span>}>Clinical sources</SectionTitle>
      <div className="flex flex-col gap-3">
        {items.map((d) => {
          const T = TYPE_META[d.source_type];
          return (
            <div key={d.id} id={`ev-${d.id}`} className="rounded-lg border bg-card p-4 transition-shadow">
              <div className="mb-1.5 flex flex-wrap items-center gap-2 text-sm">
                <Badge variant="outline">{d.id}</Badge>
                <Badge variant="muted"><T.icon className="h-3 w-3" /> {T.label}</Badge>
                <span className="font-medium">{d.title}</span>
              </div>
              <dl className="mb-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs text-muted-foreground sm:grid-cols-[auto_1fr_auto_1fr]">
                <dt>Section</dt><dd>{d.section}</dd>
                <dt>Source</dt><dd>{d.publisher || "–"}</dd>
                <dt>{d.source_type === "label" ? "Effective" : "Date"}</dt><dd>{formatDate(d.date)}{d.version && d.source_type === "label" ? ` · version ${d.version}` : d.version ? ` · ${d.version}` : ""}</dd>
                <dt>Identifier</dt><dd className="break-all">{d.identifier}</dd>
              </dl>
              <p className="text-sm leading-relaxed">{d.text.length > 700 ? `${d.text.slice(0, 700)}…` : d.text}</p>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
                <a href={d.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-primary hover:underline">
                  Open original <ExternalLink className="h-3 w-3" />
                </a>
                {d.dense_similarity != null && <span className="text-muted-foreground">similarity {fmtNum(d.dense_similarity)}</span>}
                {d.bm25 != null && <span className="text-muted-foreground">BM25 {fmtNum(d.bm25, 1)}</span>}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function formatDate(d: string) {
  return /^\d{8}$/.test(d) ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}` : d || "–";
}

// ------------------------------------------------------------------ sources tab

function SourcesPanel() {
  const [drug, setDrug] = useState("");
  const [type, setType] = useState("");
  const st = useQuery({ queryKey: ["evidence-status"], queryFn: () => apiGet<EvidenceStatus>("/evidence/status") });
  const docs = useQuery({
    queryKey: ["evidence-docs", drug, type],
    queryFn: () => apiGet<EvidenceDocs>("/evidence/documents", { drug: drug || undefined, source_type: type || undefined, limit: 300 }),
    enabled: st.data?.available ?? false,
  });
  return (
    <div className="mt-4 flex flex-col gap-6">
      <QueryState q={st}>
        {(s) => !s.available ? (
          <Alert variant="warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Clinical sources not indexed yet</AlertTitle>
            <AlertDescription>Run <code>scripts\build_evidence.ps1</code> once to download FDA labels, FDA safety communications and PubMed reviews and build the index. FAERS answers work without it.</AlertDescription>
          </Alert>
        ) : (
          <Card>
            <CardHeader><CardTitle>Index</CardTitle><CardDescription>Built {s.index?.created_at}; reproducible from the cached downloads in data/evidence/raw.</CardDescription></CardHeader>
            <CardContent>
              <dl className="grid grid-cols-[180px_1fr] gap-x-3 gap-y-1 text-sm">
                <dt className="text-muted-foreground">Documents</dt><dd>{fmtInt(s.index?.documents)} ({Object.entries(s.index?.documents_by_type ?? {}).map(([k, v]) => `${TYPE_META[k as SourceType]?.label ?? k}: ${v}`).join(", ")})</dd>
                <dt className="text-muted-foreground">Passages</dt><dd>{fmtInt(s.index?.chunks)}</dd>
                <dt className="text-muted-foreground">Drugs with a label</dt><dd>{fmtInt(s.index?.drugs_with_label)} of {fmtInt(s.sources?.drugs_requested)} requested</dd>
                <dt className="text-muted-foreground">Embedding model</dt><dd>{s.index?.embedding.model_id ?? `unavailable (${s.index?.embedding.reason}) – keyword search only`}{s.index?.embedding.revision && ` @ ${s.index.embedding.revision.slice(0, 7)}`}</dd>
                <dt className="text-muted-foreground">Retrieval</dt><dd>{s.index?.retrieval_modes.join(", ")}</dd>
                <dt className="text-muted-foreground">Generator</dt><dd>{s.generator.available ? s.generator.model_id : "not installed (sources are quoted instead)"}</dd>
                <dt className="text-muted-foreground">Insufficient-evidence threshold</dt><dd>similarity &lt; {fmtNum(s.sufficiency_threshold)}</dd>
                <dt className="text-muted-foreground">Content hash</dt><dd className="font-mono text-xs">{s.index?.content_sha256}</dd>
                {s.failures != null && s.failures > 0 && (<><dt className="text-muted-foreground">Download failures</dt><dd>{s.failures} (see data/evidence/sources_manifest.json)</dd></>)}
              </dl>
            </CardContent>
          </Card>
        )}
      </QueryState>
      {st.data?.available && (
        <section>
          <div className="mb-3 flex flex-wrap items-end gap-3">
            <div><Label htmlFor="src-drug">Drug</Label><Input id="src-drug" value={drug} onChange={(e) => setDrug(e.target.value.toUpperCase())} placeholder="e.g. SEMAGLUTIDE" className="w-56" /></div>
            <div><Label htmlFor="src-type">Type</Label>
              <select id="src-type" value={type} onChange={(e) => setType(e.target.value)} className="h-9 rounded-md border border-input bg-card px-2 text-sm">
                <option value="">All</option><option value="label">FDA labels</option><option value="regulatory">FDA safety communications</option><option value="literature">Published reviews</option>
              </select></div>
          </div>
          <QueryState q={docs}>
            {(d) => (
              <div className="overflow-x-auto rounded-lg border bg-card">
                <Table>
                  <THead><TR><TH>Type</TH><TH>Title</TH><TH>Source</TH><TH>Date / version</TH><TH>Drugs</TH><TH /></TR></THead>
                  <TBody>
                    {d.rows.map((x) => (
                      <TR key={x.doc_id}>
                        <TD className="whitespace-nowrap text-xs">{TYPE_META[x.source_type].label}</TD>
                        <TD className="max-w-md">{x.title}<div className="text-xs text-muted-foreground">{x.identifier}</div></TD>
                        <TD className="text-xs">{x.publisher}</TD>
                        <TD className="whitespace-nowrap text-xs">{formatDate(x.date)}{x.version && ` · ${x.version}`}</TD>
                        <TD className="max-w-[180px] text-xs">{x.drugs.split("|").slice(0, 4).join(", ")}</TD>
                        <TD><a href={x.url} target="_blank" rel="noreferrer" className="text-primary" aria-label={`Open ${x.title}`}><ExternalLink className="h-4 w-4" /></a></TD>
                      </TR>
                    ))}
                  </TBody>
                </Table>
                <p className="p-3 text-xs text-muted-foreground">{fmtInt(d.total)} documents match{d.total > d.rows.length && `; first ${d.rows.length} shown`}.</p>
              </div>
            )}
          </QueryState>
        </section>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ evidence evaluation (shown in the evaluation tab)

function IrRow({ name, m, note }: { name: string; m?: IrMetrics; note?: string }) {
  if (!m || !m.n) return null;
  return (
    <TR>
      <TD className="font-medium">{name}{note && <div className="text-xs font-normal text-muted-foreground">{note}</div>}</TD>
      <TD className="text-right tabular-nums">{pct(m.recall_at_1)}</TD>
      <TD className="text-right tabular-nums">{pct(m.recall_at_5)}{m.recall_at_5_95ci && <div className="text-xs text-muted-foreground">{ci(m.recall_at_5_95ci)}</div>}</TD>
      <TD className="text-right tabular-nums">{pct(m.recall_at_10)}</TD>
      <TD className="text-right tabular-nums">{fmtNum(m.mrr, 3)}</TD>
      <TD className="text-right tabular-nums">{fmtNum(m.ndcg_at_10, 3)}</TD>
    </TR>
  );
}

export function EvidenceEvaluationPanel() {
  const q = useQuery({ queryKey: ["evidence-eval"], queryFn: () => apiGet<EvidenceEvaluation>("/evidence/evaluation"), retry: false });
  if (q.isError) return <p className="text-sm text-muted-foreground">Clinical-source evaluation not run yet (scripts\build_evidence.ps1).</p>;
  return (
    <QueryState q={q} height="h-64">
      {(ev) => {
        const w = ev.retrieval.within_drug;
        const g = ev.generation;
        return (
          <div className="flex flex-col gap-6">
            <Card>
              <CardHeader>
                <CardTitle>4 · Finding the right label section (vector, keyword and hybrid search)</CardTitle>
                <CardDescription>
                  Gold standard: the section FDA itself coded each label passage under. {ev.retrieval.test_queries} templated questions on {ev.test_drugs} test drugs
                  (label wording and lay wording), searched within the drug's sources. {ev.dev_drugs} other drugs were used only to set the insufficient-evidence threshold.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto"><Table>
                  <THead><TR><TH>System</TH><TH className="text-right">Top-1</TH><TH className="text-right">Top-5 (95% CI)</TH><TH className="text-right">Top-10</TH><TH className="text-right">MRR</TH><TH className="text-right">nDCG@10</TH></TR></THead>
                  <TBody>
                    <IrRow name={`Dense vectors${ev.mode_selection?.selected_mode === "dense" ? " (used by the app)" : ""}`} m={w.dense?.all} note="bge-small-en-v1.5, cosine" />
                    <IrRow name={`Hybrid${ev.mode_selection?.selected_mode === "hybrid" ? " (used by the app)" : ""}`} m={w.hybrid?.all} note="vectors + BM25, reciprocal-rank fusion" />
                    <IrRow name="BM25 keywords only" m={w.bm25?.all} />
                    <IrRow name="Dense, lay wording" m={w.dense?.lay_wording} note="e.g. “Who should not take Ozempic?”" />
                    <IrRow name="Hybrid, lay wording" m={w.hybrid?.lay_wording} />
                    <IrRow name="BM25, lay wording" m={w.bm25?.lay_wording} />
                  </TBody>
                </Table></div>
                {ev.mode_selection && (
                  <Disclaimer>
                    The app's retriever was chosen on the {ev.dev_drugs} development drugs only ({ev.mode_selection.criterion}):{" "}
                    {Object.entries(ev.mode_selection.dev).map(([m, d]) => `${m} MRR ${fmtNum(d.mrr, 3)}`).join(", ")} → <strong>{ev.mode_selection.selected_mode}</strong>.
                    The table shows the untouched test drugs.
                  </Disclaimer>
                )}
              </CardContent>
            </Card>
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader><CardTitle>5 · Knowing when not to answer</CardTitle>
                  <CardDescription>Answerable questions vs questions no indexed source can answer (price, packaging colour, unrelated topics).</CardDescription></CardHeader>
                <CardContent className="text-sm">
                  {ev.sufficiency.status === "ok" && ev.sufficiency.test ? (
                    <>Threshold {fmtNum(ev.sufficiency.threshold)} (chosen on dev drugs). On test drugs: balanced accuracy <strong>{pct(ev.sufficiency.test.balanced_accuracy)}</strong>;
                      {" "}{pct(ev.sufficiency.test.answerable_kept)} of answerable questions answered, {pct(ev.sufficiency.test.unanswerable_rejected)} of unanswerable ones declined.</>
                  ) : ev.sufficiency.reason}
                </CardContent>
              </Card>
              <Card className="provisional-frame">
                <CardHeader><CardTitle>6 · Generated answers</CardTitle><CardDescription>{g.generator?.model_id ?? "local generator"} on test drugs</CardDescription></CardHeader>
                <CardContent className="text-sm">
                  {g.status === "ok" ? (
                    <ul className="list-disc pl-4">
                      <li>“Does X have a boxed warning?” ({g.n} drugs, gold = FDA label): <strong>{pct(g.boxed_warning_accuracy)}</strong> correct (95% CI {ci(g.boxed_warning_accuracy_95ci)}); retrieval-only baseline {pct(g.baseline_retrieval_only_accuracy)}.</li>
                      <li>Drafts passing all source checks: <strong>{pct(g.verifier_pass_rate)}</strong>; said “sources do not answer”: {pct(g.explicit_insufficient_rate)}.</li>
                      <li>Median time per answer: {fmtNum(g.seconds_median, 1)} s on CPU.</li>
                    </ul>
                  ) : <>Not evaluated: {g.reason ?? g.status}.</>}
                </CardContent>
              </Card>
            </div>
            {ev.concordance.status === "ok" && ev.concordance.most_reported && (
              <Card>
                <CardHeader><CardTitle>7 · Are strong FAERS signals already in the label?</CardTitle>
                  <CardDescription>Descriptive, not an accuracy measure. Strong = all three methods, ≥ 10 cases, no indication-bias flag, drug has an indexed label. {ev.concordance.caveat}</CardDescription></CardHeader>
                <CardContent className="text-sm">
                  <ul className="list-disc pl-4">
                    <li>300 most-reported strong signals: reaction term found in the label for <strong>{pct(ev.concordance.most_reported.share_reaction_text_in_label)}</strong> ({pct(ev.concordance.most_reported.share_in_label_excluding_product_use_terms)} excluding product-use terms).</li>
                    {ev.concordance.most_disproportionate && <li>300 most disproportionate (highest lower ROR bound): {pct(ev.concordance.most_disproportionate.share_reaction_text_in_label)} – the most extreme ratios are mostly unusual report clusters and terms that labels do not use.</li>}
                  </ul>
                </CardContent>
              </Card>
            )}
            <ul className="list-disc pl-4 text-xs text-muted-foreground">{ev.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
          </div>
        );
      }}
    </QueryState>
  );
}
