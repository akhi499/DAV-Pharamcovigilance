import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Search, Sparkles } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { Disclaimer, PageHeader, ProvisionalBadge, QueryState } from "@/components/common";
import { Plot, palette } from "@/components/Plot";
import { ReviewDialog } from "@/components/ReviewDialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label, NativeSelect } from "@/components/ui/input";
import { TBody, TD, TH, THead, TR, Table } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiGet, type ConceptSignals, type NlpStatus, type Prediction, type PredictionsPage, type ReviewStatus, type SearchResp } from "@/lib/api";
import { categoryLabel, useCategories } from "@/lib/categories";
import { fmtInt, fmtNum, fmtPct } from "@/lib/utils";

const REVIEW_BADGE: Record<ReviewStatus, { text: string; v: "muted" | "default" | "signal" | "danger" | "outline" }> = {
  unreviewed: { text: "Unreviewed", v: "muted" },
  accepted: { text: "Accepted", v: "default" },
  corrected: { text: "Corrected", v: "signal" },
  rejected: { text: "Rejected", v: "danger" },
  uncertain: { text: "Unsure", v: "outline" },
};

export default function Categories() {
  const [tab, setTab] = useState("overview");
  const [catFilter, setCatFilter] = useState("");
  const status = useQuery({ queryKey: ["nlp-status"], queryFn: () => apiGet<NlpStatus>("/nlp/status") });
  return (
    <>
      <PageHeader
        title="Reaction categories"
        purpose="A biomedical language model (SapBERT) reads each of the 12,678 reaction terms and places it in one of 23 body-system categories, or abstains when unsure. Use it to explore, search by meaning and review its suggestions."
      />
      <Alert variant="provisional" className="mb-6">
        <Sparkles className="h-4 w-4 text-provisional" />
        <AlertTitle>Everything outlined in violet is a provisional model output</AlertTitle>
        <AlertDescription>
          Categories are defined by this project, not MedDRA System Organ Classes. Predictions are unvalidated groupings for exploration and must
          not be used as clinical codes. Statistical signals are computed without them.
          {status.data?.provenance && (
            <span className="mt-1 block text-xs text-muted-foreground">
              {status.data.provenance.prediction_source === "neural" ? `Model ${status.data.provenance.model?.model_id} @ ${String(status.data.provenance.model?.model_revision).slice(0, 7)}` : "Lexical fallback (language model weights unavailable)"}
              {" "}· run {status.data.provenance.run_id} · taxonomy {status.data.provenance.taxonomy.version}
            </span>
          )}
        </AlertDescription>
      </Alert>
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="h-auto flex-wrap">
          <TabsTrigger value="overview">Categories</TabsTrigger>
          <TabsTrigger value="terms">Term predictions</TabsTrigger>
          <TabsTrigger value="search">Search by meaning</TabsTrigger>
          <TabsTrigger value="concept">Signals for a concept</TabsTrigger>
        </TabsList>
        <TabsContent value="overview"><CategoryOverview onPick={(id) => { setCatFilter(id); setTab("terms"); }} /></TabsContent>
        <TabsContent value="terms"><TermTable initialCategory={catFilter} /></TabsContent>
        <TabsContent value="search"><SemanticSearch /></TabsContent>
        <TabsContent value="concept"><ConceptExplorer /></TabsContent>
      </Tabs>
    </>
  );
}

function CategoryOverview({ onPick }: { onPick: (id: string) => void }) {
  const cats = useCategories();
  return (
    <QueryState q={cats} height="h-96">
      {(c) => {
        const sorted = [...c.categories].sort((a, b) => b.case_mentions - a.case_mentions);
        const total = sorted.reduce((s, x) => s + x.case_mentions, 0) + c.abstained.case_mentions;
        return (
          <div className="grid gap-6 lg:grid-cols-5">
            <div className="provisional-frame p-5 lg:sticky lg:top-6 lg:col-span-3 lg:self-start">
              <div className="mb-2 flex items-center justify-between"><h3 className="font-semibold">Reaction mentions per category</h3><ProvisionalBadge /></div>
              <Plot
                ariaLabel="Reaction mentions per provisional category"
                height={620}
                data={[{
                  type: "bar", orientation: "h", y: [...sorted.map((x) => x.label), "Abstained"].reverse(),
                  x: [...sorted.map((x) => x.case_mentions), c.abstained.case_mentions].reverse(),
                  customdata: [...sorted.map((x) => x.terms), c.abstained.terms].reverse(),
                  marker: { color: [...sorted.map(() => palette.provisional), palette.muted].reverse() },
                  hovertemplate: "%{y}: %{x:,} case mentions across %{customdata:,} terms<extra></extra>",
                }]}
                layout={{ margin: { l: 210, r: 16, t: 8, b: 40 } }}
              />
              <Disclaimer>
                {fmtPct(1 - c.abstained.case_mentions / total)} of all case–reaction mentions fall in a category; the rest belong to terms where the model abstained.
              </Disclaimer>
            </div>
            <div className="flex flex-col gap-2 lg:col-span-2">
              {sorted.map((x) => (
                <button key={x.id} onClick={() => onPick(x.id)} className="rounded-lg border bg-card p-3 text-left hover:border-provisional/60">
                  <div className="flex justify-between gap-2"><span className="font-medium">{x.label}</span><span className="text-sm tabular-nums text-muted-foreground">{fmtInt(x.terms)} terms</span></div>
                  <p className="text-sm text-muted-foreground">{x.description}</p>
                  <p className="mt-1 text-xs text-muted-foreground">Defined by: {x.anchors.slice(0, 5).join(", ")}…</p>
                </button>
              ))}
            </div>
          </div>
        );
      }}
    </QueryState>
  );
}

function TermTable({ initialCategory }: { initialCategory: string }) {
  const cats = useCategories();
  const [f, setF] = useState({ category: initialCategory, q: "", review_status: "any", disagree_only: false, sort: "cases" });
  const [offset, setOffset] = useState(0);
  const [reviewing, setReviewing] = useState<Prediction | null>(null);
  const q = useQuery({
    queryKey: ["nlp-predictions", f, offset],
    queryFn: () => apiGet<PredictionsPage>("/nlp/predictions", { ...f, limit: 50, offset }),
    placeholderData: keepPreviousData,
  });
  const set = (patch: Partial<typeof f>) => { setOffset(0); setF({ ...f, ...patch }); };
  return (
    <>
      <div className="grid gap-3 rounded-lg border bg-card p-4 sm:grid-cols-2 lg:grid-cols-5">
        <div className="flex flex-col gap-1"><Label htmlFor="t-q">Term contains</Label><Input id="t-q" value={f.q} onChange={(e) => set({ q: e.target.value })} placeholder="e.g. hepat" /></div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="t-cat">Category</Label>
          <NativeSelect id="t-cat" value={f.category} onChange={(e) => set({ category: e.target.value })}>
            <option value="">All</option><option value="abstain">Abstained</option>
            {cats.data?.categories.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
          </NativeSelect>
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="t-rs">Review status</Label>
          <NativeSelect id="t-rs" value={f.review_status} onChange={(e) => set({ review_status: e.target.value })}>
            <option value="any">Any</option>{Object.entries(REVIEW_BADGE).map(([k, v]) => <option key={k} value={k}>{v.text}</option>)}
          </NativeSelect>
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="t-sort">Sort by</Label>
          <NativeSelect id="t-sort" value={f.sort} onChange={(e) => set({ sort: e.target.value })}>
            <option value="cases">Cases</option><option value="confidence">Confidence</option><option value="margin">Margin</option><option value="score">Similarity</option>
          </NativeSelect>
        </div>
        <label className="flex items-center gap-2 self-end pb-2 text-sm">
          <input type="checkbox" className="h-4 w-4" checked={f.disagree_only} onChange={(e) => set({ disagree_only: e.target.checked })} />
          Model and baseline disagree
        </label>
      </div>
      <QueryState q={q} height="h-96">
        {(page) => (
          <div className="provisional-frame mt-4 overflow-hidden bg-card">
            <div className="flex items-center justify-between px-4 py-3 text-sm">
              <span className="flex items-center gap-2"><ProvisionalBadge /> {fmtInt(page.total)} terms</span>
              <span className="flex gap-2">
                <Button size="sm" variant="outline" disabled={offset === 0} onClick={() => setOffset(offset - 50)}>Previous</Button>
                <Button size="sm" variant="outline" disabled={offset + 50 >= page.total} onClick={() => setOffset(offset + 50)}>Next</Button>
              </span>
            </div>
            <Table>
              <THead>
                <TR><TH>Reaction term</TH><TH className="text-right">Cases</TH><TH>Model category</TH><TH>Confidence</TH><TH className="text-right">Margin</TH><TH>Lexical baseline</TH><TH>Review</TH><TH /></TR>
              </THead>
              <TBody>
                {page.rows.map((r) => (
                  <TR key={r.pt_key}>
                    <TD className="font-medium">{r.pt}<span className="block text-xs font-normal text-muted-foreground">closest anchor: {r.nearest_anchor}</span></TD>
                    <TD className="text-right tabular-nums">{fmtInt(r.cases)}</TD>
                    <TD>
                      {r.prediction === "abstain"
                        ? <span className="text-muted-foreground">Abstained <span className="text-xs">({r.abstain_reason === "ambiguous" ? `between ${categoryLabel(r.top_category, cats.data)} and ${categoryLabel(r.second_category, cats.data)}` : "not close to any category"})</span></span>
                        : <span className="text-provisional">{categoryLabel(r.prediction, cats.data)}</span>}
                      {r.review_status === "corrected" && r.reviewed_category && <span className="block text-xs">Reviewer: {categoryLabel(r.reviewed_category, cats.data)}</span>}
                    </TD>
                    <TD className="w-32">
                      <div className="h-1.5 rounded bg-muted" aria-hidden><div className="h-full rounded bg-provisional" style={{ width: `${r.confidence * 100}%` }} /></div>
                      <span className="text-xs tabular-nums text-muted-foreground">{fmtPct(r.confidence, 0)} · sim {fmtNum(r.score)}</span>
                    </TD>
                    <TD className="text-right tabular-nums">{fmtNum(r.margin, 3)}</TD>
                    <TD className={r.agrees_with_baseline ? "text-muted-foreground" : ""}>{categoryLabel(r.baseline_top_category, cats.data)}</TD>
                    <TD><Badge variant={REVIEW_BADGE[r.review_status].v}>{REVIEW_BADGE[r.review_status].text}</Badge></TD>
                    <TD><Button size="sm" variant="outline" onClick={() => setReviewing(r)}>Review</Button></TD>
                  </TR>
                ))}
                {page.rows.length === 0 && <TR><TD colSpan={8} className="py-8 text-center text-muted-foreground">No terms match. Clear a filter to see more.</TD></TR>}
              </TBody>
            </Table>
          </div>
        )}
      </QueryState>
      <Disclaimer>Confidence is a softmax over category similarities: it ranks predictions but is not a calibrated probability of being correct. Margin is the gap between the best and second-best category.</Disclaimer>
      <ReviewDialog row={reviewing} onClose={() => setReviewing(null)} />
    </>
  );
}

function SemanticSearch() {
  const [text, setText] = useState("");
  const [query, setQuery] = useState("");
  const cats = useCategories();
  const q = useQuery({ queryKey: ["nlp-search", query], queryFn: () => apiGet<SearchResp>("/nlp/search", { q: query, k: 25 }), enabled: query.length >= 2 });
  return (
    <div className="max-w-3xl">
      <p className="mb-3 text-sm text-muted-foreground">Describe a reaction in your own words. The model finds FAERS terms with the same meaning even when they share no words, for example "liver damage" finds "Hepatocellular injury".</p>
      <form className="flex gap-2" role="search" onSubmit={(e) => { e.preventDefault(); setQuery(text.trim()); }}>
        <label htmlFor="sem-q" className="sr-only">Describe a reaction</label>
        <Input id="sem-q" value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. irregular heartbeat, kidney damage, hair falling out" maxLength={200} />
        <Button type="submit" disabled={text.trim().length < 2}><Search className="h-4 w-4" /> Search</Button>
      </form>
      {query && (
        <div className="mt-4">
          <QueryState q={q}>
            {(r) => (
              <div className="provisional-frame bg-card p-4">
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span>{r.mode === "neural" ? "Matched by SapBERT meaning similarity" : "Language model unavailable: matched by spelling similarity only"}</span>
                  <ProvisionalBadge />
                </div>
                <ol className="divide-y">
                  {r.results.map((h) => (
                    <li key={h.pt_key} className="grid grid-cols-[1fr_auto] items-center gap-x-4 py-2 text-sm sm:grid-cols-[1fr_140px_120px_80px]">
                      <span className="font-medium">{h.pt}</span>
                      <span className="tabular-nums text-muted-foreground sm:text-right">{fmtInt(h.cases)} cases</span>
                      <span className="text-provisional">{categoryLabel(h.prediction, cats.data)}</span>
                      <span className="flex items-center gap-1.5 tabular-nums">
                        <span className="h-1.5 w-10 rounded bg-muted"><span className="block h-full rounded bg-provisional" style={{ width: `${Math.max(0, h.similarity) * 100}%` }} /></span>
                        {h.similarity.toFixed(2)}
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </QueryState>
        </div>
      )}
    </div>
  );
}

function ConceptExplorer() {
  const [form, setForm] = useState({ q: "", drug: "", min: "0.75" });
  const [args, setArgs] = useState<typeof form | null>(null);
  const cats = useCategories();
  const q = useQuery({
    queryKey: ["concept", args],
    queryFn: () => apiGet<ConceptSignals>("/nlp/concept-signals", { q: args!.q, drug: args!.drug || undefined, min_similarity: Number(args!.min), k: 20 }),
    enabled: !!args,
  });
  return (
    <>
      <p className="mb-3 max-w-3xl text-sm text-muted-foreground">
        One clinical idea is often split across many MedDRA terms. Type a concept: the model gathers the matching terms, then the table shows the
        unchanged statistical signals for those terms. The language model only chooses which terms to look at.
      </p>
      <form className="grid gap-3 rounded-lg border bg-card p-4 sm:grid-cols-4" onSubmit={(e) => { e.preventDefault(); if (form.q.trim().length >= 2) setArgs({ ...form }); }}>
        <div className="flex flex-col gap-1 sm:col-span-2"><Label htmlFor="c-q">Concept</Label><Input id="c-q" value={form.q} onChange={(e) => setForm({ ...form, q: e.target.value })} placeholder="e.g. pancreas inflammation" /></div>
        <div className="flex flex-col gap-1"><Label htmlFor="c-d">Drug (optional)</Label><Input id="c-d" value={form.drug} onChange={(e) => setForm({ ...form, drug: e.target.value })} placeholder="e.g. TIRZEPATIDE" /></div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="c-m">Minimum similarity</Label>
          <NativeSelect id="c-m" value={form.min} onChange={(e) => setForm({ ...form, min: e.target.value })}>
            <option value="0.85">0.85 (strict)</option><option value="0.75">0.75</option><option value="0.65">0.65 (broad)</option>
          </NativeSelect>
        </div>
        <div className="sm:col-span-4"><Button type="submit">Find signals</Button></div>
      </form>
      {args && (
        <div className="mt-4">
          <QueryState q={q}>
            {(r) => (
              <div className="grid gap-6 lg:grid-cols-3">
                <div className="provisional-frame bg-card p-4 lg:self-start">
                  <div className="mb-2 flex items-center justify-between"><h3 className="font-semibold">Terms gathered</h3><ProvisionalBadge /></div>
                  {r.terms.length === 0 && <p className="text-sm text-muted-foreground">No terms reach this similarity. Choose a broader setting.</p>}
                  <ul className="flex flex-wrap gap-1.5">
                    {r.terms.map((t) => <li key={t.pt_key}><Badge variant="provisional">{t.pt} · {t.similarity.toFixed(2)}</Badge></li>)}
                  </ul>
                  <p className="mt-3 text-xs text-muted-foreground">Mode: {r.mode === "neural" ? "SapBERT" : "lexical fallback"}. Categories: {[...new Set(r.terms.map((t) => categoryLabel(t.prediction, cats.data)))].join(", ")}</p>
                </div>
                <Card className="lg:col-span-2">
                  <CardHeader><CardTitle>Drugs with flagged pairs for this concept</CardTitle><CardDescription>Flagged by at least one method; counts summed across the gathered terms (a case can count under several terms).</CardDescription></CardHeader>
                  <CardContent>
                    <Table>
                      <THead><TR><TH>Drug</TH><TH className="text-right">Flagged terms</TH><TH className="text-right">Sum of a</TH><TH className="text-right">Max ROR lower CI</TH></TR></THead>
                      <TBody>
                        {r.by_drug.map((d) => (
                          <TR key={d.drug}>
                            <TD><Link className="underline-offset-2 hover:underline" to={`/signals?drug=${encodeURIComponent(d.drug)}&method=any&min_cases=3`}>{d.drug}</Link></TD>
                            <TD className="text-right tabular-nums">{d.flagged_terms}</TD><TD className="text-right tabular-nums">{fmtInt(d.total_cases)}</TD><TD className="text-right tabular-nums">{fmtNum(d.max_ror_lower)}</TD>
                          </TR>
                        ))}
                        {r.by_drug.length === 0 && <TR><TD colSpan={4} className="py-6 text-center text-muted-foreground">No flagged pairs for these terms.</TD></TR>}
                      </TBody>
                    </Table>
                  </CardContent>
                </Card>
              </div>
            )}
          </QueryState>
        </div>
      )}
    </>
  );
}
