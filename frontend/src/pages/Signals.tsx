import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Disclaimer, PageHeader, ProvisionalBadge, QueryState, Stat, StatStrip } from "@/components/common";
import { Plot, palette } from "@/components/Plot";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input, Label, NativeSelect } from "@/components/ui/input";
import { TBody, TD, TH, THead, TR, Table } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiGet, type SignalRow, type SignalsPage, type SignalsSummary, type Stratum } from "@/lib/api";
import { categoryLabel, useCategories } from "@/lib/categories";
import { fmtInt, fmtNum, fmtPct } from "@/lib/utils";

const PAGE = 50;

export default function Signals() {
  const [params, setParams] = useSearchParams();
  const [form, setForm] = useState({
    drug: params.get("drug") ?? "",
    pt: params.get("pt") ?? "",
    method: params.get("method") ?? "all",
    min_cases: params.get("min_cases") ?? "5",
    exclude_indication_bias: params.get("xib") !== "0",
    category: params.get("category") ?? "",
    hide_product_use: params.get("xpu") === "1",
    sort: params.get("sort") ?? "a",
  });
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<SignalRow | null>(null);
  const filters = useMemo(() => ({
    drug: params.get("drug") ?? undefined,
    pt: params.get("pt") ?? undefined,
    method: params.get("method") ?? "all",
    min_cases: Number(params.get("min_cases") ?? 5),
    exclude_indication_bias: params.get("xib") !== "0",
    category: params.get("category") ?? undefined,
    exclude_category: params.get("xpu") === "1" ? "product_use" : undefined,
    sort: params.get("sort") ?? "a",
  }), [params]);

  const summary = useQuery({ queryKey: ["signals-summary"], queryFn: () => apiGet<SignalsSummary>("/signals/summary") });
  const cats = useCategories();
  const rows = useQuery({
    queryKey: ["signals", filters, offset],
    queryFn: () => apiGet<SignalsPage>("/signals", { ...filters, only_signals: true, limit: PAGE, offset }),
    placeholderData: keepPreviousData,
  });

  const apply = (e: React.FormEvent) => {
    e.preventDefault();
    const p: Record<string, string> = {};
    if (form.drug.trim()) p.drug = form.drug.trim().toUpperCase();
    if (form.pt.trim()) p.pt = form.pt.trim();
    p.method = form.method;
    p.min_cases = String(Math.max(3, Number(form.min_cases) || 3));
    if (!form.exclude_indication_bias) p.xib = "0";
    if (form.category) p.category = form.category;
    if (form.hide_product_use) p.xpu = "1";
    p.sort = form.sort;
    setOffset(0);
    setParams(p);
  };

  return (
    <>
      <PageHeader
        title="Signal detection"
        purpose="Drug–reaction pairs reported together more often than expected across the rest of the database, measured with ROR, PRR and the information component (IC)."
      />
      <QueryState q={summary} height="h-24">
        {(s) => (
          <StatStrip>
            <Stat label="Analysis population" value={fmtInt(s.analysis_population_cases)} hint="cases" />
            <Stat label="Pairs tested (≥3 cases)" value={fmtInt(s.pairs_tested_min_cases)} hint={`of ${fmtInt(s.drug_event_pairs_observed)} observed`} />
            <Stat label="Flagged by all three methods" value={fmtInt(s.all_three_methods)} tone="signal" hint={fmtPct(s.all_three_methods / s.pairs_tested_min_cases) + " of tested"} />
            <Stat label="Possible indication bias" value={fmtInt(s.possible_indication_bias_rows)} hint="reaction also listed as indication" />
          </StatStrip>
        )}
      </QueryState>

      <Alert variant="info" className="mt-4">
        <Info className="h-4 w-4" />
        <AlertDescription>
          A flagged pair is a hypothesis for review, not proof of harm. Ratios compare reporting, not risk: FAERS has no count of exposed patients,
          so incidence and absolute risk cannot be estimated, and reporting is influenced by publicity, litigation and stimulated reporting.
        </AlertDescription>
      </Alert>

      <form onSubmit={apply} className="mt-6 grid gap-3 rounded-lg border bg-card p-4 sm:grid-cols-2 lg:grid-cols-4" aria-label="Signal filters">
        <div className="flex flex-col gap-1"><Label htmlFor="f-drug">Drug</Label><Input id="f-drug" value={form.drug} placeholder="Any drug" onChange={(e) => setForm({ ...form, drug: e.target.value })} /></div>
        <div className="flex flex-col gap-1"><Label htmlFor="f-pt">Reaction contains</Label><Input id="f-pt" value={form.pt} placeholder="e.g. pancreatitis" onChange={(e) => setForm({ ...form, pt: e.target.value })} /></div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="f-method">Flagged by</Label>
          <NativeSelect id="f-method" value={form.method} onChange={(e) => setForm({ ...form, method: e.target.value })}>
            <option value="all">All three methods</option><option value="any">Any method</option>
            <option value="ror">ROR (lower CI &gt; 1)</option><option value="prr">PRR (Evans)</option><option value="ic">IC025 &gt; 0</option>
          </NativeSelect>
        </div>
        <div className="flex flex-col gap-1"><Label htmlFor="f-min">Minimum cases (a)</Label><Input id="f-min" type="number" min={3} value={form.min_cases} onChange={(e) => setForm({ ...form, min_cases: e.target.value })} /></div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="f-cat" className="flex items-center gap-2">Reaction category <ProvisionalBadge /></Label>
          <NativeSelect id="f-cat" value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })} disabled={cats.isError}>
            <option value="">Any category</option>
            {cats.data?.categories.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
          </NativeSelect>
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="f-sort">Sort by</Label>
          <NativeSelect id="f-sort" value={form.sort} onChange={(e) => setForm({ ...form, sort: e.target.value })}>
            <option value="a">Number of cases</option><option value="ror_lower">ROR lower 95% CI</option><option value="ic025">IC025</option>
            <option value="prr">PRR</option><option value="chi2">Chi-square</option>
          </NativeSelect>
        </div>
        <label className="flex items-center gap-2 self-end pb-2 text-sm">
          <input type="checkbox" className="h-4 w-4 accent-[hsl(var(--primary))]" checked={form.exclude_indication_bias} onChange={(e) => setForm({ ...form, exclude_indication_bias: e.target.checked })} />
          Hide likely indication bias
        </label>
        <label className="flex items-center gap-2 self-end pb-2 text-sm">
          <input type="checkbox" className="h-4 w-4" checked={form.hide_product_use} onChange={(e) => setForm({ ...form, hide_product_use: e.target.checked })} disabled={cats.isError} />
          Hide product-use and error terms <ProvisionalBadge />
        </label>
        <div className="flex items-end lg:col-start-4"><Button type="submit" className="w-full">Apply filters</Button></div>
      </form>

      <QueryState q={rows} height="h-96">
        {(page) => (
          <>
            {page.rows.length > 0 && (
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle>Reporting odds ratios with 95% confidence intervals</CardTitle>
                  <CardDescription>First {Math.min(25, page.rows.length)} rows of the table below, log scale. The dashed line marks ROR = 1 (no disproportion).</CardDescription>
                </CardHeader>
                <CardContent><Forest rows={page.rows.slice(0, 25)} /></CardContent>
              </Card>
            )}
            <Card className="mt-6">
              <div className="flex items-center justify-between px-4 py-3 text-sm">
                <span>{fmtInt(page.total)} pairs match</span>
                <span className="flex gap-2">
                  <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</Button>
                  <Button variant="outline" size="sm" disabled={offset + PAGE >= page.total} onClick={() => setOffset(offset + PAGE)}>Next</Button>
                </span>
              </div>
              <Table>
                <THead>
                  <TR>
                    <TH>Drug</TH><TH>Reaction</TH><TH className="text-right">Cases (a)</TH><TH className="text-right">Expected</TH>
                    <TH className="text-right">ROR (95% CI)</TH><TH className="text-right">PRR</TH><TH className="text-right">IC025</TH><TH>Notes</TH>
                  </TR>
                </THead>
                <TBody>
                  {page.rows.map((r) => (
                    <TR key={r.drug + r.pt_key} className="cursor-pointer hover:bg-accent/60" tabIndex={0} onClick={() => setSelected(r)} onKeyDown={(e) => e.key === "Enter" && setSelected(r)}>
                      <TD className="font-medium">{r.drug}</TD>
                      <TD>
                        {r.pt}
                        {r.nlp_category && r.nlp_category !== "abstain" && <span className="ml-2 text-xs text-provisional">{categoryLabel(r.nlp_category, cats.data)}</span>}
                      </TD>
                      <TD className="text-right tabular-nums">{fmtInt(r.a)}</TD>
                      <TD className="text-right tabular-nums text-muted-foreground">{fmtNum(r.expected, 1)}</TD>
                      <TD className="whitespace-nowrap text-right tabular-nums">{fmtNum(r.ror)} <span className="text-muted-foreground">({fmtNum(r.ror_lower)}–{fmtNum(r.ror_upper)})</span></TD>
                      <TD className="text-right tabular-nums">{fmtNum(r.prr)}</TD>
                      <TD className="text-right tabular-nums">{fmtNum(r.ic025)}</TD>
                      <TD className="space-x-1 whitespace-nowrap">
                        {r.possible_indication_bias && <Badge variant="signal">Indication?</Badge>}
                        {r.haldane_applied && <Badge variant="muted">+0.5</Badge>}
                      </TD>
                    </TR>
                  ))}
                  {page.rows.length === 0 && (
                    <TR><TD colSpan={8} className="py-8 text-center text-muted-foreground">No pairs match these filters. Lower the minimum cases or choose "Any method".</TD></TR>
                  )}
                </TBody>
              </Table>
            </Card>
            <Disclaimer>
              "Indication?" means at least 20% of these cases also list the reaction as the reason for treatment (confounding by indication).
              "+0.5" means a zero cell was corrected with the Haldane–Anscombe adjustment, so the estimate is unstable.
              q-values (Benjamini–Hochberg) are available in the detail view.
            </Disclaimer>
          </>
        )}
      </QueryState>
      <SignalDetail row={selected} onClose={() => setSelected(null)} />
    </>
  );
}

function Forest({ rows }: { rows: SignalRow[] }) {
  const labels = rows.map((r) => `${r.drug.slice(0, 22)} · ${r.pt.slice(0, 28)}`).reverse();
  const rr = [...rows].reverse();
  return (
    <Plot
      ariaLabel="Forest plot of ROR"
      height={Math.max(260, rows.length * 22 + 60)}
      data={[{
        type: "scatter", mode: "markers", x: rr.map((r) => r.ror), y: labels,
        error_x: { type: "data", symmetric: false, array: rr.map((r) => r.ror_upper - r.ror), arrayminus: rr.map((r) => r.ror - r.ror_lower), color: palette.primary, thickness: 1.4, width: 0 },
        marker: { color: rr.map((r) => (r.possible_indication_bias ? palette.signal : palette.primary)), size: rr.map((r) => 5 + Math.min(10, Math.log10(r.a + 1) * 3)) },
        customdata: rr.map((r) => [r.a, r.ror_lower, r.ror_upper]),
        hovertemplate: "%{y}<br>ROR %{x:.2f} (%{customdata[1]:.2f}–%{customdata[2]:.2f})<br>a = %{customdata[0]:,}<extra></extra>",
      }]}
      layout={{
        xaxis: { type: "log", title: { text: "ROR (log scale)" } },
        margin: { l: 300, r: 16, t: 8, b: 44 },
        shapes: [{ type: "line", x0: 1, x1: 1, yref: "paper", y0: 0, y1: 1, line: { dash: "dash", color: palette.muted } }],
      }}
    />
  );
}

function SignalDetail({ row, onClose }: { row: SignalRow | null; onClose: () => void }) {
  const [by, setBy] = useState<"sex" | "age_group">("sex");
  const strat = useQuery({
    queryKey: ["strat", row?.drug, row?.pt_key, by],
    queryFn: () => apiGet<{ strata: Stratum[]; note: string }>("/signals/stratified", { drug: row!.drug, pt_key: row!.pt_key, by }),
    enabled: !!row,
  });
  const cats = useCategories();
  if (!row) return null;
  return (
    <Dialog open={!!row} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
        <DialogTitle>{row.drug} and {row.pt}</DialogTitle>
        <DialogDescription>Case-level 2×2 table over the analysis population and every disproportionality measure.</DialogDescription>
        <div className="grid gap-5 md:grid-cols-2">
          <table className="text-sm" aria-label="Two by two contingency table">
            <thead><tr><th /><th className="px-2 py-1 text-right font-medium">{row.pt.slice(0, 24)}</th><th className="px-2 py-1 text-right font-medium">Other reactions</th></tr></thead>
            <tbody className="tabular-nums">
              <tr className="border-t"><th className="py-1 text-left font-medium">{row.drug.slice(0, 24)}</th><td className="px-2 text-right font-semibold">a = {fmtInt(row.a)}</td><td className="px-2 text-right">b = {fmtInt(row.b)}</td></tr>
              <tr className="border-t"><th className="py-1 text-left font-medium">Other drugs</th><td className="px-2 text-right">c = {fmtInt(row.c)}</td><td className="px-2 text-right">d = {fmtInt(row.d)}</td></tr>
            </tbody>
          </table>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm tabular-nums">
            <dt className="text-muted-foreground">ROR (95% CI)</dt><dd>{fmtNum(row.ror)} ({fmtNum(row.ror_lower)}–{fmtNum(row.ror_upper)})</dd>
            <dt className="text-muted-foreground">PRR (95% CI)</dt><dd>{fmtNum(row.prr)} ({fmtNum(row.prr_lower)}–{fmtNum(row.prr_upper)})</dd>
            <dt className="text-muted-foreground">χ² (Yates)</dt><dd>{fmtNum(row.chi2, 1)}</dd>
            <dt className="text-muted-foreground">IC / IC025</dt><dd>{fmtNum(row.ic)} / {fmtNum(row.ic025)}</dd>
            <dt className="text-muted-foreground">q-value (BH)</dt><dd>{row.q_value === 0 ? "< 1e-300" : row.q_value < 1e-4 ? row.q_value.toExponential(1) : fmtNum(row.q_value, 4)}</dd>
            <dt className="text-muted-foreground">Expected a</dt><dd>{fmtNum(row.expected, 1)}</dd>
            <dt className="text-muted-foreground">Indication overlap</dt><dd>{fmtPct(row.indication_overlap)}</dd>
            <dt className="text-muted-foreground">Methods flagging</dt><dd>{row.methods_flagging} of 3</dd>
          </dl>
        </div>
        {row.nlp_category && (
          <div className="provisional-frame flex flex-wrap items-center gap-2 px-3 py-2 text-sm">
            <ProvisionalBadge /> Model category: <strong>{categoryLabel(row.nlp_category, cats.data)}</strong>
            {row.nlp_confidence != null && <span className="text-muted-foreground">(relative confidence {fmtPct(row.nlp_confidence, 0)})</span>}
          </div>
        )}
        <Tabs value={by} onValueChange={(v) => setBy(v as "sex" | "age_group")}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-semibold">Subgroup analysis</h3>
            <TabsList><TabsTrigger value="sex">By sex</TabsTrigger><TabsTrigger value="age_group">By age group</TabsTrigger></TabsList>
          </div>
          <TabsContent value={by}>
            <QueryState q={strat}>
              {(s) => {
                const est = s.strata.filter((x) => x.estimable);
                return (
                  <>
                    <Plot
                      ariaLabel="Stratified ROR"
                      height={Math.max(180, est.length * 34 + 60)}
                      data={[{
                        type: "scatter", mode: "markers", y: est.map((x) => x.stratum), x: est.map((x) => x.ror),
                        error_x: { type: "data", symmetric: false, array: est.map((x) => x.ror_upper - x.ror), arrayminus: est.map((x) => x.ror - x.ror_lower), color: palette.primary, width: 0 },
                        marker: { color: palette.primary, size: 8 }, customdata: est.map((x) => x.a),
                        hovertemplate: "%{y}: ROR %{x:.2f}, a = %{customdata}<extra></extra>",
                      }]}
                      layout={{ xaxis: { type: "log", title: { text: "ROR within subgroup" } }, margin: { l: 90, r: 16, t: 8, b: 44 },
                        shapes: [{ type: "line", x0: 1, x1: 1, yref: "paper", y0: 0, y1: 1, line: { dash: "dash", color: palette.muted } }] }}
                    />
                    <p className="text-xs text-muted-foreground">
                      Subgroups with fewer than 3 cases are hidden: {s.strata.filter((x) => !x.estimable).map((x) => `${x.stratum} (a=${x.a})`).join(", ") || "none"}.
                      Overlapping intervals mean the data cannot distinguish the subgroups.
                    </p>
                  </>
                );
              }}
            </QueryState>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
