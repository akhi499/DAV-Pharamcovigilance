import { useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { Disclaimer, PageHeader, QueryState, SectionTitle, Stat, StatStrip } from "@/components/common";
import { Plot, palette } from "@/components/Plot";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiGet, type Demographics, type Overview as OverviewT, type TrendRow } from "@/lib/api";
import { fmtInt, fmtPct } from "@/lib/utils";

export default function Overview() {
  const ov = useQuery({ queryKey: ["overview"], queryFn: () => apiGet<OverviewT>("/overview") });
  const trend = useQuery({ queryKey: ["trend"], queryFn: () => apiGet<TrendRow[]>("/trend") });
  const demo = useQuery({ queryKey: ["demographics"], queryFn: () => apiGet<Demographics>("/demographics") });

  return (
    <>
      <PageHeader
        title="Adverse event reports, 2025 Q1"
        purpose="What the FDA received between January and March 2025, how the raw extract was reduced to an analysable set of cases, and what share of reports were serious."
      />
      <QueryState q={ov} height="h-28">
        {(d) => (
          <>
            <StatStrip>
              <Stat label="Cases after de-duplication" value={fmtInt(d.kpis.cases)} hint={`${d.kpis.first_fda_date.slice(0, 7)} to ${d.kpis.last_fda_date}`} />
              <Stat label="Serious outcome" value={fmtPct(d.kpis.serious / d.kpis.cases)} hint={`${fmtInt(d.kpis.serious)} cases`} />
              <Stat label="Death reported" value={fmtInt(d.kpis.deaths)} tone="danger" hint={fmtPct(d.kpis.deaths / d.kpis.cases) + " of cases"} />
              <Stat label="Suspect drugs / reaction terms" value={`${fmtInt(d.kpis.distinct_suspect_drugs)} / ${fmtInt(d.kpis.distinct_reaction_terms)}`} hint="distinct names / MedDRA PTs" />
            </StatStrip>

            <div className="mt-6 grid gap-6 lg:grid-cols-5">
              <Card className="lg:col-span-3">
                <CardHeader>
                  <CardTitle>From raw extract to analysis population</CardTitle>
                  <CardDescription>Each step removes reports that would double-count the same patient event.</CardDescription>
                </CardHeader>
                <CardContent>
                  {d.dedup && d.signals_summary ? <CaseFlow d={d} /> : <p className="text-sm">Run the pipeline to see the case flow.</p>}
                </CardContent>
              </Card>
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Reported outcomes</CardTitle>
                  <CardDescription>A case can carry more than one outcome.</CardDescription>
                </CardHeader>
                <CardContent>
                  <QueryState q={demo}>
                    {(dm) => (
                      <Plot
                        ariaLabel="Bar chart of reported outcomes"
                        height={300}
                        data={[{
                          type: "bar", orientation: "h",
                          y: dm.outcomes.map((o) => o.label).reverse(), x: dm.outcomes.map((o) => o.cases).reverse(),
                          marker: { color: dm.outcomes.map((o) => (o.code === "DE" ? palette.danger : palette.primary)).reverse() },
                          hovertemplate: "%{y}: %{x:,} cases<extra></extra>",
                        }]}
                        layout={{ margin: { l: 140, r: 16, t: 8, b: 36 } }}
                      />
                    )}
                  </QueryState>
                </CardContent>
              </Card>
            </div>

            <SectionTitle>Reports received per day</SectionTitle>
            <Card>
              <CardContent className="pt-5">
                <QueryState q={trend}>
                  {(t) => {
                    const rows = t.filter((r) => r.date >= "2025-01-01");
                    return (
                      <Plot
                        ariaLabel="Daily reports received by FDA"
                        height={260}
                        data={[
                          { type: "scatter", mode: "lines", name: "All cases", x: rows.map((r) => r.date), y: rows.map((r) => r.cases), line: { color: palette.primary, width: 1.6 } },
                          { type: "scatter", mode: "lines", name: "Serious", x: rows.map((r) => r.date), y: rows.map((r) => r.serious), line: { color: palette.signal, width: 1.6 } },
                        ]}
                        layout={{ yaxis: { title: { text: "cases" } } }}
                      />
                    );
                  }}
                </QueryState>
                <Disclaimer>Receipt date (FDA_DT), not event date. Weekly dips are weekends; spikes usually reflect batch submissions by manufacturers.</Disclaimer>
              </CardContent>
            </Card>

            <Alert variant="info" className="mt-6">
              <Info className="h-4 w-4" />
              <AlertDescription>{d.disclaimer}</AlertDescription>
            </Alert>
            <p className="mt-3 break-all text-xs text-muted-foreground">
              Source: {d.manifest.source_zip} (SHA-256 {d.manifest.source_sha256}), ingested {d.manifest.ingested_at}.
            </p>
          </>
        )}
      </QueryState>
    </>
  );
}

function CaseFlow({ d }: { d: OverviewT }) {
  const s = d.dedup!;
  const sig = d.signals_summary!;
  const steps = [
    { label: "Reports in DEMO file", n: s.demo_rows_raw, note: "" },
    { label: "Minus FDA-deleted cases", n: s.demo_rows_raw - s.rows_removed_deleted, note: `−${fmtInt(s.rows_removed_deleted)}` },
    { label: "Latest version per case", n: s.cases_after_dedup, note: `−${fmtInt(s.rows_removed_older_versions)}` },
    { label: "Minus probable duplicates", n: s.cases_after_dedup - s.probable_duplicates_flagged, note: `−${fmtInt(s.probable_duplicates_flagged)}` },
    { label: "Analysis population", n: sig.analysis_population_cases, note: "has suspect drug + reaction" },
  ];
  const max = steps[0].n;
  return (
    <ol className="flex flex-col gap-2.5" aria-label="Case flow">
      {steps.map((st, i) => (
        <li key={st.label} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 sm:grid-cols-[180px_1fr_110px]">
          <span className="text-sm">{st.label}</span>
          <span className="order-3 col-span-2 h-6 rounded bg-muted sm:order-none sm:col-span-1">
            <span
              className="block h-full rounded"
              style={{ width: `${Math.max(2, (st.n / max) * 100)}%`, background: i === steps.length - 1 ? palette.primary : "hsl(var(--primary) / 0.45)" }}
            />
          </span>
          <span className="text-right text-sm tabular-nums">
            <span className="font-semibold">{fmtInt(st.n)}</span>
            {st.note && <span className="block text-xs text-muted-foreground">{st.note}</span>}
          </span>
        </li>
      ))}
    </ol>
  );
}
