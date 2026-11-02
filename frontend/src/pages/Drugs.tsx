import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Disclaimer, PageHeader, ProvisionalBadge, QueryState, SectionTitle, Stat, StatStrip } from "@/components/common";
import { Plot, palette } from "@/components/Plot";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { apiGet, type DrugCount, type DrugProfile, type ReactionCount } from "@/lib/api";
import { categoryLabel, useCategories } from "@/lib/categories";
import { fmtInt, fmtPct } from "@/lib/utils";

interface CatProfile { drug: string; rows: { category: string; drug_mentions: number; all_mentions: number; drug_share: number; all_share: number }[]; note: string }

function useDebounced<T>(v: T, ms = 250) {
  const [d, setD] = useState(v);
  useEffect(() => {
    const t = setTimeout(() => setD(v), ms);
    return () => clearTimeout(t);
  }, [v, ms]);
  return d;
}

export default function Drugs() {
  const [params, setParams] = useSearchParams();
  const drug = params.get("drug") ?? "";
  const [text, setText] = useState(drug);
  const dq = useDebounced(text.trim());
  const sugg = useQuery({
    queryKey: ["drug-search", dq],
    queryFn: () => apiGet<{ drug: string; cases: number }[]>("/drugs/search", { q: dq }),
    enabled: dq.length >= 2 && dq.toUpperCase() !== drug,
  });
  const top = useQuery({ queryKey: ["top-drugs"], queryFn: () => apiGet<DrugCount[]>("/drugs/top", { limit: 20 }) });
  const topR = useQuery({ queryKey: ["top-reactions"], queryFn: () => apiGet<ReactionCount[]>("/reactions/top", { limit: 20 }) });
  const choose = (d: string) => {
    setText(d);
    setParams({ drug: d });
  };

  return (
    <>
      <PageHeader title="Drug explorer" purpose="Look up any suspect drug to see who it was reported in, its most frequent reactions and indications, then jump to its disproportionality signals." />
      <form
        className="relative mb-6 max-w-xl"
        role="search"
        onSubmit={(e) => {
          e.preventDefault();
          if (text.trim()) choose(text.trim().toUpperCase());
        }}
      >
        <label htmlFor="drug-q" className="sr-only">Drug name or active ingredient</label>
        <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" aria-hidden />
        <Input id="drug-q" className="pl-9" placeholder="Drug or active ingredient, e.g. semaglutide" value={text} onChange={(e) => setText(e.target.value)} autoComplete="off" />
        {sugg.data && sugg.data.length > 0 && dq.toUpperCase() !== drug && (
          <ul className="absolute z-20 mt-1 w-full overflow-hidden rounded-md border bg-popover shadow-md" role="listbox">
            {sugg.data.map((s) => (
              <li key={s.drug}>
                <button type="button" className="flex w-full justify-between px-3 py-2 text-left text-sm hover:bg-accent" onClick={() => choose(s.drug)}>
                  <span>{s.drug}</span>
                  <span className="tabular-nums text-muted-foreground">{fmtInt(s.cases)}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </form>
      {drug ? <Profile drug={drug} /> : (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader><CardTitle>Most reported suspect drugs</CardTitle><CardDescription>Select a bar to open its profile.</CardDescription></CardHeader>
            <CardContent>
              <QueryState q={top}>
                {(rows) => (
                  <ul className="flex flex-col">
                    {rows.map((r) => (
                      <li key={r.drug}>
                        <button className="grid w-full grid-cols-[1fr_auto] gap-2 rounded px-2 py-1.5 text-left text-sm hover:bg-accent" onClick={() => choose(r.drug)}>
                          <span className="truncate">{r.drug}</span>
                          <span className="tabular-nums text-muted-foreground">{fmtInt(r.cases)}</span>
                          <span className="col-span-2 h-1.5 rounded bg-muted"><span className="block h-full rounded bg-primary" style={{ width: `${(r.cases / rows[0].cases) * 100}%` }} /></span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </QueryState>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Most reported reaction terms</CardTitle><CardDescription>Cases mentioning each MedDRA preferred term.</CardDescription></CardHeader>
            <CardContent>
              <QueryState q={topR}>
                {(rows) => (
                  <Plot
                    ariaLabel="Most reported reaction terms"
                    height={520}
                    data={[{ type: "bar", orientation: "h", y: rows.map((r) => r.pt).reverse(), x: rows.map((r) => r.cases).reverse(), marker: { color: palette.primary }, hovertemplate: "%{y}: %{x:,}<extra></extra>" }]}
                    layout={{ margin: { l: 200, r: 16, t: 8, b: 36 } }}
                  />
                )}
              </QueryState>
              <Disclaimer>Several top terms ("Off label use", "Drug ineffective", "Product dose omission issue") describe product use rather than a bodily reaction.</Disclaimer>
            </CardContent>
          </Card>
        </div>
      )}
    </>
  );
}

function Profile({ drug }: { drug: string }) {
  const p = useQuery({ queryKey: ["drug-profile", drug], queryFn: () => apiGet<DrugProfile>(`/drugs/${encodeURIComponent(drug)}/profile`) });
  const cp = useQuery({ queryKey: ["drug-cat-profile", drug], queryFn: () => apiGet<CatProfile>("/nlp/drug-profile", { drug }), retry: false });
  const cats = useCategories();
  return (
    <QueryState q={p} height="h-96">
      {(d) => (
        <>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 className="font-serif text-2xl font-semibold">{d.drug}</h2>
            <Button asChild variant="outline"><Link to={`/signals?drug=${encodeURIComponent(d.drug)}`}>View signals for this drug</Link></Button>
          </div>
          <StatStrip>
            <Stat label="Cases (suspect role)" value={fmtInt(d.cases)} />
            <Stat label="Serious" value={fmtPct(d.serious / d.cases)} hint={`${fmtInt(d.serious)} cases`} />
            <Stat label="Death reported" value={fmtInt(d.deaths)} tone="danger" />
            <Stat label="Median age" value={d.median_age == null ? "–" : `${Math.round(d.median_age)} y`} />
          </StatStrip>
          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader><CardTitle>Most frequent reactions</CardTitle><CardDescription>Raw counts; see Signal detection for disproportionality.</CardDescription></CardHeader>
              <CardContent>
                <Plot ariaLabel={`Top reactions for ${d.drug}`} height={480}
                  data={[{ type: "bar", orientation: "h", y: d.top_reactions.map((r) => r.pt).reverse(), x: d.top_reactions.map((r) => r.cases).reverse(), marker: { color: palette.primary }, hovertemplate: "%{y}: %{x:,}<extra></extra>" }]}
                  layout={{ margin: { l: 200, r: 16, t: 8, b: 36 } }} />
              </CardContent>
            </Card>
            <div className="flex flex-col gap-6">
              <Card>
                <CardHeader><CardTitle>Reported indications</CardTitle><CardDescription>Why the drug was being taken, when stated.</CardDescription></CardHeader>
                <CardContent>
                  <ul className="text-sm">
                    {d.top_indications.map((i) => (
                      <li key={i.indi_pt} className="flex justify-between border-t py-1.5 first:border-t-0"><span>{i.indi_pt}</span><span className="tabular-nums text-muted-foreground">{fmtInt(i.cases)}</span></li>
                    ))}
                    {d.top_indications.length === 0 && <li className="text-muted-foreground">No indications reported.</li>}
                  </ul>
                </CardContent>
              </Card>
              <div className="provisional-frame p-5">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <h3 className="font-semibold">Reaction mix by category</h3>
                  <ProvisionalBadge />
                </div>
                <p className="mb-3 text-sm text-muted-foreground">Share of this drug's reaction mentions in each model-assigned category, against the whole database.</p>
                <QueryState q={cp}>
                  {(c) => {
                    const rows = c.rows.filter((r) => r.drug_mentions > 0).slice(0, 10);
                    return (
                      <Plot ariaLabel="Category profile" height={300}
                        data={[
                          { type: "bar", orientation: "h", name: d.drug, y: rows.map((r) => categoryLabel(r.category, cats.data)).reverse(), x: rows.map((r) => r.drug_share * 100).reverse(), marker: { color: palette.provisional }, hovertemplate: "%{y}: %{x:.1f}%<extra></extra>" },
                          { type: "bar", orientation: "h", name: "All drugs", y: rows.map((r) => categoryLabel(r.category, cats.data)).reverse(), x: rows.map((r) => r.all_share * 100).reverse(), marker: { color: palette.muted }, hovertemplate: "%{y}: %{x:.1f}%<extra></extra>" },
                        ]}
                        layout={{ barmode: "group", xaxis: { ticksuffix: "%" }, margin: { l: 170, r: 16, t: 8, b: 40 } }} />
                    );
                  }}
                </QueryState>
                <Disclaimer>Exploratory only. Categories are provisional model outputs and this is not a disproportionality statistic.</Disclaimer>
              </div>
            </div>
          </div>
          <SectionTitle>Reported age and sex</SectionTitle>
          <Card>
            <CardContent className="pt-5">
              <Plot ariaLabel="Age and sex of cases" height={260}
                data={["F", "M", "Unknown"].map((s, i) => ({
                  type: "bar" as const, name: s === "F" ? "Female" : s === "M" ? "Male" : "Not reported",
                  x: ["0-1", "2-11", "12-17", "18-44", "45-64", "65-74", "75+", "Unknown"],
                  y: ["0-1", "2-11", "12-17", "18-44", "45-64", "65-74", "75+", "Unknown"].map((g) => d.demographics.find((x) => x.sex === s && x.age_group === g)?.cases ?? 0),
                  marker: { color: [palette.female, palette.male, palette.muted][i] },
                }))}
                layout={{ barmode: "stack" }} />
            </CardContent>
          </Card>
        </>
      )}
    </QueryState>
  );
}
