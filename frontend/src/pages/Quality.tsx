import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, Info, TriangleAlert } from "lucide-react";
import { PageHeader, QueryState, SectionTitle } from "@/components/common";
import { Plot, palette } from "@/components/Plot";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TBody, TD, TH, THead, TR, Table } from "@/components/ui/table";
import { apiGet, type Quality as QualityT } from "@/lib/api";
import { fmtInt, titleCase } from "@/lib/utils";

const FIELD_LABELS: Record<string, string> = {
  age_years: "Age", weight_kg: "Weight", event_date: "Event date", sex: "Sex", reporter_type: "Reporter type", occr_country: "Country",
};
const STATUS = {
  pass: { icon: CheckCircle2, variant: "default" as const, text: "Pass" },
  warn: { icon: TriangleAlert, variant: "signal" as const, text: "Disclose" },
  info: { icon: Info, variant: "muted" as const, text: "Info" },
};

export default function Quality() {
  const q = useQuery({ queryKey: ["quality"], queryFn: () => apiGet<QualityT>("/quality") });
  return (
    <>
      <PageHeader
        title="Data quality"
        purpose="Spontaneous reports are often incomplete. These checks show what is missing or implausible, and how each issue is handled before analysis."
      />
      <QueryState q={q}>
        {(d) => {
          const fields = Object.entries(d.field_missing_pct).sort((a, b) => b[1] - a[1]);
          const dims = [...new Set(d.checks.map((c) => c.dimension))];
          return (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Missing values by field</CardTitle>
                  <CardDescription>Share of {fmtInt(d.n_cases)} de-duplicated cases where the field is empty or unusable.</CardDescription>
                </CardHeader>
                <CardContent>
                  <Plot
                    ariaLabel="Missing values by field"
                    height={260}
                    data={[{
                      type: "bar", orientation: "h",
                      y: fields.map(([k]) => FIELD_LABELS[k] ?? k).reverse(), x: fields.map(([, v]) => v).reverse(),
                      marker: { color: fields.map(([, v]) => (v > 20 ? palette.signal : palette.primary)).reverse() },
                      text: fields.map(([, v]) => `${v.toFixed(1)}%`).reverse(), textposition: "outside",
                      hovertemplate: "%{y}: %{x:.1f}% missing<extra></extra>",
                    }]}
                    layout={{ xaxis: { range: [0, 100], ticksuffix: "%" }, margin: { l: 110, r: 40, t: 8, b: 36 } }}
                  />
                </CardContent>
              </Card>
              {dims.map((dim) => (
                <section key={dim}>
                  <SectionTitle>{titleCase(dim)}</SectionTitle>
                  <Card>
                    <Table>
                      <THead>
                        <TR>
                          <TH className="w-28">Status</TH>
                          <TH>Check</TH>
                          <TH className="text-right">Value</TH>
                          <TH>Handling</TH>
                        </TR>
                      </THead>
                      <TBody>
                        {d.checks.filter((c) => c.dimension === dim).map((c) => {
                          const s = STATUS[c.status];
                          return (
                            <TR key={c.check}>
                              <TD><Badge variant={s.variant}><s.icon className="h-3 w-3" aria-hidden />{s.text}</Badge></TD>
                              <TD>{c.check}</TD>
                              <TD className="whitespace-nowrap text-right tabular-nums">
                                {c.value == null ? "–" : c.unit.startsWith("%") ? `${c.value.toFixed(2)}%` : fmtInt(c.value)}
                                <span className="ml-1 text-xs text-muted-foreground">{c.unit.startsWith("%") ? "" : c.unit}</span>
                              </TD>
                              <TD className="text-muted-foreground">{c.detail}</TD>
                            </TR>
                          );
                        })}
                      </TBody>
                    </Table>
                  </Card>
                </section>
              ))}
            </>
          );
        }}
      </QueryState>
    </>
  );
}
