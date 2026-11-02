import { useQuery } from "@tanstack/react-query";
import { Disclaimer, PageHeader, QueryState } from "@/components/common";
import { Plot, palette } from "@/components/Plot";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { apiGet, type Demographics as DemoT } from "@/lib/api";
import { fmtInt, fmtNum } from "@/lib/utils";

const AGE_ORDER = ["0-1", "2-11", "12-17", "18-44", "45-64", "65-74", "75+"];
const SEX = { F: "Female", M: "Male", Unknown: "Not reported" } as Record<string, string>;

export default function Demographics() {
  const q = useQuery({ queryKey: ["demographics"], queryFn: () => apiGet<DemoT>("/demographics") });
  return (
    <>
      <PageHeader
        title="Who the reports are about"
        purpose="Age, sex, reporter and country of the reported cases, and how the share of serious outcomes differs between subgroups. These describe reporting patterns, not who is at greater risk."
      />
      <QueryState q={q} height="h-96">
        {(d) => {
          const pyr = (sex: string) => AGE_ORDER.map((g) => d.pyramid.find((p) => p.age_group === g && p.sex === sex)?.cases ?? 0);
          const ages = d.age_group.filter((a) => a.level !== "Unknown");
          const unknownAge = d.age_group.find((a) => a.level === "Unknown");
          return (
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Age and sex</CardTitle>
                  <CardDescription>Cases with both age and sex reported. {fmtInt(unknownAge?.cases)} cases had no usable age.</CardDescription>
                </CardHeader>
                <CardContent>
                  <Plot
                    ariaLabel="Age-sex pyramid"
                    data={[
                      { type: "bar", orientation: "h", name: "Female", y: AGE_ORDER, x: pyr("F").map((v) => -v), marker: { color: palette.female }, customdata: pyr("F"), hovertemplate: "Female %{y}: %{customdata:,}<extra></extra>" },
                      { type: "bar", orientation: "h", name: "Male", y: AGE_ORDER, x: pyr("M"), marker: { color: palette.male }, hovertemplate: "Male %{y}: %{x:,}<extra></extra>" },
                    ]}
                    layout={{ barmode: "relative", xaxis: { tickformat: ",d", title: { text: "cases (female left, male right)" } }, yaxis: { title: { text: "age group (years)" } } }}
                  />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Serious outcomes by age group</CardTitle>
                  <CardDescription>Share of reported cases in each group with a serious or fatal outcome.</CardDescription>
                </CardHeader>
                <CardContent>
                  <Plot
                    ariaLabel="Serious and fatal outcome share by age group"
                    data={[
                      { type: "bar", name: "Serious", x: ages.map((a) => a.level), y: ages.map((a) => a.serious_pct), marker: { color: palette.primary }, hovertemplate: "%{x}: %{y:.1f}% serious<extra></extra>" },
                      { type: "bar", name: "Death", x: ages.map((a) => a.level), y: ages.map((a) => a.death_pct), marker: { color: palette.danger }, hovertemplate: "%{x}: %{y:.1f}% death<extra></extra>" },
                    ]}
                    layout={{ barmode: "group", yaxis: { ticksuffix: "%" } }}
                  />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Sex and seriousness</CardTitle>
                  <CardDescription>Chi-square test of independence (female vs male, serious vs not).</CardDescription>
                </CardHeader>
                <CardContent>
                  <table className="w-full text-sm">
                    <thead className="text-muted-foreground"><tr><th className="py-1 text-left font-medium">Sex</th><th className="text-right font-medium">Cases</th><th className="text-right font-medium">Serious</th><th className="text-right font-medium">Death</th></tr></thead>
                    <tbody>
                      {d.sex.map((s) => (
                        <tr key={s.level} className="border-t">
                          <td className="py-1.5">{SEX[s.level] ?? s.level}</td>
                          <td className="text-right tabular-nums">{fmtInt(s.cases)}</td>
                          <td className="text-right tabular-nums">{s.serious_pct.toFixed(1)}%</td>
                          <td className="text-right tabular-nums">{s.death_pct.toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="mt-4 text-sm">
                    χ² = {fmtNum(d.serious_by_sex.chi2, 1)}, p = {d.serious_by_sex.p_value.toExponential(1)}, Cramér's V = {fmtNum(d.serious_by_sex.cramers_v, 3)}.
                    With hundreds of thousands of reports even tiny differences are "significant"; V near 0 means the association is very weak.
                  </p>
                  <Disclaimer>{d.serious_by_sex.note}</Disclaimer>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Who reported</CardTitle>
                  <CardDescription>Reports from physicians are more often serious than consumer reports.</CardDescription>
                </CardHeader>
                <CardContent>
                  <Plot
                    ariaLabel="Cases by reporter type"
                    data={[{
                      type: "bar", x: d.reporter_type.map((r) => r.label ?? r.level), y: d.reporter_type.map((r) => r.cases),
                      marker: { color: palette.primary }, customdata: d.reporter_type.map((r) => r.serious_pct),
                      hovertemplate: "%{x}: %{y:,} cases, %{customdata:.1f}% serious<extra></extra>",
                    }]}
                  />
                </CardContent>
              </Card>
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Country where the event occurred</CardTitle>
                  <CardDescription>Top 20 by case count. FAERS includes foreign reports submitted by manufacturers.</CardDescription>
                </CardHeader>
                <CardContent>
                  <Plot
                    ariaLabel="Cases by country"
                    height={280}
                    data={[{ type: "bar", x: d.countries.map((c) => c.country), y: d.countries.map((c) => c.cases), marker: { color: palette.primary }, hovertemplate: "%{x}: %{y:,}<extra></extra>" }]}
                    layout={{ yaxis: { type: "log", title: { text: "cases (log scale)" } } }}
                  />
                  <Disclaimer>{d.note}</Disclaimer>
                </CardContent>
              </Card>
            </div>
          );
        }}
      </QueryState>
    </>
  );
}
