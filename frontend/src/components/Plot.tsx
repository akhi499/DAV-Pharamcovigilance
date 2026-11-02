import Plotly from "plotly.js-dist-min";
import { useEffect, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import type { Data, Layout } from "plotly.js";

const PlotlyComponent = createPlotlyComponent(Plotly);

function useDark() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(document.documentElement.classList.contains("dark")));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return dark;
}

export const palette = {
  primary: "#0E6170",
  primaryDark: "#4CC3D0",
  signal: "#B7791F",
  danger: "#B42318",
  provisional: "#6D4FB0",
  muted: "#8A97A6",
  female: "#0E6170",
  male: "#B7791F",
};

interface Props {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
  ariaLabel: string;
}

/** Thin themed wrapper around Plotly so every chart shares fonts, colours and margins. */
export function Plot({ data, layout, height = 320, ariaLabel }: Props) {
  const dark = useDark();
  const fg = dark ? "#DCE3EA" : "#16202C";
  const grid = dark ? "#2A3542" : "#E1E6EC";
  const base: Partial<Layout> = {
    height,
    autosize: true,
    margin: { l: 56, r: 16, t: 12, b: 44 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Public Sans, system-ui, sans-serif", size: 12, color: fg },
    xaxis: { gridcolor: grid, zerolinecolor: grid, automargin: true },
    yaxis: { gridcolor: grid, zerolinecolor: grid, automargin: true },
    legend: { orientation: "h", y: -0.2 },
    hoverlabel: { font: { family: "Public Sans, system-ui, sans-serif" } },
  };
  const merged: Partial<Layout> = {
    ...base,
    ...layout,
    xaxis: { ...base.xaxis, ...(layout?.xaxis ?? {}) },
    yaxis: { ...base.yaxis, ...(layout?.yaxis ?? {}) },
  };
  return (
    <div role="img" aria-label={ariaLabel} className="w-full">
      <PlotlyComponent
        data={data}
        layout={merged}
        config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
