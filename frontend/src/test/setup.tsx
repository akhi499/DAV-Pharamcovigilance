import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Plotly needs a real canvas/WebGL; charts are replaced by a labelled placeholder in tests.
vi.mock("@/components/Plot", () => ({
  Plot: ({ ariaLabel }: { ariaLabel: string }) => <div role="img" aria-label={ariaLabel} />,
  palette: new Proxy({}, { get: () => "#000" }),
}));

window.matchMedia = window.matchMedia || ((q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, onchange: null, dispatchEvent: () => false }) as unknown as MediaQueryList);
globalThis.ResizeObserver = globalThis.ResizeObserver || class { observe() {} unobserve() {} disconnect() {} };

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
