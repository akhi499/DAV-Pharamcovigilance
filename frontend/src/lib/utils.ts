import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const nf = new Intl.NumberFormat("en-US");
export const fmtInt = (n: number | null | undefined) => (n == null ? "–" : nf.format(Math.round(n)));
export const fmtPct = (n: number | null | undefined, d = 1) => (n == null ? "–" : `${(n * 100).toFixed(d)}%`);
export function fmtNum(n: number | null | undefined, d = 2): string {
  if (n == null || !Number.isFinite(n)) return "–";
  const a = Math.abs(n);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return n.toExponential(1);
  return n.toFixed(d);
}
export const titleCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
