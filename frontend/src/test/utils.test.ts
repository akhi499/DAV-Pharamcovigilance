import { describe, expect, it } from "vitest";
import { fmtInt, fmtNum, fmtPct } from "@/lib/utils";

describe("formatters", () => {
  it("formats integers with separators and handles nulls", () => {
    expect(fmtInt(400513)).toBe("400,513");
    expect(fmtInt(null)).toBe("–");
  });
  it("formats percentages", () => {
    expect(fmtPct(0.5611)).toBe("56.1%");
    expect(fmtPct(undefined)).toBe("–");
  });
  it("switches to exponent for extreme values and guards non-finite", () => {
    expect(fmtNum(1683083.57)).toBe("1.7e+6");
    expect(fmtNum(2.345)).toBe("2.35");
    expect(fmtNum(Number.POSITIVE_INFINITY)).toBe("–");
  });
});
