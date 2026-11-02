import { describe, expect, it, vi } from "vitest";
import { ApiError, apiGet } from "@/lib/api";

describe("apiGet", () => {
  it("builds query strings and drops empty params", async () => {
    const fn = vi.fn(async () => new Response("[]", { status: 200 }));
    vi.stubGlobal("fetch", fn);
    await apiGet("/signals", { drug: "X", pt: "", limit: 5, category: undefined });
    expect(fn).toHaveBeenCalledWith("/api/signals?drug=X&limit=5");
  });
  it("surfaces the server's validation message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "sort must be one of a" }), { status: 422 })));
    await expect(apiGet("/signals")).rejects.toMatchObject({ status: 422, message: "sort must be one of a" });
  });
  it("explains when the backend is down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("network"); }));
    const err = (await apiGet("/overview").catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toMatch(/not reachable/);
  });
});
