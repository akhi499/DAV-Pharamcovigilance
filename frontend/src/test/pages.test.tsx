import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "@/App";
import { ReviewDialog } from "@/components/ReviewDialog";
import Ask from "@/pages/Ask";
import Categories from "@/pages/Categories";
import Signals from "@/pages/Signals";
import { mockFetch, prediction } from "./fixtures";
import { renderWithProviders } from "./render";

describe("App shell", () => {
  it("renders navigation with the language-model section", async () => {
    mockFetch();
    renderWithProviders(<App />, "/methods");
    const nav = screen.getAllByRole("navigation", { name: "Main" })[0];
    expect(within(nav).getByRole("link", { name: /Signal detection/ })).toBeInTheDocument();
    expect(within(nav).getByRole("link", { name: /Reaction categories/ })).toBeInTheDocument();
    expect(within(nav).getByRole("link", { name: /Ask the data/ })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Methods and limitations" })).toBeInTheDocument();
  });
  it("shows a clear error when the backend is unreachable", async () => {
    mockFetch({});
    renderWithProviders(<App />, "/quality");
    expect(await screen.findByText(/Could not load this section/)).toBeInTheDocument();
  });
});

describe("Signals page", () => {
  it("shows the causality caveat, the signal row and the indication-bias marker", async () => {
    mockFetch();
    renderWithProviders(<Signals />, "/signals?drug=DRUGX");
    expect(await screen.findByText("DRUGX")).toBeInTheDocument();
    expect(screen.getByText(/not proof of harm/)).toBeInTheDocument();
    expect(screen.getByText("Indication?")).toBeInTheDocument();
    expect(screen.getByLabelText("Drug")).toHaveValue("DRUGX");
  });
  it("sends filters to the API", async () => {
    const fetchMock = mockFetch();
    renderWithProviders(<Signals />, "/signals");
    await screen.findByText("DRUGX");
    await userEvent.type(screen.getByLabelText("Reaction contains"), "hepat");
    await userEvent.click(screen.getByRole("button", { name: "Apply filters" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u).includes("pt=hepat"))).toBe(true));
  });
});

describe("Reaction categories page", () => {
  it("marks model output as provisional and lists predictions", async () => {
    mockFetch();
    renderWithProviders(<Categories />, "/categories");
    expect(await screen.findByText(/provisional model output/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "Term predictions" }));
    expect(await screen.findByText("closest anchor: hepatitis")).toBeInTheDocument();
    expect(screen.getAllByText("Provisional").length).toBeGreaterThan(0);
    expect(within(screen.getByRole("table")).getByText("Unreviewed")).toBeInTheDocument();
  });
});

describe("ReviewDialog", () => {
  it("requires a reviewer and posts the decision", async () => {
    const fetchMock = mockFetch();
    renderWithProviders(<ReviewDialog row={prediction as never} onClose={() => {}} />);
    const save = await screen.findByRole("button", { name: "Save review" });
    expect(save).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Your name or initials"), "AM");
    await userEvent.click(screen.getByLabelText(/Correct/));
    expect(save).toBeDisabled(); // corrected requires a category
    await userEvent.selectOptions(screen.getByLabelText("Correct category"), "skin");
    await userEvent.click(save);
    await waitFor(() => {
      const post = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
      expect(post).toBeTruthy();
      expect(JSON.parse(String(post![1]!.body))).toMatchObject({ pt_key: "hepatitis", decision: "corrected", corrected_category: "skin", reviewer: "AM" });
    });
  });
});

describe("Ask the data page", () => {
  it("posts the question and shows a cited, provisional answer with its evidence", async () => {
    const fetchMock = mockFetch();
    renderWithProviders(<Ask />, "/ask");
    await userEvent.type(screen.getByLabelText("Your question"), "Is DRUGX associated with hepatitis?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByText(/reported together in 12 cases/)).toBeInTheDocument();
    const post = fetchMock.mock.calls.find(([u, init]) => String(u).includes("/api/rag/ask") && init?.method === "POST");
    expect(JSON.parse(String(post![1]!.body))).toEqual({ question: "Is DRUGX associated with hepatitis?" });
    expect(screen.getAllByRole("link", { name: "S1" }).length).toBeGreaterThan(0);          // citation chip
    expect(screen.getByText(/does not show that the drug causes/)).toBeInTheDocument();       // causality caveat
    expect(screen.getAllByText("Provisional").length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: /Open DRUGX and Hepatitis in signal detection/ })).toHaveAttribute(
      "href", expect.stringContaining("/signals?drug=DRUGX&pt=hepatitis"));
    expect(screen.getByText("Indication? 30%")).toBeInTheDocument();
    expect(screen.getByText(/related: Hepatitis acute/)).toBeInTheDocument();
  });
  it("separates FAERS observations from established clinical evidence with source details", async () => {
    mockFetch();
    renderWithProviders(<Ask />, "/ask?q=Is%20DRUGX%20associated%20with%20hepatitis%3F");
    expect(await screen.findByRole("heading", { name: "Observed in FAERS reports" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Established clinical evidence" })).toBeInTheDocument();
    expect(screen.getByText(/AI-written, passed source checks/)).toBeInTheDocument();
    expect(screen.getByText(/Hepatitis in DRUGX label/)).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "L1" }).length).toBeGreaterThan(1);           // chips in deterministic + generated text
    expect(screen.getByRole("link", { name: /Open original/ })).toHaveAttribute("href", expect.stringContaining("dailymed.nlm.nih.gov"));
    expect(screen.getByText("2026-01-30 · version 14")).toBeInTheDocument();
  });
  it("lists indexed sources in the Sources tab", async () => {
    mockFetch();
    renderWithProviders(<Ask />, "/ask");
    await userEvent.click(screen.getByRole("tab", { name: "Sources" }));
    expect(await screen.findByText("BAAI/bge-small-en-v1.5 @ 5c38ec7")).toBeInTheDocument();
    expect(await screen.findByText("Drugxa (Drugx) - FDA prescribing information")).toBeInTheDocument();
  });
  it("disables the generator option when no local model is installed", async () => {
    mockFetch();
    renderWithProviders(<Ask />, "/ask");
    const box = await screen.findByRole("checkbox");
    await waitFor(() => expect(box).toBeDisabled());
    expect(screen.getByText(/not installed/)).toBeInTheDocument();
  });
  it("shows the evaluation against independent references", async () => {
    mockFetch();
    renderWithProviders(<Ask />, "/ask");
    await userEvent.click(screen.getByRole("tab", { name: "How well it works" }));
    expect(await screen.findByText(/Linking words to MedDRA reaction terms/)).toBeInTheDocument();
    expect(screen.getByText("SapBERT (used by the app)")).toBeInTheDocument();
    expect(screen.getByText(/No generative model weights were available/)).toBeInTheDocument();
  });
  it("reports a backend error clearly", async () => {
    mockFetch({ "POST /api/rag/ask": undefined });
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Knowledge base not built" }), { status: 503 })));
    renderWithProviders(<Ask />, "/ask?q=hello%20world");
    expect(await screen.findByText("Knowledge base not built")).toBeInTheDocument();
  });
});
