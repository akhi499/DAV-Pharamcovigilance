import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ProvisionalBadge } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input, Label, NativeSelect } from "@/components/ui/input";
import { apiPost, type Prediction } from "@/lib/api";
import { categoryLabel, useCategories } from "@/lib/categories";
import { fmtPct } from "@/lib/utils";

const DECISIONS = [
  { v: "accepted", label: "Accept", help: "The category is right for this term." },
  { v: "corrected", label: "Correct", help: "Pick the right category." },
  { v: "rejected", label: "Reject", help: "Wrong, and no single category fits." },
  { v: "uncertain", label: "Unsure", help: "Needs a clinical expert." },
] as const;

function savedReviewer(): string {
  try {
    return localStorage.getItem("pv-reviewer") ?? "";
  } catch {
    return "";
  }
}

export function ReviewDialog({ row, onClose }: { row: Prediction | null; onClose: () => void }) {
  const cats = useCategories();
  const qc = useQueryClient();
  const [decision, setDecision] = useState<string>("accepted");
  const [corrected, setCorrected] = useState("");
  const [reviewer, setReviewer] = useState(savedReviewer);
  const [note, setNote] = useState("");
  const m = useMutation({
    mutationFn: () => apiPost("/nlp/review", {
      pt_key: row!.pt_key, decision, reviewer: reviewer.trim(), note,
      corrected_category: decision === "corrected" ? corrected : null,
    }),
    onSuccess: () => {
      try {
        localStorage.setItem("pv-reviewer", reviewer.trim());
      } catch {
        /* ignore */
      }
      qc.invalidateQueries({ queryKey: ["nlp-predictions"] });
      qc.invalidateQueries({ queryKey: ["nlp-reviews"] });
      setNote("");
      onClose();
    },
  });
  if (!row) return null;
  const invalid = !reviewer.trim() || (decision === "corrected" && !corrected);
  return (
    <Dialog open={!!row} onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogTitle>Review “{row.pt}”</DialogTitle>
        <DialogDescription>
          Your decision is logged with your name and time. Reviews mark a prediction as checked; they are not used to compute model accuracy,
          because seeing the model's answer first biases the reviewer.
        </DialogDescription>
        <div className="provisional-frame flex flex-wrap items-center gap-2 px-3 py-2 text-sm">
          <ProvisionalBadge />
          <span>Model says <strong>{categoryLabel(row.prediction, cats.data)}</strong></span>
          {row.prediction === "abstain" && <span className="text-muted-foreground">(best guess {categoryLabel(row.top_category, cats.data)})</span>}
          <span className="text-muted-foreground">relative confidence {fmtPct(row.confidence, 0)}</span>
        </div>
        <form
          className="flex flex-col gap-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (!invalid) m.mutate();
          }}
        >
          <fieldset className="grid grid-cols-2 gap-2">
            <legend className="mb-1 text-sm font-medium">Decision</legend>
            {DECISIONS.map((d) => (
              <label key={d.v} className="flex cursor-pointer gap-2 rounded-md border p-2 text-sm has-[:checked]:border-primary has-[:checked]:bg-accent">
                <input type="radio" name="decision" value={d.v} checked={decision === d.v} onChange={() => setDecision(d.v)} className="mt-0.5 accent-[hsl(var(--primary))]" />
                <span><span className="font-medium">{d.label}</span><span className="block text-xs text-muted-foreground">{d.help}</span></span>
              </label>
            ))}
          </fieldset>
          {decision === "corrected" && (
            <div className="flex flex-col gap-1">
              <Label htmlFor="rv-cat">Correct category</Label>
              <NativeSelect id="rv-cat" value={corrected} onChange={(e) => setCorrected(e.target.value)} required>
                <option value="">Choose…</option>
                {cats.data?.categories.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
              </NativeSelect>
            </div>
          )}
          <div className="flex flex-col gap-1">
            <Label htmlFor="rv-name">Your name or initials</Label>
            <Input id="rv-name" value={reviewer} maxLength={64} onChange={(e) => setReviewer(e.target.value)} required />
          </div>
          <div className="flex flex-col gap-1">
            <Label htmlFor="rv-note">Note (optional)</Label>
            <Input id="rv-note" value={note} maxLength={500} onChange={(e) => setNote(e.target.value)} />
          </div>
          {m.isError && <p className="text-sm text-destructive">{(m.error as Error).message}</p>}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={invalid || m.isPending}>{m.isPending ? "Saving…" : "Save review"}</Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
