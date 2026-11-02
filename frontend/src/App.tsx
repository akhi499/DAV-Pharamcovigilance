import { Suspense, lazy, useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { Activity, BarChart3, BookOpen, Brain, ClipboardCheck, FlaskConical, Menu, MessageSquareText, Moon, Pill, Sun, Tags, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

const Overview = lazy(() => import("./pages/Overview"));
const Quality = lazy(() => import("./pages/Quality"));
const Demographics = lazy(() => import("./pages/Demographics"));
const Drugs = lazy(() => import("./pages/Drugs"));
const Signals = lazy(() => import("./pages/Signals"));
const Categories = lazy(() => import("./pages/Categories"));
const ModelPage = lazy(() => import("./pages/Model"));
const Ask = lazy(() => import("./pages/Ask"));
const Methods = lazy(() => import("./pages/Methods"));

export const NAV = [
  { to: "/", label: "Overview", icon: Activity, group: "Data" },
  { to: "/quality", label: "Data quality", icon: ClipboardCheck, group: "Data" },
  { to: "/demographics", label: "Demographics", icon: Users, group: "Data" },
  { to: "/drugs", label: "Drug explorer", icon: Pill, group: "Analysis" },
  { to: "/signals", label: "Signal detection", icon: BarChart3, group: "Analysis" },
  { to: "/ask", label: "Ask the data", icon: MessageSquareText, group: "Language model", ai: true },
  { to: "/categories", label: "Reaction categories", icon: Tags, group: "Language model", ai: true },
  { to: "/model", label: "Model & evaluation", icon: Brain, group: "Language model", ai: true },
  { to: "/methods", label: "Methods & limits", icon: BookOpen, group: "About" },
] as const;

function useTheme() {
  const [dark, setDark] = useState(() => {
    try {
      const s = localStorage.getItem("pv-theme");
      if (s) return s === "dark";
    } catch {
      /* storage unavailable */
    }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  });
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("pv-theme", dark ? "dark" : "light");
    } catch {
      /* ignore */
    }
  }, [dark]);
  return [dark, setDark] as const;
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const groups = [...new Set(NAV.map((n) => n.group))];
  return (
    <nav aria-label="Main" className="flex flex-col gap-5">
      {groups.map((g) => (
        <div key={g}>
          <div className={cn("mb-1.5 px-3 text-xs font-medium text-muted-foreground", g === "Language model" && "text-provisional")}>{g}</div>
          <ul className="flex flex-col gap-0.5">
            {NAV.filter((n) => n.group === g).map(({ to, label, icon: Icon, ...rest }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={to === "/"}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent",
                      isActive && ("ai" in rest ? "bg-provisional-soft font-medium text-provisional" : "bg-accent font-medium text-accent-foreground"),
                    )
                  }
                >
                  <Icon className="h-4 w-4 shrink-0" aria-hidden />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

export default function App() {
  const [dark, setDark] = useTheme();
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-screen md:grid md:grid-cols-[240px_1fr]">
      <aside className="hidden border-r bg-card md:block">
        <div className="sticky top-0 flex h-screen flex-col p-4">
          <Brand />
          <div className="mt-6 flex-1 overflow-y-auto">
            <Sidebar />
          </div>
          <ThemeToggle dark={dark} setDark={setDark} />
        </div>
      </aside>
      <div className="flex items-center justify-between border-b bg-card px-4 py-3 md:hidden">
        <Brand />
        <div className="flex gap-1">
          <ThemeToggle dark={dark} setDark={setDark} compact />
          <Button variant="ghost" size="icon" aria-label="Open menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            <Menu className="h-5 w-5" />
          </Button>
        </div>
      </div>
      {open && (
        <div className="border-b bg-card p-4 md:hidden">
          <Sidebar onNavigate={() => setOpen(false)} />
        </div>
      )}
      <main className="min-w-0 px-4 py-6 md:px-10 md:py-8">
        <div className="mx-auto max-w-7xl">
          <Suspense fallback={<Skeleton className="h-96 w-full" />}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/quality" element={<Quality />} />
              <Route path="/demographics" element={<Demographics />} />
              <Route path="/drugs" element={<Drugs />} />
              <Route path="/signals" element={<Signals />} />
              <Route path="/ask" element={<Ask />} />
              <Route path="/categories" element={<Categories />} />
              <Route path="/model" element={<ModelPage />} />
              <Route path="/methods" element={<Methods />} />
              <Route path="*" element={<p>This page does not exist. Use the menu to pick a section.</p>} />
            </Routes>
          </Suspense>
        </div>
      </main>
    </div>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid h-8 w-8 place-items-center rounded-md bg-primary text-primary-foreground">
        <FlaskConical className="h-4 w-4" aria-hidden />
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold">Drug Safety Workbench</div>
        <div className="text-xs text-muted-foreground">FDA FAERS 2025 Q1</div>
      </div>
    </div>
  );
}

function ThemeToggle({ dark, setDark, compact }: { dark: boolean; setDark: (v: boolean) => void; compact?: boolean }) {
  return (
    <Button variant="ghost" size={compact ? "icon" : "sm"} onClick={() => setDark(!dark)} aria-label={dark ? "Use light theme" : "Use dark theme"} className={compact ? "" : "justify-start"}>
      {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      {!compact && (dark ? "Light theme" : "Dark theme")}
    </Button>
  );
}
