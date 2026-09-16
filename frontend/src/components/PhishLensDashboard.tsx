import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleUserRound,
  Clock3,
  Database,
  ExternalLink,
  FileWarning,
  FlaskConical,
  Info,
  Menu,
  Microscope,
  Moon,
  Network,
  Radar,
  Search,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sun,
  TableProperties,
  X,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { analyzeUrl, featureRanking, modelResults, validateUrl, type ScreeningResult } from "@/lib/phishlens";

type View = "analyzer" | "performance" | "robustness" | "features" | "about";

const views: { id: View; label: string; icon: LucideIcon }[] = [
  { id: "analyzer", label: "URL Analyzer", icon: Search },
  { id: "performance", label: "Model Performance", icon: BarChart3 },
  { id: "robustness", label: "Robustness Validation", icon: Activity },
  { id: "features", label: "Feature Analysis", icon: BrainCircuit },
  { id: "about", label: "About Project", icon: BookOpen },
];

const examples = [
  "https://www.example.com",
  "http://192.168.1.10/login?verify=123456789",
  "https://secure-login.example.com/account/verify",
];

function Panel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`border border-border bg-card shadow-panel ${className}`}>{children}</section>;
}

function SectionHeading({ eyebrow, title, copy }: { eyebrow: string; title: string; copy?: string }) {
  return (
    <header className="mb-7 max-w-3xl animate-rise">
      <p className="mb-3 font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-primary">{eyebrow}</p>
      <h1 className="font-display text-3xl font-semibold leading-tight text-foreground md:text-4xl">{title}</h1>
      {copy && <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground md:text-base">{copy}</p>}
    </header>
  );
}

function Metric({ label, value, note, tone = "default" }: { label: string; value: string; note?: string; tone?: "default" | "safe" | "warn" | "danger" }) {
  return (
    <div className={`metric metric-${tone}`}>
      <p className="text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">{label}</p>
      <p className="mt-2 font-display text-2xl font-semibold text-foreground">{value}</p>
      {note && <p className="mt-1 text-xs text-muted-foreground">{note}</p>}
    </div>
  );
}

function MiniBars({ data, valueKey, suffix = "%", max = 100 }: { data: typeof modelResults; valueKey: "accuracy" | "f1" | "auc" | "time"; suffix?: string; max?: number }) {
  return (
    <div className="space-y-3">
      {data.map((item) => {
        const value = item[valueKey];
        const isBest = value === Math.max(...data.map((row) => row[valueKey]));
        return (
          <div key={item.model} className="grid grid-cols-[7.5rem_1fr_4.5rem] items-center gap-3 text-xs">
            <span className={item.model === "Proposed ANN" ? "font-semibold text-primary" : "truncate text-muted-foreground"}>{item.model}</span>
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div className={`h-full rounded-full ${isBest ? "bg-safe" : item.model === "Proposed ANN" ? "bg-primary" : "bg-chart-muted"}`} style={{ width: `${Math.max(3, value / max * 100)}%` }} />
            </div>
            <span className={`text-right font-mono ${isBest ? "text-safe" : "text-foreground"}`}>{value.toFixed(valueKey === "time" ? 1 : 3)}{suffix}</span>
          </div>
        );
      })}
    </div>
  );
}

function Sidebar({ view, setView, open, setOpen, apiOnline }: { view: View; setView: (view: View) => void; open: boolean; setOpen: (open: boolean) => void; apiOnline: boolean }) {
  return (
    <>
      {open && <button aria-label="Close navigation" className="fixed inset-0 z-40 bg-overlay lg:hidden" onClick={() => setOpen(false)} />}
      <aside className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-border bg-sidebar transition-transform duration-300 lg:translate-x-0 ${open ? "translate-x-0" : "-translate-x-full"}`}>
        <div className="flex h-20 items-center justify-between border-b border-border px-5">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center border border-primary/40 bg-primary/10 text-primary"><Shield size={19} /></div>
            <div><p className="font-display text-lg font-semibold text-foreground">PhishLens</p><p className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground">Research Console</p></div>
          </div>
          <Button className="lg:hidden" variant="ghost" size="icon" onClick={() => setOpen(false)} aria-label="Close navigation"><X /></Button>
        </div>
        <nav className="flex-1 space-y-1 px-3 py-5" aria-label="Primary navigation">
          <p className="px-3 pb-2 font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground">Workspace</p>
          {views.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} onClick={() => { setView(item.id); setOpen(false); }} className={`nav-item ${view === item.id ? "nav-item-active" : ""}`}><Icon size={17} /><span>{item.label}</span>{view === item.id && <span className="ml-auto size-1.5 rounded-full bg-primary" />}</button>;
          })}
        </nav>
        <div className="border-t border-border p-4">
          <div className="border border-border bg-secondary/50 p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-foreground"><span className={`size-2 rounded-full ${apiOnline ? "bg-safe shadow-status-safe" : "bg-warning shadow-status-warning"}`} />{apiOnline ? "Inference API online" : "Inference API unavailable"}</div>
            <p className="mt-1 pl-4 text-[11px] text-muted-foreground">{apiOnline ? "Live URL-structure screening" : "Start the backend to analyze URLs"}</p>
          </div>
          <p className="mt-4 text-[10px] leading-4 text-muted-foreground">For research and educational use. Do not open suspicious URLs.</p>
        </div>
      </aside>
    </>
  );
}

function Topbar({ onMenu }: { onMenu: () => void }) {
  const [dark, setDark] = useState(true);
  useEffect(() => document.documentElement.classList.toggle("dark", dark), [dark]);
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/90 px-4 backdrop-blur-md md:px-7 lg:ml-64">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" className="lg:hidden" onClick={onMenu} aria-label="Open navigation"><Menu /></Button>
        <div className="hidden items-center gap-2 text-xs text-muted-foreground sm:flex"><span className="size-2 rounded-full bg-safe shadow-status-safe" /><span>Interface operational</span></div>
      </div>
      <div className="flex items-center gap-2">
        <div className="hidden items-center gap-2 border border-border bg-secondary px-3 py-2 text-[11px] font-medium text-secondary-foreground sm:flex"><FlaskConical size={14} className="text-primary" />Research Mode</div>
        <Button variant="outline" size="icon" onClick={() => setDark((value) => !value)} aria-label="Toggle color theme">{dark ? <Sun /> : <Moon />}</Button>
        <div className="grid size-9 place-items-center border border-border bg-card text-muted-foreground" aria-label="PhishLens project"><CircleUserRound size={18} /></div>
      </div>
    </header>
  );
}

function UrlAnalyzer() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<ScreeningResult | null>(null);
  const [error, setError] = useState("");
  const [loadingStep, setLoadingStep] = useState(-1);
  const [featuresOpen, setFeaturesOpen] = useState(false);

  async function submit() {
    const checked = validateUrl(input);
    if (!checked.valid) { setError(checked.message ?? "Invalid URL"); setResult(null); return; }
    setError(""); setResult(null); setLoadingStep(0);
    const progressTimers = [window.setTimeout(() => setLoadingStep(1), 350), window.setTimeout(() => setLoadingStep(2), 700)];
    try {
      const response = await analyzeUrl(input);
      setResult(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to reach the analysis service.");
    } finally {
      progressTimers.forEach(window.clearTimeout);
      setLoadingStep(-1);
    }
  }

  return (
    <>
      <SectionHeading eyebrow="Phishing Detection Lab" title="Analyze a website before you trust it." copy="Inspect URL structure, estimate risk, and understand the signals behind the result." />
      <Panel className="animate-rise-delay p-4 md:p-5">
        <label htmlFor="url-input" className="mb-2 block text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">Website URL</label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <div className="relative flex-1"><Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" size={18} /><input id="url-input" value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => event.key === "Enter" && submit()} placeholder="https://example.com" maxLength={2048} spellCheck={false} className="h-12 w-full border border-input bg-input-background pl-10 pr-4 font-mono text-sm text-foreground outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20" /></div>
          <Button size="lg" onClick={submit} disabled={loadingStep >= 0}><ShieldCheck />Analyze URL</Button>
          <Button size="lg" variant="outline" onClick={() => { setInput(""); setResult(null); setError(""); }}><X />Clear</Button>
        </div>
        {error && <div role="alert" className="mt-3 flex items-center gap-2 border-l-2 border-danger bg-danger/10 px-3 py-2 text-sm text-danger"><XCircle size={16} />{error}</div>}
        <div className="mt-4 flex flex-wrap items-center gap-2"><span className="mr-1 text-[11px] text-muted-foreground">Try an example</span>{examples.map((url) => <button key={url} className="example-chip" onClick={() => { setInput(url); setResult(null); setError(""); }}>{url}</button>)}</div>
      </Panel>

      {loadingStep >= 0 && <Panel className="mt-5 grid min-h-72 place-items-center p-8"><div className="w-full max-w-sm"><div className="mx-auto mb-6 grid size-16 place-items-center border border-primary/30 bg-primary/10 text-primary"><Radar className="animate-spin-slow" size={30} /></div>{["Extracting URL signals", "Calculating risk estimate", "Preparing explanation"].map((text, index) => <div key={text} className={`mb-3 flex items-center gap-3 text-sm transition ${index <= loadingStep ? "text-foreground" : "text-muted-foreground/50"}`}><span className={`grid size-5 place-items-center rounded-full border ${index < loadingStep ? "border-safe bg-safe text-safe-foreground" : index === loadingStep ? "border-primary text-primary" : "border-border"}`}>{index < loadingStep ? <Check size={12} /> : index + 1}</span>{text}</div>)}</div></Panel>}

      {!result && loadingStep < 0 && <Panel className="mt-5 grid min-h-72 place-items-center p-8 text-center"><div><div className="relative mx-auto mb-5 grid size-20 place-items-center border border-border bg-secondary text-primary"><Shield size={35} /><Search className="absolute bottom-4 right-4 rounded-full bg-card p-1 text-muted-foreground" size={21} /></div><h2 className="font-display text-lg font-semibold">Enter a URL to begin screening.</h2><p className="mt-2 text-sm text-muted-foreground">Results are generated by the local backend's URL-structure screening pipeline.</p><p className="mt-5 inline-flex items-center gap-2 text-xs text-muted-foreground"><ShieldCheck size={14} />URLs are treated as text and are never opened or fetched.</p></div></Panel>}

      {result && <ResultPanel result={result} onReset={() => { setResult(null); setInput(""); }} featuresOpen={featuresOpen} setFeaturesOpen={setFeaturesOpen} />}
    </>
  );
}

function ResultPanel({ result, onReset, featuresOpen, setFeaturesOpen }: { result: ScreeningResult; onReset: () => void; featuresOpen: boolean; setFeaturesOpen: (open: boolean) => void }) {
  const dangerous = result.classification === "LIKELY PHISHING";
  const statusIcon = dangerous ? ShieldAlert : ShieldCheck;
  const StatusIcon = statusIcon;
  return <div className="mt-5 space-y-5 animate-result">
    <Panel className={`overflow-hidden border-t-2 ${dangerous ? "border-t-danger" : "border-t-safe"}`}>
      <div className="flex flex-col justify-between gap-4 border-b border-border p-5 md:flex-row md:items-start">
        <div><div className="mb-3 flex flex-wrap items-center gap-2"><span className="badge-neutral">Live API result</span><span className={`status-badge ${dangerous ? "status-danger" : "status-safe"}`}><StatusIcon size={14} />{result.classification}</span><span className={`status-badge ${result.risk === "High" ? "status-danger" : result.risk === "Medium" ? "status-warning" : "status-safe"}`}>{result.risk} risk</span></div><p className="max-w-2xl break-all font-mono text-xs text-muted-foreground">{result.normalizedUrl}</p><p className="mt-2 flex items-center gap-1.5 text-[11px] text-muted-foreground"><Clock3 size={12} />{new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(result.timestamp))}</p></div>
        <Button variant="outline" onClick={onReset}><Search />Analyze another URL</Button>
      </div>
      <div className="grid gap-px bg-border md:grid-cols-[1.15fr_.85fr]">
        <div className="bg-card p-6"><p className="text-xs uppercase tracking-[0.1em] text-muted-foreground">Phishing probability</p><p className={`mt-2 font-display text-5xl font-semibold ${dangerous ? "text-danger" : "text-safe"}`}>{result.phishing.toFixed(2)}%</p><div className="mt-6"><div className="mb-2 flex justify-between text-xs text-muted-foreground"><span>Lower risk</span><span>Higher risk</span></div><Progress value={result.phishing} className={dangerous ? "risk-progress" : "safe-progress"} /></div></div>
        <div className="grid grid-cols-2 gap-px bg-border"><div className="bg-card p-5"><p className="text-xs text-muted-foreground">Legitimate probability</p><p className="mt-2 font-display text-2xl font-semibold text-safe">{result.legitimate.toFixed(2)}%</p></div><div className="bg-card p-5"><p className="text-xs text-muted-foreground">Risk category</p><p className={`mt-2 font-display text-2xl font-semibold ${result.risk === "Low" ? "text-safe" : result.risk === "Medium" ? "text-warning" : "text-danger"}`}>{result.risk}</p></div><div className="col-span-2 bg-card p-5"><p className="flex items-start gap-2 text-xs leading-5 text-muted-foreground"><Info size={15} className="mt-0.5 shrink-0 text-primary" />This live screening uses URL-structure features. It does not visit the submitted website or claim an ANN probability.</p></div></div>
      </div>
    </Panel>

    <Panel className="p-5"><div className="mb-4"><h2 className="font-display text-xl font-semibold">Why did we get this result?</h2><p className="mt-1 text-sm text-muted-foreground">Signals found in the URL text are listed below.</p></div><div className="divide-y divide-border">{result.signals.map((signal) => <div key={signal.name} className="grid gap-3 py-3 sm:grid-cols-[1.3fr_2fr_auto] sm:items-center"><div className="flex items-center gap-3">{signal.status === "risk" ? <AlertTriangle className="text-danger" size={17} /> : signal.status === "safe" ? <Check className="text-safe" size={17} /> : <Info className="text-muted-foreground" size={17} />}<span className="text-sm font-medium">{signal.name}</span></div><p className="text-xs leading-5 text-muted-foreground">{signal.description}</p><span className={`signal-label signal-${signal.status}`}>{signal.status === "risk" ? "Raises risk" : signal.status === "safe" ? "Lowers risk" : "Neutral"}</span></div>)}</div><div className="mt-4 flex items-start gap-2 border-l-2 border-warning bg-warning/10 px-3 py-2 text-xs leading-5 text-warning"><AlertTriangle size={15} className="mt-0.5 shrink-0" />These explanations describe URL-level signals. They are not proof that a website is malicious or safe.</div></Panel>

    <Collapsible open={featuresOpen} onOpenChange={setFeaturesOpen}><Panel><CollapsibleTrigger asChild><Button variant="ghost" className="h-14 w-full justify-between rounded-none px-5"><span className="flex items-center gap-2 font-display text-base"><TableProperties size={17} className="text-primary" />Extracted URL signals</span><ChevronDown className={`transition ${featuresOpen ? "rotate-180" : ""}`} /></Button></CollapsibleTrigger><CollapsibleContent><div className="border-t border-border"><Table><TableHeader><TableRow><TableHead>Feature</TableHead><TableHead>Value</TableHead><TableHead>Interpretation</TableHead></TableRow></TableHeader><TableBody>{result.features.map((row) => <TableRow key={row.feature}><TableCell className="font-mono text-xs text-primary">{row.feature}</TableCell><TableCell className="font-mono">{row.value}</TableCell><TableCell className="text-muted-foreground">{row.interpretation}</TableCell></TableRow>)}</TableBody></Table></div></CollapsibleContent></Panel></Collapsible>
  </div>;
}

function ModelPerformance() {
  return <><SectionHeading eyebrow="Evaluation Benchmarks" title="Model performance" copy="A compact comparison of six classifiers on the saved PhiUSIIL evaluation split." /><div className="mb-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><Metric label="Best accuracy" value="99.985%" note="XGBoost" tone="safe" /><Metric label="Proposed ANN" value="99.975%" note="Accuracy" /><Metric label="Best F1-score" value="99.983%" note="XGBoost" tone="safe" /><Metric label="Models compared" value="06" note="Same evaluation split" /></div><div className="grid gap-5 xl:grid-cols-2"><Panel className="p-5"><ChartTitle title="Accuracy" subtitle="Test split · percentage" /><MiniBars data={modelResults} valueKey="accuracy" /></Panel><Panel className="p-5"><ChartTitle title="F1-score" subtitle="Class-balanced performance" /><MiniBars data={modelResults} valueKey="f1" /></Panel><Panel className="p-5"><ChartTitle title="ROC-AUC" subtitle="Area under ROC curve" /><MiniBars data={modelResults} valueKey="auc" /></Panel><Panel className="p-5"><ChartTitle title="Training time" subtitle="Illustrative saved-run seconds" /><MiniBars data={modelResults} valueKey="time" suffix="s" max={70} /></Panel></div><Panel className="mt-5"><div className="border-b border-border px-5 py-4"><h2 className="font-display font-semibold">Comparison table</h2></div><Table><TableHeader><TableRow><TableHead>Model</TableHead><TableHead>Accuracy</TableHead><TableHead>F1-score</TableHead><TableHead>ROC-AUC</TableHead><TableHead>Research note</TableHead></TableRow></TableHeader><TableBody>{modelResults.map((row) => <TableRow key={row.model} className={row.model === "Proposed ANN" ? "bg-primary/5" : ""}><TableCell className="font-medium">{row.model}{row.model === "Proposed ANN" && <span className="ml-2 badge-neutral">Proposed</span>}</TableCell><TableCell className={row.model === "XGBoost" ? "font-semibold text-safe" : "font-mono"}>{row.accuracy.toFixed(3)}%{row.model === "XGBoost" && <span className="ml-2 text-[10px]">BEST</span>}</TableCell><TableCell className={row.model === "XGBoost" ? "font-semibold text-safe" : "font-mono"}>{row.f1.toFixed(3)}%</TableCell><TableCell className="font-mono">{(row.auc / 100).toFixed(6)}</TableCell><TableCell className="text-xs text-muted-foreground">Saved evaluation</TableCell></TableRow>)}</TableBody></Table></Panel><div className="mt-5 flex items-start gap-3 border border-primary/30 bg-primary/5 p-4 text-sm leading-6 text-muted-foreground"><Info className="mt-0.5 shrink-0 text-primary" size={18} /><p>The results shown here are from the saved research evaluation and will later be loaded dynamically from the backend. XGBoost is slightly stronger in the saved single-split comparison; the Proposed ANN remains competitive.</p></div></>;
}

function ChartTitle({ title, subtitle }: { title: string; subtitle: string }) { return <div className="mb-5 flex items-start justify-between"><div><h2 className="font-display font-semibold">{title}</h2><p className="mt-1 text-xs text-muted-foreground">{subtitle}</p></div><BarChart3 size={17} className="text-primary" /></div>; }

function Robustness() {
  const folds = [99.968, 99.981, 99.971, 99.978, 99.972];
  return <><SectionHeading eyebrow="Generalization Study" title="Robustness validation" copy="Validation beyond one train-test split reveals both model stability and the limits of transfer." /><Panel className="p-5"><div className="flex flex-col justify-between gap-3 md:flex-row"><div><h2 className="font-display text-xl font-semibold">5-Fold Cross-Validation</h2><p className="mt-1 max-w-2xl text-sm text-muted-foreground">Each fold uses four partitions for training and one partition for testing. Every record is tested exactly once.</p></div><span className="badge-neutral self-start">5 folds</span></div><div className="my-6 grid gap-3 sm:grid-cols-3"><Metric label="Mean accuracy" value="99.974%" note="σ 0.005%" tone="safe" /><Metric label="Mean F1-score" value="99.969%" note="σ 0.006%" tone="safe" /><Metric label="Mean ROC-AUC" value="0.999997" note="σ 0.000002" tone="safe" /></div><div className="flex h-44 items-end gap-3 border-b border-border px-2 pt-5">{folds.map((fold, index) => <div key={fold} className="flex h-full flex-1 flex-col justify-end text-center"><span className="mb-2 font-mono text-[10px] text-foreground">{fold.toFixed(3)}%</span><div className="mx-auto w-full max-w-14 bg-primary transition-all" style={{ height: `${55 + (fold - 99.96) * 1200}%` }} /><span className="mt-2 text-[10px] text-muted-foreground">FOLD {index + 1}</span></div>)}</div></Panel><Panel className="mt-5 p-5"><h2 className="font-display text-xl font-semibold">Cross-Dataset Validation</h2><div className="mt-5 grid gap-px bg-border md:grid-cols-2"><div className="bg-card p-5"><p className="text-xs uppercase tracking-[0.08em] text-muted-foreground">PhiUSIIL test split</p><p className="mt-3 font-display text-3xl font-semibold text-safe">93.197%</p><p className="text-xs text-muted-foreground">Accuracy</p><div className="mt-5 grid grid-cols-2 gap-3"><Metric label="F1-score" value="92.154%" /><Metric label="ROC-AUC" value="0.981094" /></div></div><div className="bg-card p-5"><p className="text-xs uppercase tracking-[0.08em] text-muted-foreground">UCI external dataset</p><p className="mt-3 font-display text-3xl font-semibold text-danger">47.445%</p><p className="text-xs text-muted-foreground">Accuracy</p><div className="mt-5 grid grid-cols-2 gap-3"><Metric label="F1-score" value="62.603%" tone="warn" /><Metric label="ROC-AUC" value="0.583023" tone="danger" /></div></div></div><div className="mt-5 flex gap-3 border-l-2 border-warning bg-warning/10 p-4"><AlertTriangle className="mt-0.5 shrink-0 text-warning" size={19} /><div><p className="font-semibold text-warning">External generalization warning</p><p className="mt-1 text-sm leading-6 text-muted-foreground">The model performs very well on the original data distribution, but external generalization is weak because the UCI dataset uses different feature definitions and distributions.</p></div></div></Panel><Panel className="mt-5 p-5"><h2 className="font-display text-xl font-semibold">URL-Only Ablation</h2><p className="mt-1 text-sm text-muted-foreground">Accuracy retained when webpage-content features are removed.</p><div className="mt-6 space-y-5">{[{ label: "Full feature ANN", count: 30, score: 99.975 }, { label: "URL-only ANN", count: 20, score: 99.775 }].map((row) => <div key={row.label}><div className="mb-2 flex justify-between text-sm"><span className="font-medium">{row.label} <span className="ml-2 text-xs text-muted-foreground">{row.count} features</span></span><span className="font-mono">{row.score.toFixed(3)}%</span></div><Progress value={row.score} /></div>)}</div></Panel></>;
}

function FeatureAnalysis() {
  const max = featureRanking[0][1];
  return <><SectionHeading eyebrow="Signal Intelligence" title="Feature analysis" copy="Explore the variables that carry the strongest statistical relationship with the target label." /><div className="mb-5 grid gap-3 sm:grid-cols-3"><Metric label="Cleaned features" value="48" note="After preprocessing" /><Metric label="Selected features" value="30" note="Top-ranked signals" tone="safe" /><Metric label="Selection method" value="Mutual Info" note="Univariate ranking" /></div><div className="grid gap-5 xl:grid-cols-[1.3fr_.7fr]"><Panel className="p-5"><ChartTitle title="Top feature ranking" subtitle="Relative mutual-information score" /><div className="space-y-3">{featureRanking.map(([name, score], index) => <div key={name} className="grid grid-cols-[1.3rem_9rem_1fr_3rem] items-center gap-2 text-xs"><span className="font-mono text-muted-foreground">{String(index + 1).padStart(2, "0")}</span><span className="truncate text-foreground">{name}</span><div className="h-2 bg-muted"><div className="h-full bg-primary" style={{ width: `${score / max * 100}%` }} /></div><span className="text-right font-mono text-muted-foreground">{score.toFixed(3)}</span></div>)}</div></Panel><Panel className="p-5"><Microscope className="mb-4 text-primary" size={24} /><h2 className="font-display text-lg font-semibold">What mutual information means</h2><p className="mt-3 text-sm leading-6 text-muted-foreground">Mutual information estimates how strongly a feature is associated with the target label. The project selects the 30 most informative features from the cleaned 48-feature set.</p><div className="mt-5 border-t border-border pt-4"><p className="text-xs text-muted-foreground">Higher scores indicate stronger statistical association, not necessarily causation.</p></div></Panel></div><Panel className="mt-5"><div className="border-b border-border px-5 py-4"><h2 className="font-display font-semibold">Feature ranking table</h2></div><Table><TableHeader><TableRow><TableHead>Rank</TableHead><TableHead>Feature</TableHead><TableHead>MI score</TableHead><TableHead>Feature family</TableHead></TableRow></TableHeader><TableBody>{featureRanking.map(([name, score], index) => <TableRow key={name}><TableCell className="font-mono text-muted-foreground">{index + 1}</TableCell><TableCell className="font-mono text-xs text-primary">{name}</TableCell><TableCell className="font-mono">{score.toFixed(3)}</TableCell><TableCell className="text-muted-foreground">{name.includes("URL") || name === "IsHTTPS" ? "URL structure" : "Page content"}</TableCell></TableRow>)}</TableBody></Table></Panel></>;
}

function AboutProject() {
  const pipeline = ["Dataset", "Cleaning", "Leakage screening", "Stratified split", "StandardScaler", "Mutual Information selection", "ANN training", "Baseline comparison", "Robustness validation", "Result display"];
  const tech = ["Python", "Pandas", "NumPy", "Scikit-learn", "XGBoost", "TensorFlow / Keras", "React", "Tailwind CSS"];
  return <><SectionHeading eyebrow="Research Overview" title="About PhishLens" copy="A machine-learning research project for distinguishing phishing websites from legitimate websites using URL and webpage-derived signals." /><div className="grid gap-5 xl:grid-cols-[1fr_.8fr]"><Panel className="p-5"><h2 className="font-display text-xl font-semibold">Project goal</h2><p className="mt-3 text-sm leading-6 text-muted-foreground">Evaluate whether engineered website signals can support accurate phishing detection, compare an artificial neural network against established baselines, and test how well those findings transfer to a different dataset.</p><div className="mt-6 grid gap-3 sm:grid-cols-2"><div className="border border-border p-4"><Database className="text-primary" size={19} /><p className="mt-3 font-medium">PhiUSIIL</p><p className="mt-1 text-xs text-muted-foreground">Primary training and evaluation dataset</p></div><div className="border border-border p-4"><ExternalLink className="text-warning" size={19} /><p className="mt-3 font-medium">UCI Phishing Websites</p><p className="mt-1 text-xs text-muted-foreground">External generalization dataset</p></div></div><div className="mt-5 border-l-2 border-primary bg-primary/5 p-4 text-sm"><p><code className="mr-2 text-danger">0</code>= Phishing</p><p className="mt-2"><code className="mr-2 text-safe">1</code>= Legitimate</p></div></Panel><Panel className="p-5"><h2 className="font-display text-xl font-semibold">ANN architecture</h2><div className="mt-5 space-y-2 font-mono text-xs">{["Input (30)", "Dense (128, ReLU)", "Dropout (0.30)", "Dense (64, ReLU)", "Dropout (0.30)", "Dense (32, ReLU)", "Dense (1, Sigmoid)"].map((layer, index, array) => <div key={layer} className="text-center"><div className={`border px-3 py-2 ${index === 0 || index === array.length - 1 ? "border-primary/50 bg-primary/10 text-primary" : "border-border bg-secondary text-foreground"}`}>{layer}</div>{index < array.length - 1 && <div className="mx-auto h-3 w-px bg-border" />}</div>)}</div></Panel></div><Panel className="mt-5 p-5"><h2 className="font-display text-xl font-semibold">Machine-learning pipeline</h2><div className="mt-5 flex flex-wrap items-center gap-2">{pipeline.map((step, index) => <div key={step} className="flex items-center gap-2"><span className="border border-border bg-secondary px-3 py-2 text-xs text-foreground"><span className="mr-2 font-mono text-primary">{String(index + 1).padStart(2, "0")}</span>{step}</span>{index < pipeline.length - 1 && <span className="text-muted-foreground">→</span>}</div>)}</div></Panel><div className="mt-5 grid gap-5 lg:grid-cols-2"><Panel className="p-5"><h2 className="font-display text-xl font-semibold">Technology stack</h2><div className="mt-4 flex flex-wrap gap-2">{tech.map((item) => <span className="badge-neutral" key={item}>{item}</span>)}</div></Panel><Panel className="p-5"><h2 className="font-display text-xl font-semibold">Research limitations</h2><div className="mt-4 flex gap-3"><FileWarning className="mt-0.5 shrink-0 text-warning" size={19} /><p className="text-sm leading-6 text-muted-foreground">The raw URL extractor does not reproduce every original PhiUSIIL engineered feature. Current URL results are demonstration estimates, while external dataset performance shows meaningful distribution shift.</p></div></Panel></div><Panel className="mt-5 p-5"><h2 className="font-display text-xl font-semibold">Future roadmap</h2><div className="mt-5 grid gap-3 md:grid-cols-3"><RoadmapItem icon={Network} title="Connect inference API" copy="Replace deterministic demo scoring with validated server predictions." /><RoadmapItem icon={Database} title="Load research artifacts" copy="Read current metrics and figures directly from saved outputs." /><RoadmapItem icon={ShieldCheck} title="Model-compatible pipeline" copy="Reproduce the exact feature schema before claiming ANN probability." /></div></Panel></>;
}

function RoadmapItem({ icon: Icon, title, copy }: { icon: LucideIcon; title: string; copy: string }) { return <div className="border-l border-border pl-4"><Icon className="text-primary" size={18} /><p className="mt-3 text-sm font-semibold">{title}</p><p className="mt-1 text-xs leading-5 text-muted-foreground">{copy}</p></div>; }

export function PhishLensDashboard() {
  const [view, setView] = useState<View>("analyzer");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  useEffect(() => {
    fetch(`${(import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "")}/api/health`)
      .then((response) => setApiOnline(response.ok))
      .catch(() => setApiOnline(false));
  }, []);
  const currentLabel = useMemo(() => views.find((item) => item.id === view)?.label ?? "URL Analyzer", [view]);
  return <div className="min-h-screen bg-background text-foreground"><Sidebar view={view} setView={setView} open={sidebarOpen} setOpen={setSidebarOpen} apiOnline={apiOnline} /><Topbar onMenu={() => setSidebarOpen(true)} /><main className="lg:ml-64"><div className="mx-auto max-w-[1440px] px-4 py-7 md:px-7 md:py-9"><div className="mb-5 flex items-center gap-2 text-[11px] text-muted-foreground"><span>PhishLens</span><span>/</span><span className="text-foreground">{currentLabel}</span></div>{view === "analyzer" && <UrlAnalyzer />}{view === "performance" && <ModelPerformance />}{view === "robustness" && <Robustness />}{view === "features" && <FeatureAnalysis />}{view === "about" && <AboutProject />}</div></main></div>;
}
