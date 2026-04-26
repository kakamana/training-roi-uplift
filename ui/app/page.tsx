"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type Uplift = {
  cate: number;
  ci_lower: number;
  ci_upper: number;
  ci_level: number;
  n_boot: number;
  decision_aid_disclaimer: string;
};

const TRAININGS = [
  "T-SKILL-001",
  "T-SKILL-002",
  "T-LEAD-001",
  "T-LEAD-002",
  "T-MIXED-001",
];

const DEPTS = ["Sales", "R&D", "Ops", "HR", "Finance"];

function histogram(values: number[], bins = 24) {
  if (!values.length) return { edges: [] as number[], counts: [] as number[] };
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const w = (hi - lo) / bins || 1;
  const counts = Array(bins).fill(0);
  for (const v of values) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / w)));
    counts[i] += 1;
  }
  const edges = Array(bins + 1).fill(0).map((_, i) => lo + i * w);
  return { edges, counts };
}

export default function Home() {
  const [pop, setPop] = useState<number[] | null>(null);
  const [pred, setPred] = useState<Uplift | null>(null);
  const [loading, setLoading] = useState(false);

  // Single-employee form
  const [form, setForm] = useState({
    pre_perf: 2.6,
    tenure_yrs: 4.5,
    role_level: 3,
    dept: "R&D",
    training_id: "T-SKILL-001",
  });

  async function loadPop() {
    const res = await fetch(`${API}/population_cate`);
    setPop(await res.json());
  }

  useEffect(() => {
    loadPop();
  }, []);

  async function runUplift() {
    setLoading(true);
    try {
      const res = await fetch(`${API}/uplift`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          employee_features: {
            pre_perf: Number(form.pre_perf),
            tenure_yrs: Number(form.tenure_yrs),
            role_level: Number(form.role_level),
            dept: form.dept,
          },
          training_id: form.training_id,
          n_boot: 50,
        }),
      });
      setPred(await res.json());
    } finally {
      setLoading(false);
    }
  }

  const hist = pop ? histogram(pop) : null;
  const maxCount = hist ? Math.max(...hist.counts, 1) : 1;

  return (
    <main className="min-h-screen p-8 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold">Training ROI Predictor (Causal Uplift)</h1>
      <p className="opacity-70 mb-6">
        Per-employee CATE from an X-learner with bootstrap CI. Population view + single-employee explainer.
      </p>

      <section className="mt-2">
        <h2 className="font-semibold">Population CATE distribution</h2>
        {hist && (
          <div className="mt-2 flex items-end gap-[2px] h-32 border rounded-2xl p-2">
            {hist.counts.map((c, i) => (
              <div
                key={i}
                title={`bin ${i}: ${c}`}
                className="bg-black w-full"
                style={{ height: `${(c / maxCount) * 100}%` }}
              />
            ))}
          </div>
        )}
      </section>

      <section className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-2xl border p-4">
          <h2 className="font-semibold mb-3">Single-employee CATE</h2>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <Field label="pre_perf (1-5)" value={form.pre_perf}
                   onChange={(v) => setForm({ ...form, pre_perf: Number(v) })}/>
            <Field label="tenure_yrs" value={form.tenure_yrs}
                   onChange={(v) => setForm({ ...form, tenure_yrs: Number(v) })}/>
            <Field label="role_level (1-5)" value={form.role_level}
                   onChange={(v) => setForm({ ...form, role_level: Number(v) })}/>
            <label className="flex flex-col">
              <span className="text-xs uppercase opacity-60">dept</span>
              <select value={form.dept}
                      onChange={(e) => setForm({ ...form, dept: e.target.value })}
                      className="rounded-xl border p-2 mt-1">
                {DEPTS.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </label>
            <label className="flex flex-col col-span-2">
              <span className="text-xs uppercase opacity-60">training_id</span>
              <select value={form.training_id}
                      onChange={(e) => setForm({ ...form, training_id: e.target.value })}
                      className="rounded-xl border p-2 mt-1">
                {TRAININGS.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </label>
          </div>
          <button
            onClick={runUplift}
            disabled={loading}
            className="mt-4 rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
          >
            {loading ? "Estimating..." : "Estimate uplift"}
          </button>
        </div>

        <div className="rounded-2xl border p-4">
          <h2 className="font-semibold mb-3">Result</h2>
          {pred ? (
            <>
              <div className="text-3xl font-bold">
                CATE = {pred.cate.toFixed(3)}
              </div>
              <div className="text-sm opacity-70 mt-1">
                90% CI: [{pred.ci_lower.toFixed(3)}, {pred.ci_upper.toFixed(3)}]
              </div>
              <div className="text-xs opacity-60 mt-2">
                Bootstrap n = {pred.n_boot}
              </div>
              <p className="mt-4 text-xs italic opacity-60">{pred.decision_aid_disclaimer}</p>
            </>
          ) : (
            <p className="text-sm opacity-60">Submit the form to see CATE + CI.</p>
          )}
        </div>
      </section>
    </main>
  );
}

function Field({ label, value, onChange }: { label: string; value: number | string; onChange: (v: string) => void }) {
  return (
    <label className="flex flex-col">
      <span className="text-xs uppercase opacity-60">{label}</span>
      <input
        type="number"
        step="0.1"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-xl border p-2 mt-1"
      />
    </label>
  );
}
