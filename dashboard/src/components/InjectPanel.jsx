import { useEffect, useState } from "react";
import Ballot from "./Ballot.jsx";
import { getPresets, inject } from "../api.js";

// Four fields worth exposing live. The rest of the transaction comes from the
// chosen preset — the point of the panel is to fire a scenario in one click,
// not to fill in thirteen fields on stage.
const EDITABLE = [
  { key: "TransactionAmount", label: "Amount", step: "1" },
  { key: "AccountBalance", label: "Balance", step: "1" },
  { key: "LoginAttempts", label: "Login attempts", step: "1" },
  { key: "TransactionDuration", label: "Duration (s)", step: "1" },
];

export default function InjectPanel({ onScored }) {
  const [presets, setPresets] = useState([]);
  const [selected, setSelected] = useState(null);
  const [overrides, setOverrides] = useState({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getPresets()
      .then((data) => {
        if (cancelled) return;
        setPresets(data.presets);
        const first = data.presets[0];
        if (first) {
          setSelected(first.name);
          setOverrides(pickEditable(first.fields));
        }
      })
      .catch((err) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, []);

  function pickEditable(fields) {
    return Object.fromEntries(EDITABLE.map(({ key }) => [key, fields[key]]));
  }

  function choose(preset) {
    setSelected(preset.name);
    setOverrides(pickEditable(preset.fields));
    setResult(null);
    setError(null);
  }

  async function submit(event) {
    event.preventDefault();
    if (!selected || busy) return;

    setBusy(true);
    setError(null);
    try {
      // Only send fields the operator actually changed, so an untouched panel
      // fires the preset exactly as defined rather than a rounded copy of it.
      const preset = presets.find((p) => p.name === selected);
      const changed = Object.fromEntries(
        Object.entries(overrides)
          .filter(([key, value]) => Number(value) !== Number(preset.fields[key]))
          .map(([key, value]) => [key, Number(value)]),
      );

      const scored = await inject(selected, changed);
      setResult(scored);
      onScored(scored);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit}>
      <div className="presets">
        {presets.map((preset) => (
          <button
            key={preset.name}
            type="button"
            className="preset"
            aria-pressed={preset.name === selected}
            disabled={busy}
            onClick={() => choose(preset)}
          >
            <div className="name">{preset.label}</div>
            <div className="why">{preset.description}</div>
          </button>
        ))}
        {!presets.length && !error && (
          <div className="why">Loading scenarios…</div>
        )}
      </div>

      {selected && (
        <div className="overrides">
          <div className="micro" style={{ marginBottom: 10 }}>
            Adjust before sending
          </div>
          {EDITABLE.map(({ key, label, step }) => (
            <div className="field" key={key}>
              <label htmlFor={`f-${key}`}>{label}</label>
              <input
                id={`f-${key}`}
                type="number"
                step={step}
                value={overrides[key] ?? ""}
                disabled={busy}
                onChange={(e) =>
                  setOverrides((prev) => ({ ...prev, [key]: e.target.value }))
                }
              />
            </div>
          ))}
        </div>
      )}

      <button type="submit" className="inject-btn" disabled={busy || !selected}>
        {busy ? "Scoring…" : "Score transaction"}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result-card">
          <div className="micro" style={{ marginBottom: 8 }}>
            Result
          </div>
          <Ballot ensemble={result.ensemble} size="lg" />
          <div className="ballot-legend">
            <span className="verdict" data-anomaly={result.ensemble.is_anomaly}>
              {result.ensemble.is_anomaly ? "Flagged" : "Clear"}
            </span>
            <span style={{ fontSize: 12, color: "var(--ink-dim)" }}>
              <span className="num">{result.ensemble.votes_for}</span> of{" "}
              <span className="num">{result.ensemble.votes_total}</span>
            </span>
          </div>
          <p className="sentence" style={{ marginTop: 12, marginBottom: 0 }}>
            {result.explanation.plain_english}
          </p>
        </div>
      )}
    </form>
  );
}
