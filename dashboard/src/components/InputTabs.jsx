import { useState } from "react";
import CsvPanel from "./CsvPanel.jsx";
import InjectPanel from "./InjectPanel.jsx";

const TABS = [
  { id: "scenarios", label: "Scenarios" },
  { id: "csv", label: "Upload CSV" },
];

/**
 * The two ways a transaction gets into the console.
 *
 * Scenarios fire one preset at a time and are what a live walkthrough uses; a
 * CSV carries a whole file at once. They share nothing but the rail, so the tab
 * strip is the whole of the coordination between them — each panel keeps its own
 * state, and switching away does not discard it.
 */
export default function InputTabs({ onScored, onBatch, onBusyChange }) {
  const [tab, setTab] = useState("scenarios");

  return (
    <div>
      <div className="tabs" role="tablist" aria-label="Transaction input">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            className="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "scenarios" ? (
        <InjectPanel onScored={onScored} />
      ) : (
        <CsvPanel onBatch={onBatch} onBusyChange={onBusyChange} />
      )}
    </div>
  );
}
