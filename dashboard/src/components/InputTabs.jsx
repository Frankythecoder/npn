import { useState } from "react";
import CsvPanel from "./CsvPanel.jsx";

const TABS = [{ id: "csv", label: "Test" }];

/**
 * How a transaction gets into the console: a CSV carries a whole file at once.
 * The panel keeps its own state, so switching away from the rail does not
 * discard it.
 */
export default function InputTabs({ onBatch, onBusyChange }) {
  const [tab, setTab] = useState("csv");

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

      <CsvPanel onBatch={onBatch} onBusyChange={onBusyChange} />
    </div>
  );
}
