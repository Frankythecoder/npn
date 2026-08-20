import { useEffect, useState } from "react";
import BatchResults from "./components/BatchResults.jsx";
import InputTabs from "./components/InputTabs.jsx";
import { getHealth } from "./api.js";
import { loadBatch, persistBatch, toBatchView } from "./batch.js";

const POLL_MS = 3000;

export default function App() {
  const [live, setLive] = useState(null);
  // A scored upload. Held here rather than in the rail, so the report is the
  // main column's own state rather than the upload panel's.
  //
  // Seeded from localStorage: a reload used to drop the report, which left the
  // operator looking at the same transactions with the file's name and its
  // counts stripped off.
  const [batch, setBatch] = useState(loadBatch);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      try {
        const health = await getHealth();
        if (!cancelled) setLive(Boolean(health.model_loaded));
      } catch {
        if (!cancelled) setLive(false);
      }
    };

    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Reduced to what the report renders before it is stored, because the file's
  // cleared rows are only ever a count and there can be tens of thousands.
  function handleBatch(uploaded) {
    const view = toBatchView(uploaded);
    setBatch(view);
    persistBatch(view);
  }

  return (
    <div className="app">
      <header className="masthead">
        <h1>Fraud Analytics</h1>
        <span className="micro">Four models vote · two agree to flag</span>
        <div className="rule">
          <span className="status" data-live={live === true}>
            <span className="dot" aria-hidden="true" />
            {live === null
              ? "Connecting"
              : live
                ? "Model loaded"
                : "Backend unavailable"}
          </span>
        </div>
      </header>

      <div className="body">
        <aside className="rail">
          <InputTabs onBatch={handleBatch} />
        </aside>

        <main className="main">{batch && <BatchResults batch={batch} />}</main>
      </div>
    </div>
  );
}
