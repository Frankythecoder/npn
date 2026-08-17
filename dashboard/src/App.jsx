import { useCallback, useEffect, useRef, useState } from "react";
import Feed from "./components/Feed.jsx";
import InjectPanel from "./components/InjectPanel.jsx";
import { getHealth, getRecent } from "./api.js";

const POLL_MS = 3000;

export default function App() {
  const [results, setResults] = useState([]);
  const [expandedKey, setExpandedKey] = useState(null);
  const [freshKey, setFreshKey] = useState(null);
  const [live, setLive] = useState(null);
  const [feedError, setFeedError] = useState(null);

  // A locally-injected result is prepended immediately rather than waiting for
  // the next poll, so the demo reads as instant. The next poll replaces the
  // list wholesale, at which point the server's own copy takes over.
  const pendingRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const recent = await getRecent(50);
      setFeedError(null);
      setResults((prev) => {
        const pending = pendingRef.current;
        if (!pending) return recent;
        // Keep the optimistic row until the server's feed contains it.
        const landed = recent.some(
          (r) =>
            r.transaction_id === pending.transaction_id &&
            r.scored_at === pending.scored_at,
        );
        if (landed) {
          pendingRef.current = null;
          return recent;
        }
        return [pending, ...recent];
      });
    } catch (err) {
      setFeedError(err.message);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      await refresh();
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
  }, [refresh]);

  function handleScored(result) {
    pendingRef.current = result;
    const key = `${result.transaction_id || "tx"}-${result.scored_at}-0`;
    setResults((prev) => [result, ...prev]);
    setExpandedKey(key);
    setFreshKey(key);
    window.setTimeout(() => setFreshKey(null), 700);
  }

  const flagged = results.filter((r) => r.ensemble.is_anomaly).length;

  return (
    <div className="app">
      <header className="masthead">
        <h1>Anomaly Console</h1>
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
          <InjectPanel onScored={handleScored} />
        </aside>

        <main className="main">
          <div className="section-head">
            <span className="micro">Recent transactions</span>
            <span className="micro">
              <span className="num">{flagged}</span> flagged of{" "}
              <span className="num">{results.length}</span>
            </span>
          </div>

          {feedError && <div className="error">{feedError}</div>}

          <Feed
            results={results}
            expandedKey={expandedKey}
            onToggle={setExpandedKey}
            freshKey={freshKey}
          />

          <p className="footnote">
            Rows are ordered newest first and refresh every {POLL_MS / 1000}{" "}
            seconds. Select a row to see how each model voted and what drove the
            decision.
          </p>
        </main>
      </div>
    </div>
  );
}
