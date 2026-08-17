import { useCallback, useEffect, useRef, useState } from "react";
import BatchResults from "./components/BatchResults.jsx";
import Feed from "./components/Feed.jsx";
import InputTabs from "./components/InputTabs.jsx";
import { getHealth, getRecent } from "./api.js";
import { loadDecisions, persistDecisions, summarise } from "./decisions.js";

const POLL_MS = 3000;

export default function App() {
  const [results, setResults] = useState([]);
  const [expandedKey, setExpandedKey] = useState(null);
  const [freshKey, setFreshKey] = useState(null);
  const [live, setLive] = useState(null);
  const [feedError, setFeedError] = useState(null);
  // Decisions live outside the feed rows: the poll replaces the list wholesale,
  // so anything stored on a row object would be lost every few seconds.
  const [decisions, setDecisions] = useState(loadDecisions);
  // A scored upload. Held here rather than in the rail so it survives switching
  // back to the Scenarios tab, and so the feed and the batch share one decisions
  // map. Null means the live feed owns the main column.
  const [batch, setBatch] = useState(null);
  // True while a CSV is being scored. The feed fills with the uploaded rows as
  // the chunks land, and offering a decision on each one mid-run asks the
  // operator to review a list that is still being written.
  const [scoring, setScoring] = useState(false);

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

  function handleDecide(key, verdict) {
    setDecisions((prev) => {
      const next = { ...prev };
      if (verdict === null) {
        delete next[key];
      } else {
        next[key] = { verdict, at: new Date().toISOString() };
      }
      persistDecisions(next);
      return next;
    });
  }

  function handleScored(result) {
    pendingRef.current = result;
    const key = `${result.transaction_id || "tx"}-${result.scored_at}-0`;
    setResults((prev) => [result, ...prev]);
    setExpandedKey(key);
    setFreshKey(key);
    window.setTimeout(() => setFreshKey(null), 700);
  }

  const flagged = results.filter((r) => r.ensemble.is_anomaly).length;
  const { reviewed, overrides } = summarise(results, decisions);

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
          <InputTabs
            onScored={handleScored}
            onBatch={setBatch}
            onBusyChange={setScoring}
          />
        </aside>

        <main className="main">
          {batch ? (
            <BatchResults batch={batch} onClose={() => setBatch(null)} />
          ) : (
            <LiveFeed
              results={results}
              flagged={flagged}
              reviewed={reviewed}
              overrides={overrides}
              feedError={feedError}
              expandedKey={expandedKey}
              onToggle={setExpandedKey}
              freshKey={freshKey}
              decisions={decisions}
              onDecide={handleDecide}
              scoring={scoring}
            />
          )}
        </main>
      </div>
    </div>
  );
}

function LiveFeed({
  results,
  flagged,
  reviewed,
  overrides,
  feedError,
  expandedKey,
  onToggle,
  freshKey,
  decisions,
  onDecide,
  scoring,
}) {
  return (
    <>
      <div className="section-head">
        <span className="micro">Recent transactions</span>
        <span className="micro">
          <span className="num">{flagged}</span> flagged of{" "}
          <span className="num">{results.length}</span>
          {/* The review tallies are suppressed alongside the per-row controls
              while an upload is scoring — a running count of decisions nobody
              can make yet is noise. */}
          {!scoring && (
            <>
              {" · "}
              <span className="num">{reviewed}</span> reviewed
              {overrides > 0 && (
                <>
                  {" · "}
                  <span className="override-count">
                    <span className="num">{overrides}</span> override
                    {overrides === 1 ? "" : "s"}
                  </span>
                </>
              )}
            </>
          )}
        </span>
      </div>

      {feedError && <div className="error">{feedError}</div>}

      <Feed
        results={results}
        expandedKey={expandedKey}
        onToggle={onToggle}
        freshKey={freshKey}
        decisions={decisions}
        onDecide={onDecide}
        scoring={scoring}
      />

      <p className="footnote">
        Rows are ordered newest first and refresh every {POLL_MS / 1000} seconds.
        Select a row to see how each model voted, what drove the decision, and to
        approve or block it. Analyst decisions are held in this browser only —
        they are not sent to the API.
      </p>
    </>
  );
}
