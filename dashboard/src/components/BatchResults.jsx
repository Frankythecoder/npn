import { Fragment, useState } from "react";
import ShapChart from "./ShapChart.jsx";

// The four detectors that vote, in the registry's canonical order. Looked up by
// name rather than by position so a change to DETECTOR_ORDER cannot silently
// retitle a column.
const DETECTORS = [
  { name: "isolation_forest", short: "IF", label: "Isolation Forest" },
  { name: "lof", short: "LOF", label: "Local Outlier Factor" },
  { name: "one_class_svm", short: "SVM", label: "One-Class SVM" },
  { name: "dbscan", short: "DBS", label: "DBSCAN" },
];

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

/**
 * The result of one uploaded file.
 *
 * Only the flagged transactions are listed. Clear ones are the bulk of any file
 * and carry no decision to make, so they stay counted in the summary above
 * rather than filling a table nobody reads top to bottom.
 *
 * Each flagged row opens a review drawer carrying the same four model scores
 * and SHAP attribution the live feed shows, so a file can be worked through
 * without scoring its rows again one at a time. This view still only reports:
 * approving or blocking a transaction remains the live feed's job.
 */
export default function BatchResults({ batch }) {
  const flagged = batch.results.filter((r) => r.ensemble.is_anomaly);
  const clear = batch.results.filter((r) => !r.ensemble.is_anomaly);

  return (
    <div className="batch">
      <div className="batch-head">
        <div>
          <div className="micro" style={{ marginBottom: 4 }}>
            Uploaded file
          </div>
          <div className="batch-title">{batch.name}</div>
        </div>
      </div>

      <div className="batch-stats">
        <Stat value={batch.results.length} label="scored" />
        <Stat value={flagged.length} label="flagged" tone="high" />
        <Stat value={clear.length} label="clear" tone="away" />
        <Stat value={batch.rejected.length} label="rejected" tone="dim" />
      </div>

      {batch.cancelled && (
        <p className="footnote" style={{ marginTop: 0 }}>
          Stopped early — {batch.scanned.toLocaleString()} of the file's rows were
          processed.
        </p>
      )}

      <Section
        title="Flagged as possible fraud"
        count={flagged.length}
        tone="high"
        empty="No transaction in this file was flagged."
      >
        <ResultTable results={flagged} />
        <p className="footnote">
          The other <span className="num">{clear.length.toLocaleString()}</span>{" "}
          transactions in this file were cleared and are not listed. They were
          still scored, and appear in the live feed.
        </p>
      </Section>

      {batch.rejected.length > 0 && (
        <Section title="Rejected rows" count={batch.rejected.length} tone="dim">
          <div className="rejects">
            {batch.rejected.map((item) => (
              <div className="reject-row" key={item.row}>
                <span className="num">line {item.row}</span>
                <span>{item.reason}</span>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Stat({ value, label, tone }) {
  return (
    <div className="batch-stat" data-tone={tone}>
      <span className="num v">{value.toLocaleString()}</span>
      <span className="micro">{label}</span>
    </div>
  );
}

function Section({ title, count, tone, empty, children }) {
  return (
    <section className="batch-section">
      <div className="section-head">
        <span className="micro" data-tone={tone}>
          {title} · <span className="num">{count.toLocaleString()}</span>
        </span>
      </div>
      {count === 0 ? <p className="footnote">{empty}</p> : children}
    </section>
  );
}

function ResultTable({ results }) {
  // One drawer at a time: these rows are read in sequence, and leaving earlier
  // ones open pushes the row being worked on off the screen.
  const [openKey, setOpenKey] = useState(null);

  return (
    <div className="batch-wrap">
      <div className="batch-table">
        <div className="batch-row batch-header">
          <span>Verdict</span>
          <span>Transaction</span>
          <span className="r">Amount</span>
          {DETECTORS.map((d) => (
            <span className="r" key={d.name} title={`${d.label} score percentile`}>
              {d.short}
            </span>
          ))}
          {/* Holds the chevron's column open in the header. */}
          <span aria-hidden="true" />
        </div>

        {results.map((result, index) => {
          const key = `${result.transaction_id || "tx"}-${result.scored_at}-${index}`;
          const isOpen = key === openKey;

          return (
            <Fragment key={key}>
              <Row
                result={result}
                isOpen={isOpen}
                onToggle={() => setOpenKey(isOpen ? null : key)}
              />
              {isOpen && <ReviewDrawer result={result} />}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}

function Row({ result, isOpen, onToggle }) {
  const { ensemble, detectors } = result;
  const byName = Object.fromEntries(detectors.map((d) => [d.name, d]));

  return (
    <button
      type="button"
      className="batch-row"
      aria-expanded={isOpen}
      onClick={onToggle}
      title={result.explanation.plain_english}
    >
      <span className="verdict" data-anomaly={ensemble.is_anomaly}>
        {ensemble.is_anomaly ? "Flagged" : "Clear"}
        <span className="tally num">
          {ensemble.votes_for}/{ensemble.votes_total}
        </span>
      </span>

      <span className="txid">{result.transaction_id || "—"}</span>

      <span className="amount num r">
        {money.format(Number(result.raw?.TransactionAmount ?? 0))}
      </span>

      {DETECTORS.map((d) => {
        const cell = byName[d.name];
        return (
          <span
            className="score num r"
            key={d.name}
            data-cast={cell?.flag === 1}
            title={cell ? `${d.label}: raw score ${cell.score.toFixed(4)}` : d.label}
          >
            {cell ? cell.score_percentile.toFixed(1) : "—"}
          </span>
        );
      })}

      <span className="chev" aria-hidden="true">
        ›
      </span>
    </button>
  );
}

/**
 * The review drawer for one flagged row.
 *
 * Deliberately the same two readings the live feed's detail panel gives — how
 * each voting model scored, and which features SHAP says drove it — because an
 * operator who has learned to read one should not have to learn the other. The
 * decision controls are the one thing not carried over: this view reports.
 */
function ReviewDrawer({ result }) {
  const { detectors, ensemble, explanation } = result;
  const byName = Object.fromEntries(detectors.map((d) => [d.name, d]));

  return (
    <div className="batch-detail">
      <div>
        <div className="micro" style={{ marginBottom: 8 }}>
          How each model voted
        </div>

        <div className="batch-dets">
          {DETECTORS.map((d) => {
            const cell = byName[d.name];
            return (
              <div className="batch-det" key={d.name} data-cast={cell?.flag === 1}>
                <span className="mark" aria-hidden="true" />
                <span className="dname">{d.label}</span>
                <span
                  className="raw num"
                  title={`${d.label}: raw anomaly score`}
                >
                  {cell ? cell.score.toFixed(4) : "—"}
                </span>
                <span
                  className="pct num"
                  title="Percentile against this model's training scores"
                >
                  {cell ? cell.score_percentile.toFixed(1) : "—"}
                </span>
              </div>
            );
          })}
        </div>

        <p className="footnote">
          <span className="num">{ensemble.votes_for}</span> of{" "}
          <span className="num">{ensemble.votes_total}</span> models flagged this
          — <span className="num">{ensemble.votes_required}</span> needed.
          Percentiles rank against each model&apos;s own training scores.
        </p>
      </div>

      <div className="batch-attrib">
        <ShapChart features={explanation.top_features} />

        <div>
          <div className="micro" style={{ marginBottom: 8 }}>
            Explanation
          </div>
          <p className="sentence" style={{ margin: 0 }}>
            {explanation.plain_english}
          </p>
        </div>
      </div>
    </div>
  );
}
