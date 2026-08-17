import { APPROVE, BLOCK, decisionKey, isOverride } from "../decisions.js";

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
 * Decisions go through the same localStorage store as the live feed, so the
 * override count in the masthead stays true across both views.
 */
export default function BatchResults({ batch, decisions, onDecide, onClose }) {
  const flagged = batch.results.filter((r) => r.ensemble.is_anomaly);
  const clear = batch.results.filter((r) => !r.ensemble.is_anomaly);

  const reviewed = flagged.filter((r) => decisions[decisionKey(r)]).length;

  return (
    <div className="batch">
      <div className="batch-head">
        <div>
          <div className="micro" style={{ marginBottom: 4 }}>
            Uploaded file
          </div>
          <div className="batch-title">{batch.name}</div>
        </div>
        <button type="button" className="linkish" onClick={onClose}>
          Back to live feed
        </button>
      </div>

      <div className="batch-stats">
        <Stat value={batch.results.length} label="scored" />
        <Stat value={flagged.length} label="flagged" tone="high" />
        <Stat value={clear.length} label="clear" tone="away" />
        <Stat value={batch.rejected.length} label="rejected" tone="dim" />
        <Stat value={reviewed} label={`of ${flagged.length} reviewed`} />
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
        <ResultTable
          results={flagged}
          decisions={decisions}
          onDecide={onDecide}
          decidable
        />
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

function ResultTable({ results, decisions, onDecide, decidable = false }) {
  return (
    <div className="batch-wrap">
      <div className="batch-table" data-decidable={decidable}>
        <div className="batch-row batch-header">
          <span>Verdict</span>
          <span>Transaction</span>
          <span className="r">Amount</span>
          {DETECTORS.map((d) => (
            <span className="r" key={d.name} title={`${d.label} score percentile`}>
              {d.short}
            </span>
          ))}
          {decidable && <span>Decision</span>}
        </div>

        {results.map((result, index) => (
          <Row
            key={`${result.transaction_id || "tx"}-${result.scored_at}-${index}`}
            result={result}
            decision={decisions[decisionKey(result)]}
            onDecide={onDecide}
            decidable={decidable}
          />
        ))}
      </div>
    </div>
  );
}

function Row({ result, decision, onDecide, decidable }) {
  const { ensemble, detectors } = result;
  const byName = Object.fromEntries(detectors.map((d) => [d.name, d]));
  const verdict = decision?.verdict ?? null;
  const override = isOverride(result, verdict);
  const key = decisionKey(result);

  return (
    <div className="batch-row" title={result.explanation.plain_english}>
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

      {decidable && (
        <span className="batch-decide" data-override={override}>
          <button
            type="button"
            className="decide approve"
            aria-pressed={verdict === APPROVE}
            onClick={() => onDecide(key, verdict === APPROVE ? null : APPROVE)}
          >
            Approve
          </button>
          <button
            type="button"
            className="decide block"
            aria-pressed={verdict === BLOCK}
            onClick={() => onDecide(key, verdict === BLOCK ? null : BLOCK)}
          >
            Deny
          </button>
        </span>
      )}
    </div>
  );
}
