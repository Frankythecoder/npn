/**
 * One scored transaction, as a row in the results table.
 *
 * A verdict, the transaction, its amount, and how far into each model's
 * training distribution it fell. Each row is a button that opens its review
 * drawer, and the header holds a track open for the chevron that says so.
 */

// The four detectors that vote, in the registry's canonical order. Looked up by
// name rather than by position so a change to DETECTOR_ORDER cannot silently
// retitle a column.
export const DETECTORS = [
  { name: "isolation_forest", short: "IF", label: "Isolation Forest" },
  { name: "lof", short: "LOF", label: "Local Outlier Factor" },
  { name: "one_class_svm", short: "SVM", label: "One-Class SVM" },
  { name: "dbscan", short: "DBS", label: "DBSCAN" },
];

export const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

export function ResultTable({ children }) {
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
        {children}
      </div>
    </div>
  );
}

export function ResultRow({ result, isOpen, confirmed = false, onToggle }) {
  const { ensemble, detectors = [] } = result;
  const byName = Object.fromEntries(detectors.map((d) => [d.name, d]));
  const plain = result.explanation?.plain_english;

  return (
    <button
      type="button"
      className="batch-row"
      data-confirmed={confirmed}
      aria-expanded={isOpen}
      onClick={onToggle}
      title={confirmed ? `Confirmed as actual fraud. ${plain ?? ""}` : plain}
    >
      {/* The model's verdict is a probability; a confirmation is not. Once an
          analyst has validated the row it stops reading as "possible". */}
      <span
        className="verdict"
        data-anomaly={ensemble.is_anomaly}
        data-confirmed={confirmed}
      >
        {confirmed ? "Fraud" : ensemble.is_anomaly ? "Flagged" : "Clear"}
        <span className="tally num">
          {confirmed ? "confirmed" : `${ensemble.votes_for}/${ensemble.votes_total}`}
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
