import { Fragment } from "react";
import Ballot from "./Ballot.jsx";
import DetailPanel from "./DetailPanel.jsx";

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function rowKey(result, index) {
  // scored_at alone can collide when several transactions are injected inside
  // the same millisecond, so pair it with the id and position.
  return `${result.transaction_id || "tx"}-${result.scored_at}-${index}`;
}

export default function Feed({ results, expandedKey, onToggle, freshKey }) {
  if (!results.length) {
    return (
      <div className="feed">
        <div className="empty">
          No transactions scored yet. Inject one from the panel on the left.
        </div>
      </div>
    );
  }

  return (
    <div className="feed">
      {results.map((result, index) => {
        const key = rowKey(result, index);
        const isOpen = key === expandedKey;
        const amount = Number(result.raw?.TransactionAmount ?? 0);

        return (
          <Fragment key={key}>
            <button
              type="button"
              className={`feed-row${key === freshKey ? " is-new" : ""}`}
              aria-expanded={isOpen}
              onClick={() => onToggle(isOpen ? null : key)}
            >
              <Ballot ensemble={result.ensemble} size="sm" />
              <span className="txid">{result.transaction_id || "—"}</span>
              <span className="amount num">{money.format(amount)}</span>
              <span className="tally num">
                {result.ensemble.votes_for}/{result.ensemble.votes_total}
              </span>
              <span className="chev" aria-hidden="true">
                ›
              </span>
            </button>
            {isOpen && <DetailPanel result={result} />}
          </Fragment>
        );
      })}
    </div>
  );
}
