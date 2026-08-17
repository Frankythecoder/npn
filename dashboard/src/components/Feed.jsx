import { Fragment } from "react";
import Ballot from "./Ballot.jsx";
import DetailPanel from "./DetailPanel.jsx";
import { decisionKey, isOverride, APPROVE } from "../decisions.js";

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

export default function Feed({
  results,
  expandedKey,
  onToggle,
  freshKey,
  decisions,
  onDecide,
  // Set while a CSV upload is scoring. The rows arriving are the upload's and
  // the list is rewritten every poll, so the feed becomes a read-only ticker:
  // no decision control, and no expanding a row whose position is about to move
  // out from under the click.
  scoring = false,
}) {
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
    <div className="feed" data-scoring={scoring}>
      {results.map((result, index) => {
        const key = rowKey(result, index);
        const isOpen = !scoring && key === expandedKey;
        const amount = Number(result.raw?.TransactionAmount ?? 0);
        const dKey = decisionKey(result);
        const decision = decisions[dKey];
        const override = isOverride(result, decision?.verdict);

        return (
          <Fragment key={key}>
            <button
              type="button"
              className={`feed-row${key === freshKey ? " is-new" : ""}`}
              disabled={scoring}
              aria-expanded={scoring ? undefined : isOpen}
              onClick={() => onToggle(isOpen ? null : key)}
            >
              <Ballot ensemble={result.ensemble} size="sm" />
              <span className="txid">{result.transaction_id || "—"}</span>
              <span className="amount num">{money.format(amount)}</span>
              <span className="tally num">
                {result.ensemble.votes_for}/{result.ensemble.votes_total}
              </span>
              {!scoring && (
                <span className="review">
                  {decision ? (
                    <span
                      className="badge"
                      data-verdict={decision.verdict}
                      data-override={override}
                      title={
                        override
                          ? "Analyst overrode the model"
                          : "Analyst agreed with the model"
                      }
                    >
                      {decision.verdict === APPROVE ? "Approved" : "Blocked"}
                      {override && <span className="flagdot" aria-hidden="true" />}
                    </span>
                  ) : (
                    <span className="badge pending">Review</span>
                  )}
                </span>
              )}
              {!scoring && (
                <span className="chev" aria-hidden="true">
                  ›
                </span>
              )}
            </button>
            {isOpen && (
              <DetailPanel
                result={result}
                decision={decision}
                onDecide={(verdict) => onDecide(dKey, verdict)}
              />
            )}
          </Fragment>
        );
      })}
    </div>
  );
}
