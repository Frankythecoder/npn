import { APPROVE, BLOCK, isOverride } from "../decisions.js";

const RELATIVE = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

function decidedAgo(iso) {
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return RELATIVE.format(-Math.round(seconds / 60), "minute");
  return RELATIVE.format(-Math.round(seconds / 3600), "hour");
}

/**
 * The analyst's call on a scored transaction.
 *
 * The model's verdict is a recommendation, not an instruction, so the two
 * actions are stated in terms of what happens to the transaction — approve it
 * or block it — rather than "agree" and "disagree" with the model. Whether the
 * analyst overrode the model is then derived, not asked for twice.
 */
export default function DecisionBar({ result, decision, onDecide }) {
  const verdict = decision?.verdict ?? null;
  const override = isOverride(result, verdict);

  return (
    <div className="decision">
      <div className="micro" style={{ marginBottom: 8 }}>
        Analyst decision
      </div>

      <div className="decision-actions" role="group" aria-label="Analyst decision">
        <button
          type="button"
          className="decide approve"
          aria-pressed={verdict === APPROVE}
          onClick={() => onDecide(verdict === APPROVE ? null : APPROVE)}
        >
          Approve
        </button>
        <button
          type="button"
          className="decide block"
          aria-pressed={verdict === BLOCK}
          onClick={() => onDecide(verdict === BLOCK ? null : BLOCK)}
        >
          Block
        </button>
      </div>

      {verdict ? (
        <p className="decision-note">
          {verdict === APPROVE ? "Approved" : "Blocked"} {decidedAgo(decision.at)}
          {override && (
            <>
              {" · "}
              <span className="override-tag">
                overrides the model, which said{" "}
                {result.ensemble.is_anomaly ? "flag" : "clear"}
              </span>
            </>
          )}
          {". "}
          <button
            type="button"
            className="linkish"
            onClick={() => onDecide(null)}
          >
            Undo
          </button>
        </p>
      ) : (
        <p className="decision-note">
          Not yet reviewed. Approving a flagged transaction, or blocking a clear
          one, is recorded as an override.
        </p>
      )}
    </div>
  );
}
