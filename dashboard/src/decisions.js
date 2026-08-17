/**
 * Analyst decisions on scored transactions.
 *
 * The feed re-fetches every few seconds and replaces the list wholesale, so a
 * decision held on the row object would vanish on the next poll. Decisions are
 * therefore kept in a separate map keyed by transaction identity and merged
 * onto rows as they arrive.
 *
 * Storage is the browser's localStorage: decisions survive a reload but are
 * local to this browser. They are not sent to the API and not written to
 * Firestore — see the note in the console footer, which says so on screen
 * rather than letting a viewer assume otherwise.
 */

const STORAGE_KEY = "anomaly-console.decisions.v1";

export const APPROVE = "approve";
export const BLOCK = "block";

/** Stable identity for a scored transaction, independent of feed position. */
export function decisionKey(result) {
  return `${result.transaction_id || "tx"}::${result.scored_at}`;
}

/** localStorage throws in some privacy modes; a demo must not die for that. */
function safeRead() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function safeWrite(decisions) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(decisions));
  } catch {
    /* decisions stay in memory for this session only */
  }
}

export function loadDecisions() {
  const stored = safeRead();
  // Drop anything that isn't a recognised verdict, so a stale or hand-edited
  // storage entry cannot put the UI into a state it has no rendering for.
  return Object.fromEntries(
    Object.entries(stored).filter(
      ([, entry]) => entry?.verdict === APPROVE || entry?.verdict === BLOCK,
    ),
  );
}

export function persistDecisions(decisions) {
  safeWrite(decisions);
}

/**
 * Did the analyst disagree with the model?
 *
 * Blocking a transaction the model cleared, or approving one it flagged, is an
 * override — the cases worth counting, because each one is a labelled example
 * of the ensemble being wrong.
 */
export function isOverride(result, verdict) {
  if (!verdict) return false;
  const modelSaysFraud = result.ensemble.is_anomaly;
  return verdict === BLOCK ? !modelSaysFraud : modelSaysFraud;
}

export function summarise(results, decisions) {
  let reviewed = 0;
  let overrides = 0;
  for (const result of results) {
    const entry = decisions[decisionKey(result)];
    if (!entry) continue;
    reviewed += 1;
    if (isOverride(result, entry.verdict)) overrides += 1;
  }
  return { reviewed, overrides };
}
