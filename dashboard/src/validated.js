/**
 * Analyst confirmation that a flagged transaction was genuinely fraud.
 *
 * Deliberately a separate store from decisions.js rather than a third verdict
 * inside it. A decision is an action taken on a transaction — approve it, block
 * it. This is a statement about the label: the ensemble said "possible fraud",
 * and the analyst is confirming it was real. Keeping them apart means
 * confirming a row in the batch report is not silently also an action taken on
 * that transaction.
 *
 * Same storage contract as decisions: localStorage, so it survives a reload but
 * is local to this browser. Nothing here is sent to the API — the drawer says
 * so on screen rather than letting a viewer assume otherwise.
 */

const STORAGE_KEY = "anomaly-console.validated.v1";

/** localStorage throws in some privacy modes; a demo must not die for that. */
function safeRead() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function safeWrite(validated) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(validated));
  } catch {
    /* confirmations stay in memory for this session only */
  }
}

export function loadValidated() {
  // Drop anything without a timestamp, so a stale or hand-edited storage entry
  // cannot put the UI into a state it has no rendering for.
  return Object.fromEntries(
    Object.entries(safeRead()).filter(
      ([, entry]) => typeof entry?.at === "string",
    ),
  );
}

export function persistValidated(validated) {
  safeWrite(validated);
}

/** Count of confirmed rows among `results`. */
export function countValidated(results, validated, keyOf) {
  return results.reduce(
    (total, result) => total + (validated[keyOf(result)] ? 1 : 0),
    0,
  );
}
