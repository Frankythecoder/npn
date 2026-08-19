/**
 * The report for the last scored upload.
 *
 * The batch lived only in App's state, so a reload dropped it: the file's name,
 * its counts and its flagged rows were gone even though those transactions had
 * already been scored. This
 * keeps the report across a reload the same way decisions.js and validated.js
 * keep their maps — localStorage, local to this browser, never sent to the API.
 *
 * Only the flagged rows are stored. The report lists no others: a cleared row is
 * a number above the table and nothing more, so keeping every scored row would
 * multiply the stored bytes tenfold to render nothing extra. The counts are
 * stored as counts for the same reason.
 */

const STORAGE_KEY = "anomaly-console.batch.v1";

// localStorage is a few megabytes per origin, and a large enough file can flag
// more rows than that holds. A write that overflows is retried with the table
// capped here rather than abandoned, because the summary — which is what the
// reader looks at first — costs a few dozen bytes and is always exact.
const MAX_STORED_ROWS = 500;

/**
 * The shape BatchResults renders, derived once when an upload finishes.
 *
 * Splitting flagged from clear here rather than in the component is what makes
 * the report storable: the view needs every flagged row but only a count of the
 * cleared ones, and that distinction is lost if `results` is carried whole.
 */
export function toBatchView(batch) {
  const rows = batch.results.filter((r) => r.ensemble.is_anomaly);
  return {
    name: batch.name,
    scored: batch.results.length,
    flagged: rows.length,
    cleared: batch.results.length - rows.length,
    rows,
    rejected: batch.rejected ?? [],
    cancelled: Boolean(batch.cancelled),
    scanned: batch.scanned ?? 0,
    // Carried so the results view can say the batch was filled. A verdict on a
    // filled row must not be read as a verdict on the row alone.
    filled: batch.filled ?? [],
    // True only when the stored table is shorter than the flagged count, so a
    // restored report never implies it is listing rows it does not have.
    partial: false,
  };
}

function safeWrite(view) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(view));
    return true;
  } catch {
    return false;
  }
}

export function clearBatch() {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* nothing to clear; the batch is in memory for this session only */
  }
}

export function persistBatch(view) {
  if (safeWrite(view)) return;

  if (view.rows.length > MAX_STORED_ROWS) {
    const capped = {
      ...view,
      rows: view.rows.slice(0, MAX_STORED_ROWS),
      partial: true,
    };
    if (safeWrite(capped)) return;
  }

  // Storage is unavailable or still too small. Drop whatever is there rather
  // than leaving an earlier file's report to be restored in this one's place.
  clearBatch();
}

function count(value) {
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

/**
 * Can this row be drawn?
 *
 * A stored row is only ever as trustworthy as the browser it came from, and the
 * table and its drawer read these fields without guarding. Dropping an entry
 * that is missing one is cheaper than a blank console.
 */
function renderable(result) {
  return Boolean(
    result &&
      result.ensemble &&
      Array.isArray(result.detectors) &&
      result.explanation &&
      typeof result.explanation.plain_english === "string",
  );
}

export function loadBatch() {
  let stored;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    stored = raw ? JSON.parse(raw) : null;
  } catch {
    // Unreadable, unparseable, or a privacy mode that refuses storage.
    return null;
  }

  if (!stored || typeof stored.name !== "string") return null;

  const held = Array.isArray(stored.rows) ? stored.rows : [];
  const rows = held.filter(renderable);

  return {
    name: stored.name,
    scored: count(stored.scored),
    flagged: count(stored.flagged),
    cleared: count(stored.cleared),
    rows,
    rejected: Array.isArray(stored.rejected)
      ? stored.rejected.filter((item) => item && typeof item.reason === "string")
      : [],
    cancelled: Boolean(stored.cancelled),
    scanned: count(stored.scanned),
    filled: Array.isArray(stored.filled)
      ? stored.filled.filter((column) => typeof column === "string")
      : [],
    partial: Boolean(stored.partial) || rows.length < held.length,
  };
}
