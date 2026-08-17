/**
 * CSV parsing and column classification for uploads.
 *
 * Parsing happens in the browser rather than the API for two reasons: the file
 * never has to be uploaded twice when a column check fails, and the backend
 * stays on JSON, so no multipart dependency is added to a service that has none.
 *
 * The column groups below mirror backend/csvingest.py. They are duplicated
 * rather than fetched because they gate the UI *before* anything is sent — the
 * point of the pre-flight report is that a bad file never leaves the browser.
 * If a raw input is ever added to the model, both lists have to change; the
 * backend asserts its own copy against RAW_INPUT_FIELDS at import, so the
 * mismatch surfaces there rather than silently here.
 */

/** Without at least one of these, a file is not a transaction file. */
export const CRUCIAL_COLUMNS = [
  "TransactionAmount",
  "AccountBalance",
  "CustomerAge",
  "TransactionDuration",
  "LoginAttempts",
];

/** Every column the scorer reads. Anything absent is filled server-side. */
export const SCORING_COLUMNS = [
  ...CRUCIAL_COLUMNS,
  "TransactionType",
  "Channel",
  "CustomerOccupation",
  "Location",
  "AccountID",
  "DeviceID",
  "TransactionDate",
];

/** Recognised columns of original.csv that the model has never read. */
export const IGNORED_COLUMNS = [
  "TransactionID",
  "IP Address",
  "MerchantID",
  "PreviousTransactionDate",
];

/**
 * Rows per request.
 *
 * Measured at roughly 65ms a row, so 200 is about 13 seconds of work — short
 * enough that the feed's own three-second polling is not starved behind it on a
 * service pinned to one worker, and frequent enough that the progress bar
 * actually moves. The API's own ceiling is 500; this stays well under it.
 */
export const CHUNK_SIZE = 200;

/**
 * Parse CSV text into a header and rows.
 *
 * Hand-rolled rather than a library: the whole grammar needed here is quoted
 * fields, doubled quotes inside them, and either line ending. Adding a
 * dependency to a dashboard whose only two are React and React-DOM would cost
 * more than it saves.
 *
 * Rows are returned verbatim — short rows, blank cells and duplicate ids all
 * pass through, because the backend rejects rows individually and reports the
 * line number. Only trailing blank lines are dropped, so that a file ending in a
 * newline does not report a phantom rejected row.
 */
export function parseCsv(text) {
  const clean = text.replace(/^﻿/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < clean.length; i += 1) {
    const ch = clean[i];

    if (quoted) {
      if (ch === '"') {
        if (clean[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }

  if (field !== "" || row.length) {
    row.push(field);
    rows.push(row);
  }

  // Trailing blank lines only. An interior blank line is a real defect in the
  // file and is left in place so it gets reported against its own line number.
  while (rows.length && rows[rows.length - 1].every((cell) => cell.trim() === "")) {
    rows.pop();
  }

  if (!rows.length) return { columns: [], rows: [], error: "The file is empty." };

  const [header, ...data] = rows;
  const columns = header.map((c) => c.trim());

  if (!data.length) {
    return { columns, rows: [], error: "The file has a header but no rows." };
  }

  return { columns, rows: data, error: null };
}

/**
 * Sort a header into what will be used, filled, and ignored.
 *
 * `ok` is the file-level gate: at least one crucial column present.
 */
export function classifyColumns(columns) {
  const seen = new Set(columns);
  const crucial = CRUCIAL_COLUMNS.filter((c) => seen.has(c));

  return {
    crucial,
    supplied: SCORING_COLUMNS.filter((c) => seen.has(c)),
    filled: SCORING_COLUMNS.filter((c) => !seen.has(c)),
    ignored: columns.filter((c) => IGNORED_COLUMNS.includes(c)),
    unknown: columns.filter(
      (c) => !SCORING_COLUMNS.includes(c) && !IGNORED_COLUMNS.includes(c),
    ),
    ok: crucial.length > 0,
  };
}

/** The message shown when the crucial-column gate rejects a file. */
export function rejectionMessage(columns) {
  const found = columns.length ? columns.join(", ") : "no columns";
  return (
    `This file has none of the required columns. At least one of ` +
    `${CRUCIAL_COLUMNS.join(", ")} must be present. Found: ${found}.`
  );
}
