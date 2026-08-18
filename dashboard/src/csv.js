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

/** Every column the scorer reads. */
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

/**
 * Columns the file must supply itself, under their own name or a synonym.
 * Mirrors REQUIRED_COLUMNS in backend/csvingest.py — nothing here has a
 * server-side fallback.
 */
export const REQUIRED_COLUMNS = [
  ...CRUCIAL_COLUMNS,
  "TransactionType",
  "Channel",
  "CustomerOccupation",
  "Location",
];

/**
 * Identity columns. Absent, these are synthesised rather than filled from
 * training data — the row is scored as a new account, which the backend says
 * in the row's warnings.
 */
export const IDENTITY_COLUMNS = ["AccountID", "DeviceID", "TransactionDate"];

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
 * Header synonyms, mirroring COLUMN_ALIASES in backend/csvingest.py.
 *
 * Duplicated for the same reason the column lists above are: the pre-flight
 * report has to tell the truth about a file *before* it is sent, and a column
 * the backend is going to read must not be shown here as unrecognised. The
 * backend resolves the header again on arrival and is the authority — this copy
 * only decides what the report says.
 *
 * Keys are match keys (lowercase, alphanumerics only). The number is the unit
 * multiplier the backend will apply; 1 is a plain rename.
 */
export const COLUMN_ALIASES = {
  amount: ["TransactionAmount", 1],
  txnamount: ["TransactionAmount", 1],
  trxamount: ["TransactionAmount", 1],
  transactionvalue: ["TransactionAmount", 1],
  transactionamt: ["TransactionAmount", 1],
  amountusd: ["TransactionAmount", 1],
  balance: ["AccountBalance", 1],
  accountbal: ["AccountBalance", 1],
  acctbalance: ["AccountBalance", 1],
  availablebalance: ["AccountBalance", 1],
  currentbalance: ["AccountBalance", 1],
  age: ["CustomerAge", 1],
  clientage: ["CustomerAge", 1],
  userage: ["CustomerAge", 1],
  accountholderage: ["CustomerAge", 1],
  duration: ["TransactionDuration", 1],
  durationseconds: ["TransactionDuration", 1],
  durationsecs: ["TransactionDuration", 1],
  durationsec: ["TransactionDuration", 1],
  sessionlength: ["TransactionDuration", 1],
  sessionduration: ["TransactionDuration", 1],
  sessionlengthseconds: ["TransactionDuration", 1],
  sessionlengthinseconds: ["TransactionDuration", 1],
  elapsedseconds: ["TransactionDuration", 1],
  durationminutes: ["TransactionDuration", 60],
  durationmins: ["TransactionDuration", 60],
  durationmin: ["TransactionDuration", 60],
  sessionlengthminutes: ["TransactionDuration", 60],
  sessionlengthinminutes: ["TransactionDuration", 60],
  sessionlengthmins: ["TransactionDuration", 60],
  elapsedminutes: ["TransactionDuration", 60],
  logins: ["LoginAttempts", 1],
  loginattemptcount: ["LoginAttempts", 1],
  numloginattempts: ["LoginAttempts", 1],
  signinattempts: ["LoginAttempts", 1],
  authattempts: ["LoginAttempts", 1],
  txntype: ["TransactionType", 1],
  trxtype: ["TransactionType", 1],
  debitcredit: ["TransactionType", 1],
  creditdebit: ["TransactionType", 1],
  transactionchannel: ["Channel", 1],
  accesschannel: ["Channel", 1],
  paymentchannel: ["Channel", 1],
  occupation: ["CustomerOccupation", 1],
  profession: ["CustomerOccupation", 1],
  customerjob: ["CustomerOccupation", 1],
  jobtitle: ["CustomerOccupation", 1],
  city: ["Location", 1],
  transactionlocation: ["Location", 1],
  customerlocation: ["Location", 1],
  transactioncity: ["Location", 1],
  accountnumber: ["AccountID", 1],
  accountno: ["AccountID", 1],
  acctid: ["AccountID", 1],
  acctno: ["AccountID", 1],
  device: ["DeviceID", 1],
  deviceidentifier: ["DeviceID", 1],
  terminalid: ["DeviceID", 1],
  transactiontimestamp: ["TransactionDate", 1],
  txndate: ["TransactionDate", 1],
  transactiondatetime: ["TransactionDate", 1],
  bookingdate: ["TransactionDate", 1],
  txnid: ["TransactionID", 1],
  transactionref: ["TransactionID", 1],
  transactionreference: ["TransactionID", 1],
  referenceid: ["TransactionID", 1],
};

const CANONICAL = new Set([...SCORING_COLUMNS, ...IGNORED_COLUMNS]);

/** Fold a header to its comparison form — mirrors _match_key in the backend. */
export function matchKey(name) {
  return String(name)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

/**
 * Rewrite a header so recognised synonyms use the names the model knows.
 *
 * Returns `{ resolved, matches }`. A canonical header always beats a synonym
 * for the same column, and the leftmost synonym beats any later one — the same
 * two rules resolve_columns() applies server-side.
 */
export function resolveColumns(columns) {
  const resolved = [...columns];
  const matches = [];
  const claimed = new Set(columns.filter((c) => CANONICAL.has(c)));

  columns.forEach((name, index) => {
    if (CANONICAL.has(name)) return;
    const entry = COLUMN_ALIASES[matchKey(name)];
    if (!entry) return;
    const [column, scale] = entry;
    if (claimed.has(column)) return;
    claimed.add(column);
    resolved[index] = column;
    matches.push({ from: name, to: column, scale });
  });

  return { resolved, matches };
}

/**
 * Sort a header into what will be used, what is missing, and what is ignored.
 *
 * Synonyms are resolved first, so a column the backend will read under another
 * name is reported as used rather than unrecognised. `matched` lists those
 * renames so the panel can show what was assumed.
 *
 * `ok` is the file-level gate, and it is now completeness: nothing is
 * substituted server-side for a column the file did not send, so a file with a
 * gap cannot be scored at all. Reporting that here means the user learns it
 * before the upload rather than from a 422 afterwards.
 */
export function classifyColumns(columns) {
  const { resolved, matches } = resolveColumns(columns);
  const seen = new Set(resolved);
  const crucial = CRUCIAL_COLUMNS.filter((c) => seen.has(c));
  const renamed = new Set(matches.map((m) => m.from));

  const missing = REQUIRED_COLUMNS.filter((c) => !seen.has(c));
  const synthesised = IDENTITY_COLUMNS.filter((c) => !seen.has(c));

  return {
    crucial,
    matched: matches,
    missing,
    synthesised,
    supplied: SCORING_COLUMNS.filter((c) => seen.has(c)),
    ignored: columns.filter(
      (c) => IGNORED_COLUMNS.includes(c) && !renamed.has(c),
    ),
    unknown: columns.filter(
      (c) =>
        !renamed.has(c) &&
        !SCORING_COLUMNS.includes(c) &&
        !IGNORED_COLUMNS.includes(c),
    ),
    ok: crucial.length > 0 && missing.length === 0,
    // Incomplete, but enough of a transaction file that the gap could be
    // filled if the operator asks. A file failing the crucial gate is not
    // offered the choice -- there would be nothing real left to score.
    fillable: crucial.length > 0 && missing.length > 0,
  };
}

/** The message shown when the crucial-column gate rejects a file. */
export function rejectionMessage(columns) {
  const { crucial, missing } = classifyColumns(columns);

  if (crucial.length === 0) {
    const found = columns.length ? columns.join(", ") : "no columns";
    return (
      `This file has none of the required columns. At least one of ` +
      `${CRUCIAL_COLUMNS.join(", ")} must be present. Found: ${found}.`
    );
  }

  // Name the gap rather than the rule: the user can act on a column list.
  return (
    `This file is missing ${missing.join(", ")}. Every scoring column must be ` +
    `supplied, under its own name or a recognised synonym — nothing is ` +
    `substituted from the training data.`
  );
}
