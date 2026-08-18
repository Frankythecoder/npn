import { useRef, useState } from "react";
import { scoreCsv } from "../api.js";
import {
  CHUNK_SIZE,
  classifyColumns,
  parseCsv,
  rejectionMessage,
} from "../csv.js";

/**
 * Upload a CSV of transactions instead of filling fields one at a time.
 *
 * The file is parsed and checked in the browser before anything is sent, so an
 * incomplete file is refused without a round trip. The column breakdown is shown
 * either way — a rejected file is exactly when the operator most needs to see
 * which columns are missing, so the report renders alongside the error and only
 * the Score button is withheld.
 *
 * Scoring is chunked. The service runs one uvicorn worker with one instance on
 * purpose — the profile store is mutated in process — so a single request
 * carrying 2,500 rows would hold that worker for the best part of a minute and
 * stall the feed's own polling. Chunks keep each request short and give the
 * operator a progress bar instead of a spinner.
 */
export default function CsvPanel({ onBatch, onBusyChange }) {
  const [file, setFile] = useState(null);
  const [parsed, setParsed] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(null);
  const [dragging, setDragging] = useState(false);
  // Opting an incomplete file into having its gap filled. Reset by reset(), so
  // it can never carry over from one file to the next.
  const [fillMissing, setFillMissing] = useState(false);

  const inputRef = useRef(null);
  const cancelRef = useRef(false);

  function reset() {
    setParsed(null);
    setReport(null);
    setError(null);
    setProgress(null);
    setFillMissing(false);
  }

  async function accept(chosen) {
    reset();
    setFile(chosen);

    if (!chosen) return;

    let text;
    try {
      text = await chosen.text();
    } catch {
      setError("That file could not be read.");
      return;
    }

    const result = parseCsv(text);
    if (result.error) {
      setError(result.error);
      return;
    }

    // Reported whether or not it passes: the breakdown is the explanation.
    const classified = classifyColumns(result.columns);
    setReport(classified);

    if (!classified.ok && !classified.fillable) {
      setError(rejectionMessage(result.columns));
      // parsed stays null, so no Score button appears for a file that cannot
      // be scored at all.
      return;
    }

    // A fillable file is parsed but not yet scoreable: the Score button stays
    // hidden until the operator ticks the box, so filling is never the path of
    // least resistance.
    setParsed(result);
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    accept(event.dataTransfer.files?.[0] ?? null);
  }

  async function run() {
    if (!parsed || progress) return;

    cancelRef.current = false;
    onBusyChange?.(true);

    // One id per run, so the chunks of this file share a profile store and a
    // re-run of the same file starts from a clean one.
    const uploadId = `u${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;

    const results = [];
    const rejected = [];
    const total = parsed.rows.length;
    setProgress({ done: 0, total });

    try {
      for (let start = 0; start < total; start += CHUNK_SIZE) {
        if (cancelRef.current) break;

        const chunk = parsed.rows.slice(start, start + CHUNK_SIZE);
        // +2: the header occupies line 1, so the first data row is line 2.
        const body = await scoreCsv(
          parsed.columns,
          chunk,
          start + 2,
          uploadId,
          fillMissing,
        );

        results.push(...body.results);
        rejected.push(...body.rejected);
        setProgress({ done: Math.min(start + CHUNK_SIZE, total), total });
      }
    } catch (err) {
      setError(err.message);
      setProgress(null);
      onBusyChange?.(false);
      return;
    }

    setProgress(null);
    onBusyChange?.(false);
    onBatch({
      name: file?.name ?? "upload.csv",
      results,
      rejected,
      cancelled: cancelRef.current,
      scanned: results.length + rejected.length,
      // Carried so the results view can say the batch was filled. A verdict on
      // a filled row must not be read as a verdict on the row alone.
      filled: fillMissing ? report.missing : [],
    });
  }

  const busy = Boolean(progress);
  const pct = progress ? Math.round((progress.done / progress.total) * 100) : 0;

  return (
    <div>
      <div
        className="drop"
        data-dragging={dragging}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          hidden
          onChange={(e) => accept(e.target.files?.[0] ?? null)}
        />
        <button
          type="button"
          className="drop-btn"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
        >
          Choose a CSV file
        </button>
        <p className="why" style={{ marginTop: 10, marginBottom: 0 }}>
          or drop one here. Every scoring column must be present, under its own
          name or a recognised synonym — nothing is substituted from the
          training data.
        </p>
      </div>

      {error && <div className="error">{error}</div>}

      {report && (
        <div className="col-report">
          <div className="col-report-head">
            <span className="fname">{file?.name}</span>
            {parsed && (
              <span className="num">
                {parsed.rows.length.toLocaleString()} rows
              </span>
            )}
          </div>

          <ColumnGroup
            label={`Used from the file (${report.supplied.length})`}
            columns={report.supplied}
            tone="ok"
          />
          {report.matched.length > 0 && (
            <ColumnGroup
              label={`Matched by meaning (${report.matched.length})`}
              columns={report.matched.map(
                (m) =>
                  `${m.from} → ${m.to}${m.scale === 1 ? "" : ` ×${m.scale}`}`,
              )}
              tone="ok"
            />
          )}
          {report.missing.length > 0 && (
            <ColumnGroup
              label={`Missing — required (${report.missing.length})`}
              columns={report.missing}
              tone="missing"
            />
          )}
          {report.fillable && (
            <label className="fill-offer">
              <input
                type="checkbox"
                checked={fillMissing}
                onChange={(e) => setFillMissing(e.target.checked)}
              />
              <span>
                Score anyway, filling{" "}
                <span className="num">{report.missing.length}</span>{" "}
                {report.missing.length === 1 ? "column" : "columns"} from the
                training data.{" "}
                <span className="fill-cost">
                  Measured on the training set, substituting a column group moves
                  4–10% of verdicts — those rows are judged partly on training
                  values, not only on what the file sent.
                </span>
              </span>
            </label>
          )}
          {report.synthesised.length > 0 && (
            <ColumnGroup
              label={`Scored as a new account (${report.synthesised.length})`}
              columns={report.synthesised}
              tone="fill"
            />
          )}
          {report.ignored.length > 0 && (
            <ColumnGroup
              label={`Ignored (${report.ignored.length})`}
              columns={report.ignored}
              tone="mute"
            />
          )}
          {report.unknown.length > 0 && (
            <ColumnGroup
              label={`Unrecognised (${report.unknown.length})`}
              columns={report.unknown}
              tone="mute"
            />
          )}
        </div>
      )}

      {progress && (
        <div className="progress-block">
          <div className="progress-line">
            <span className="micro">Scoring</span>
            <span className="num">
              {progress.done.toLocaleString()} / {progress.total.toLocaleString()}
            </span>
          </div>
          <div className="progress-track">
            <span className="progress-bar" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}

      {parsed && (report.ok || fillMissing) && (
        <button
          type="button"
          className="inject-btn"
          style={{ marginTop: 16 }}
          disabled={busy}
          onClick={run}
        >
          {busy
            ? "Scoring…"
            : `Score ${parsed.rows.length.toLocaleString()} transaction${
                parsed.rows.length === 1 ? "" : "s"
              }`}
        </button>
      )}

      {busy && (
        <button
          type="button"
          className="linkish"
          style={{ marginTop: 10 }}
          onClick={() => {
            cancelRef.current = true;
          }}
        >
          Stop after this batch
        </button>
      )}
    </div>
  );
}

function ColumnGroup({ label, columns, tone }) {
  return (
    <div className="col-group">
      <div className="micro" style={{ marginBottom: 6 }}>
        {label}
      </div>
      <div className="chips">
        {columns.map((column) => (
          <span className="chip" data-tone={tone} key={column}>
            {column}
          </span>
        ))}
      </div>
    </div>
  );
}
