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
 * The file is parsed and checked in the browser before anything is sent, so a
 * file missing every crucial column is refused without a round trip, and the
 * operator can see what will be filled in before committing to the run.
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

  const inputRef = useRef(null);
  const cancelRef = useRef(false);

  function reset() {
    setParsed(null);
    setReport(null);
    setError(null);
    setProgress(null);
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

    const classified = classifyColumns(result.columns);
    if (!classified.ok) {
      setError(rejectionMessage(result.columns));
      return;
    }

    setParsed(result);
    setReport(classified);
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
        const body = await scoreCsv(parsed.columns, chunk, start + 2, uploadId);

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
          or drop one here. Any subset of <span className="num">original.csv</span>
          's columns works — anything missing is filled from the training data.
        </p>
      </div>

      {error && <div className="error">{error}</div>}

      {report && parsed && (
        <div className="col-report">
          <div className="col-report-head">
            <span className="fname">{file?.name}</span>
            <span className="num">{parsed.rows.length.toLocaleString()} rows</span>
          </div>

          <ColumnGroup
            label={`Used from the file (${report.supplied.length})`}
            columns={report.supplied}
            tone="ok"
          />
          {report.filled.length > 0 && (
            <ColumnGroup
              label={`Filled from training data (${report.filled.length})`}
              columns={report.filled}
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

      {parsed && (
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
