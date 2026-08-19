// By default, use same-origin paths so the dashboard works when served beside
// the API. For Cloud Run + Firebase Hosting, set VITE_API_BASE_URL to the
// deployed backend origin before building.
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

function buildUrl(path) {
  if (!path.startsWith("http")) {
    return `${API_BASE}${path}`;
  }
  return path;
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(buildUrl(path), options);
  } catch {
    // fetch() only rejects when the request never reached a server.
    throw new Error("Can't reach the API. Is the backend running?");
  }

  if (!response.ok) {
    // The API returns {"detail": "..."} for 4xx; surface that text rather than
    // a status code, because it names the actual problem with the transaction.
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* response had no JSON body; the status-based message stands */
    }
    throw new Error(detail);
  }

  return response.json();
}

const jsonPost = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getPresets = () => request("/demo/presets");

export const inject = (preset, overrides) =>
  request("/demo/inject", jsonPost({ preset, overrides }));

export const getHealth = () => request("/health");

// One chunk of an uploaded CSV. `startRow` is the line number of the first row
// within the original file, so a rejection in a later chunk still names a line
// the user can go and look at. `uploadId` ties the chunks of one file together
// so an account's history accumulates across a chunk boundary.
//
// `fillMissing` opts the file into having absent scoring columns filled from the
// training distribution. Off unless the operator ticked the box, because a
// filled row's verdict is partly a verdict about the training data.
export const scoreCsv = (columns, rows, startRow, uploadId, fillMissing = false) =>
  request(
    "/score-csv",
    jsonPost({
      columns,
      rows,
      start_row: startRow,
      upload_id: uploadId,
      fill_missing: fillMissing,
    }),
  );
