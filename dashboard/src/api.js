// Every call is same-origin. In development Vite proxies these paths to the
// API on :8000; in a deployment where the dashboard is served alongside the
// API, they resolve directly. No base URL to configure either way.

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
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

export const getRecent = (limit = 50) =>
  request(`/transactions/recent?limit=${limit}`);

export const inject = (preset, overrides) =>
  request("/demo/inject", jsonPost({ preset, overrides }));

export const getHealth = () => request("/health");

// One chunk of an uploaded CSV. `startRow` is the line number of the first row
// within the original file, so a rejection in a later chunk still names a line
// the user can go and look at. `uploadId` ties the chunks of one file together
// so an account's history accumulates across a chunk boundary.
export const scoreCsv = (columns, rows, startRow, uploadId) =>
  request(
    "/score-csv",
    jsonPost({ columns, rows, start_row: startRow, upload_id: uploadId }),
  );
