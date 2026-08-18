# The dashboard is built here rather than copied in from the build context:
# dashboard/dist is gitignored, so a fresh clone -- which is what Cloud Build
# works from -- has no bundle to copy. Building it here also keeps the served
# frontend in lockstep with the code the image was built from.
FROM node:20-slim AS dashboard-build

WORKDIR /dashboard

# Manifests first so the npm layer caches across dashboard source changes.
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci

COPY dashboard/ ./
RUN npm run build


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Requirements first so the dependency layer caches across code changes.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY ml ./ml
COPY backend ./backend
COPY original.csv ./original.csv

# Lands at /app/dashboard/dist, where backend/main.py's DASHBOARD_DIR resolves.
# The API serves it at / from the same origin, which is why the dashboard needs
# no base URL and no CORS configuration.
COPY --from=dashboard-build /dashboard/dist ./dashboard/dist

# Cloud Run injects PORT; uvicorn must bind it rather than a fixed port.
# --workers 1 is load-bearing, for the same reason the service is deployed
# with --max-instances=1: the ProfileStore is mutated in-process, so a
# second worker would develop a divergent view of recent activity. Do not
# raise it without moving that state somewhere shared.
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1
