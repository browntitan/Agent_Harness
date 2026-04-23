#!/usr/bin/env sh
set -eu

WAIT_FOR_DB="${WAIT_FOR_DB:-true}"
APP_AUTO_MIGRATE="${APP_AUTO_MIGRATE:-true}"
APP_AUTO_INIT_KB="${APP_AUTO_INIT_KB:-false}"
APP_AUTO_INDEX_SKILLS="${APP_AUTO_INDEX_SKILLS:-false}"
APP_AUTO_SYNC_KB="${APP_AUTO_SYNC_KB:-false}"
APP_AUTO_SYNC_COLLECTION_ID="${APP_AUTO_SYNC_COLLECTION_ID:-${DEFAULT_COLLECTION_ID:-default}}"
DB_WAIT_TIMEOUT_SECONDS="${DB_WAIT_TIMEOUT_SECONDS:-90}"

if [ "$WAIT_FOR_DB" = "true" ]; then
  python - <<'PY'
import os
import time
import psycopg2

dsn = os.getenv("PG_DSN", "")
timeout = int(os.getenv("DB_WAIT_TIMEOUT_SECONDS", "90"))

if not dsn:
    raise SystemExit("PG_DSN is not set.")

deadline = time.time() + timeout
last_err = None
while time.time() < deadline:
    try:
        conn = psycopg2.connect(dsn=dsn)
        conn.close()
        print("Database is reachable.")
        raise SystemExit(0)
    except Exception as e:
        last_err = e
        time.sleep(2)

raise SystemExit(f"Database not reachable within {timeout}s: {last_err}")
PY
fi

if [ "$APP_AUTO_MIGRATE" = "true" ]; then
  python run.py migrate
fi

if [ "$APP_AUTO_INIT_KB" = "true" ]; then
  python run.py init-kb
fi

if [ "$APP_AUTO_INDEX_SKILLS" = "true" ]; then
  python run.py index-skills
fi

if [ "$APP_AUTO_SYNC_KB" = "true" ]; then
  python run.py sync-kb --collection-id "$APP_AUTO_SYNC_COLLECTION_ID"
fi

exec "$@"
