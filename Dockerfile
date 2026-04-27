FROM node:20-bookworm-slim AS control-panel-builder

WORKDIR /app/control_panel

COPY control_panel/package*.json ./
RUN npm ci

COPY control_panel /app/control_panel
RUN npm run build


FROM python:3.12-slim AS app-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && python -c "import graphrag" \
    && command -v graphrag >/dev/null

COPY . /app
COPY --from=control-panel-builder /app/control_panel/dist /app/control_panel/dist

RUN chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/bin/sh", "/app/docker/entrypoint.sh"]
CMD ["sleep", "infinity"]


FROM app-runtime AS app-docling

COPY requirements-docling.txt /app/requirements-docling.txt
RUN pip install -r /app/requirements-docling.txt


FROM app-runtime AS app-base
