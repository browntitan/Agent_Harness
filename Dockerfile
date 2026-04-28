ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0"
ARG CUSTOM_CA_CERT_B64=""

FROM node:20-bookworm-slim AS control-panel-builder
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG CUSTOM_CA_CERT_B64

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/NG-Certificate-Chain.crt

WORKDIR /app/control_panel

RUN if [ -n "$CUSTOM_CA_CERT_B64" ]; then \
      mkdir -p /usr/local/share/ca-certificates \
      && printf '%s' "$CUSTOM_CA_CERT_B64" | base64 -d > "$NODE_EXTRA_CA_CERTS" \
      && if command -v update-ca-certificates >/dev/null 2>&1; then update-ca-certificates; fi; \
    fi \
    && if [ -n "$HTTP_PROXY" ]; then npm config set proxy "$HTTP_PROXY"; fi \
    && if [ -n "$HTTPS_PROXY" ]; then npm config set https-proxy "$HTTPS_PROXY"; fi \
    && if [ -s "$NODE_EXTRA_CA_CERTS" ]; then npm config set cafile "$NODE_EXTRA_CA_CERTS"; fi

COPY control_panel/package*.json ./
RUN npm ci

COPY control_panel /app/control_panel
RUN npm run build


FROM python:3.12-slim AS app-runtime
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG CUSTOM_CA_CERT_B64

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

RUN if [ -n "$CUSTOM_CA_CERT_B64" ]; then \
      mkdir -p /usr/local/share/ca-certificates \
      && printf '%s' "$CUSTOM_CA_CERT_B64" | base64 -d > /usr/local/share/ca-certificates/NG-Certificate-Chain.crt \
      && if command -v update-ca-certificates >/dev/null 2>&1; then update-ca-certificates; fi; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    && update-ca-certificates \
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
