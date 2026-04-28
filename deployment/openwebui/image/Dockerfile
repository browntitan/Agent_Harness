# syntax=docker/dockerfile:1

ARG OPENWEBUI_VERSION=v0.8.6
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0"
ARG CUSTOM_CA_CERT_B64=""

FROM node:22-alpine3.20 AS build
ARG OPENWEBUI_VERSION
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

WORKDIR /src

RUN apk add --no-cache git ca-certificates \
    && if [ -n "$CUSTOM_CA_CERT_B64" ]; then \
      mkdir -p /usr/local/share/ca-certificates \
      && printf '%s' "$CUSTOM_CA_CERT_B64" | base64 -d > "$NODE_EXTRA_CA_CERTS"; \
    fi \
    && update-ca-certificates \
    && if [ -n "$HTTP_PROXY" ]; then npm config set proxy "$HTTP_PROXY"; fi \
    && if [ -n "$HTTPS_PROXY" ]; then npm config set https-proxy "$HTTPS_PROXY"; fi \
    && if [ -s "$NODE_EXTRA_CA_CERTS" ]; then npm config set cafile "$NODE_EXTRA_CA_CERTS"; fi \
    && if [ -s /etc/ssl/certs/ca-certificates.crt ]; then git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt; fi

RUN git clone --depth 1 --branch ${OPENWEBUI_VERSION} https://github.com/open-webui/open-webui.git .

COPY deployment/openwebui/image/patches/openwebui-status-upserts.patch /tmp/openwebui-status-upserts.patch

RUN git apply --recount /tmp/openwebui-status-upserts.patch

RUN npm ci --force
RUN npm run build

FROM ghcr.io/open-webui/open-webui:${OPENWEBUI_VERSION}

COPY --from=build /src/build /app/build
