# syntax=docker/dockerfile:1

ARG OPENWEBUI_VERSION=v0.8.6

FROM node:22-alpine3.20 AS build
ARG OPENWEBUI_VERSION

WORKDIR /src

RUN apk add --no-cache git

RUN git clone --depth 1 --branch ${OPENWEBUI_VERSION} https://github.com/open-webui/open-webui.git .

COPY deployment/openwebui/image/patches/openwebui-status-upserts.patch /tmp/openwebui-status-upserts.patch

RUN git apply --recount /tmp/openwebui-status-upserts.patch

RUN npm ci --force
RUN npm run build

FROM ghcr.io/open-webui/open-webui:${OPENWEBUI_VERSION}

COPY --from=build /src/build /app/build
