FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      pandas \
      numpy \
      openpyxl \
      xlrd \
      matplotlib \
      pillow

WORKDIR /workspace

CMD ["python", "--version"]
