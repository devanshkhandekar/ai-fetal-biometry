ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       python3-venv \
       ca-certificates \
       libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/worker

COPY requirements-worker-host.txt /tmp/requirements-worker-host.txt

RUN /opt/worker/bin/python -m pip install \
      --upgrade pip setuptools wheel \
    && /opt/worker/bin/python -m pip install \
      --no-cache-dir \
      -r /tmp/requirements-worker-host.txt \
    && /opt/worker/bin/python -m pip install \
      --no-cache-dir \
      "pydantic-settings>=2,<3"

WORKDIR /stack/services/platform

ENTRYPOINT ["/opt/worker/bin/python"]
CMD ["-m", "app.worker_inference"]
