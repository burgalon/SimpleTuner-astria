FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TINI_VERSION=v0.19.0 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_SYNC=1 \
    GIT_LFS_SKIP_SMUDGE=1

# system packages + CLI tools
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates git git-lfs unzip aria2 \
        libgoogle-perftools-dev libgl1 libglib2.0-0 \
        python3 python3-pip python3-dev build-essential wget && \
    # ── AWS CLI v2 (download + checksum) ────────────────────────────────
    cd /tmp && \
    curl -L -# -o awscliv2.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip && \
    unzip -qq awscliv2.zip && ./aws/install && \
    rm -rf /tmp/aws awscliv2.zip && \
    # ── handy disk-usage tool ───────────────────────────────────────────
    git clone --depth 1 https://codeberg.org/201984/dut.git /tmp/dut && \
    make -C /tmp/dut install && rm -rf /tmp/dut && \
    git lfs install --system && \
    # cleanup caches to reduce image size
    rm -rf /root/.cache/pip && \
    rm -rf /root/.cache/pypoetry && \
    rm -rf /var/cache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/cache/apt/archives/partial

# uv binary
COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

WORKDIR /app

# ---------- dependency layer ----------
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=wheels,target=wheels \
    uv venv && \
    uv sync --frozen --no-dev --no-install-project --find-links=wheels

# tini
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

ENV LD_LIBRARY_PATH='/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib'

COPY . .
