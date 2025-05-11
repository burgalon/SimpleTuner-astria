FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TINI_VERSION=v0.19.0 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_COMPILE_BYTECODE=1

# system packages + CLI tools
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates git git-lfs unzip aria2 \
        libgoogle-perftools-dev libgl1 libglib2.0-0 \
        python3 python3-pip python3-dev build-essential wget && \
    # AWS CLI v2 (pin + verify)
    cd /tmp && \
    curl -L -# -o awscliv2.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip && \
    echo "c1a9…  awscliv2.zip" | sha256sum -c - && \
    unzip -qq awscliv2.zip && ./aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip && \
    # handy disk-usage tool
    git clone --depth 1 https://codeberg.org/201984/dut.git /tmp/dut && \
    make -C /tmp/dut install && rm -rf /tmp/dut && \
    rm -rf /var/lib/apt/lists/*

# uv binary
COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/

WORKDIR /app

# ---------- dependency layer ----------
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --system --frozen --no-dev --no-install-project
# ---------- project code + extras ----------
COPY wheels/sageattention-*.whl ./wheels/
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --system --frozen --no-dev && \
    uv pip install --system wheels/*.whl

# tini
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

ENV LD_LIBRARY_PATH='/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib'
