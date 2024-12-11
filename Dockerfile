FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    # Poetry's configuration:
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'
#    LD_PRELOAD=libtcmalloc.so

# Install SimpleTuner
WORKDIR /app
COPY poetry.lock pyproject.toml /app/

RUN apt-get update -y && \
	apt-get install -y --no-install-recommends aria2 libgoogle-perftools-dev libgl1 libglib2.0-0 wget curl awscli git git-lfs python3 python3-pip build-essential python3-dev && \
  apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y && \
  python3 -m pip install pip --upgrade && \
  # https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
  curl -sSL https://install.python-poetry.org | python3  && \
  # used to check disk usage quickly
  git clone https://codeberg.org/201984/dut.git /root/dut && cd /root/dut/ && make install && \
  cd /app && \
  poetry install --no-root --no-interaction --no-ansi --without dev && \
  # cleanup caches to reduce image size
  rm -rf /root/.cache/pip && \
  rm -rf /root/.cache/pypoetry && \
  rm -rf /var/cache && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  mkdir -p /var/cache/apt/archives/partial

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

ENV LD_LIBRARY_PATH='/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib'
COPY . /app
