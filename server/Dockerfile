# painite-server on an NVIDIA NGC PyTorch base (multi-arch; arm64 for DGX Spark / GB10).
# Build:  docker build -t painite-server server/
# Run:    docker run --gpus all -p 8443:8443 -v /var/lib/painite:/var/lib/painite painite-server
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends libvips42 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/painite
COPY pyproject.toml ./
COPY painite_server ./painite_server
# The base image provides torch, torchvision and numpy (pinned by its pip
# constraints); install only what is missing, and fail the build if CUDA torch
# was replaced by a PyPI wheel.
RUN pip install --no-cache-dir '.[models]' \
 && python -c "import torch, spandrel; assert torch.version.cuda, 'CPU-only torch'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

ENV PAINITE_DATA_DIR=/var/lib/painite
VOLUME /var/lib/painite
EXPOSE 8443
ENTRYPOINT ["painite-server"]
CMD ["serve", "--port", "8443"]
