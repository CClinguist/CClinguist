#!/usr/bin/env bash
set -euo pipefail

IMAGE="my-torch-env"
CONTAINER_NAME="cclinguist-torch-env"
CONTAINER_WORKDIR="/workspace"

docker run --rm -it \
  --gpus all \
  --name "${CONTAINER_NAME}" \
  --hostname "${CONTAINER_NAME}" \
  --ipc=host \
  --shm-size=16g \
  -e TZ=Asia/Shanghai \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "./:${CONTAINER_WORKDIR}" \
  -p 8888:8888 -p 6006:6006 \
  --entrypoint /bin/bash \
  "${IMAGE}" -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate torch && exec bash'
