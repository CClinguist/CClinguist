# https://hub.docker.com/r/nvidia/cuda/tags?name=12.6 
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates gnupg2 software-properties-common \
    build-essential git bzip2 vim graphviz \
    && rm -rf /var/lib/apt/lists/*


ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

RUN $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV PATH=$CONDA_DIR/bin:$PATH

COPY torch_env.tar.gz /tmp/torch_env.tar.gz

RUN mkdir -p $CONDA_DIR/envs/torch && \
    tar -xzf /tmp/torch_env.tar.gz -C $CONDA_DIR/envs/torch && \
    rm /tmp/torch_env.tar.gz && \
    $CONDA_DIR/envs/torch/bin/conda-unpack

WORKDIR /workspace

SHELL ["conda", "run", "-n", "torch", "/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]
