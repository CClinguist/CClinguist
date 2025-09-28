# PyTorch CUDA Docker Environment

## 🚀 Features

- **CUDA 12.6.3 Support**: Based on NVIDIA CUDA 12.6.3 + cuDNN runtime
- **PyTorch Environment**: Pre-configured Conda environment with PyTorch and related dependencies
- **GPU Acceleration**: Support for NVIDIA GPU computing, suitable for deep learning training and inference
- **Lightweight Base**: Based on Ubuntu 20.04, optimized for image size

## 📋 System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Docker**: Version 20.10+
- **NVIDIA Driver**: Driver version supporting CUDA 12.6.3
- **NVIDIA Container Toolkit**: Properly installed and configured

## 📁 Project Structure

```
.   # Working directory
├── Dockerfile                  # Docker image build file
├── start_docker.sh             # Container startup script
├── docker.sh                   # Enable temporary docker permissions
├── torch_env.tar.gz            # Pre-configured Conda environment package
├── profile_generator_12ccs    
├── profile_generator_15ccs
├── self_upgrading_unknown_similar
└── README.md                   # Project documentation

# Data files can be downloaded from: [Google Drive](https://drive.google.com/drive/folders/1ITS85xJCkCdJ9o5Pi9c6OdUuwER1MLTx?usp=sharing)
```

## 🏗️ Building the Image

### 1. Prepare Environment Files

Ensure the `torch_env.tar.gz` file exists in the project root directory. This file should contain the pre-configured Conda environment. 

### 2. Build the Image

```bash
# Navigate to project directory
cd /path/to/your/project

# Build image (using default base image)
docker build -t my-torch-env .
```

### 3. Build Parameters

- `BASE_IMAGE`: Base image, defaults to `cf-dmirror.fnil.ac.cn/nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04`
- Supported base image formats:
  - `cf-dmirror.fnil.ac.cn/nvidia/cuda:<version>-cudnn-devel-ubuntu<version>`
  - `cf-dmirror.fnil.ac.cn/ubuntu:<version>`
- Other version query addresses:
  - `https://hub.docker.com/r/nvidia/cuda/tags?name=12.6`

## 🚀 Running the Container

### Startup Script

The project provides a `start_docker.sh` startup script:

```bash
# Grant execution permissions
chmod +x start_docker.sh

# Start the container
./start_docker.sh
```

### Startup Parameters

- `--gpus all`: Enable all GPU devices
- `--ipc=host`: Share host IPC namespace
- `--shm-size=16g`: Set shared memory size
- `-v /host/path:/workspace`: Mount host directory to container
- `-p 8888:8888`: Map Jupyter port
- `-p 6006:6006`: Map TensorBoard port

## 🔧 Environment Configuration

### Environment Variables

- `CUDA_HOME`: CUDA installation path (`/usr/local/cuda`)
- `LD_LIBRARY_PATH`: CUDA library path
- `CONDA_DIR`: Conda installation path (`/opt/conda`)
- `PATH`: Path containing Conda and CUDA

### Conda Environment

- Default environment: `torch`
- Auto-activation: Automatically activates the `torch` environment when container starts
- Environment management: Support package management through `conda` commands
