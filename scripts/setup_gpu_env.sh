#!/bin/bash
# Setup script for GPU-accelerated audio analysis with essentia-tensorflow
# This script installs CUDA 11.8 runtime and cuDNN 8.x for GPU support
# Tested on Ubuntu 22.04 LTS (Jammy Jellyfish)

set -e  # Exit on error

echo "============================================================"
echo "GPU Environment Setup for essentia-tensorflow"
echo "============================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
    echo "Install NVIDIA drivers first, then re-run this script."
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo ""

# Detect Ubuntu version
echo "Detecting Ubuntu version..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $PRETTY_NAME"
    UBUNTU_VERSION=$VERSION_ID
else
    echo "ERROR: Cannot detect Ubuntu version"
    exit 1
fi

# Add NVIDIA CUDA repository for Ubuntu 22.04
echo ""
echo "Adding NVIDIA CUDA repository..."
if [ "$UBUNTU_VERSION" = "22.04" ]; then
    CUDA_REPO_PKG="cuda-keyring_1.1-1_all.deb"
    if [ ! -f "/usr/share/keyrings/cuda-archive-keyring.gpg" ]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        rm -f cuda-keyring_1.1-1_all.deb
        echo "✓ NVIDIA CUDA repository added"
    else
        echo "✓ NVIDIA CUDA repository already configured"
    fi
else
    echo "WARNING: This script is optimized for Ubuntu 22.04"
    echo "Current version: $UBUNTU_VERSION"
    echo "Proceeding anyway, but packages may not be available..."
fi

# Update package lists
echo ""
echo "Updating package lists..."
apt-get update

# Install CUDA 11.8 runtime
echo ""
echo "Installing CUDA 11.8 runtime libraries..."
echo "This includes: cudart, nvrtc, opencl, cublas, cufft, curand, cusolver, cusparse"
echo "Size: ~3-4 GB"
echo ""

# Install CUDA 11.8 meta-package (includes all runtime libraries)
apt-get install -y cuda-runtime-11-8

# Install cuDNN 8.x for CUDA 11.8
echo ""
echo "Installing cuDNN 8.x for CUDA 11.8..."
apt-get install -y libcudnn8 libcudnn8-dev

# Verify CUDA 11.8 installation
echo ""
echo "============================================================"
echo "Verifying Installation"
echo "============================================================"
echo ""

echo "Checking CUDA 11.8 installation..."
if [ -d "/usr/local/cuda-11.8" ]; then
    echo "✓ CUDA 11.8 installed at /usr/local/cuda-11.8"
    if [ -f "/usr/local/cuda-11.8/lib64/libcudart.so" ]; then
        CUDART_VERSION=$(ls /usr/local/cuda-11.8/lib64/libcudart.so.* | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+$' | head -1)
        echo "  libcudart version: $CUDART_VERSION"
    fi
else
    echo "✗ CUDA 11.8 not found at /usr/local/cuda-11.8"
    echo "  Checking for runtime libraries in system paths..."
fi

echo ""
echo "Checking cuDNN 8.x installation..."
if ldconfig -p | grep -q "libcudnn.so.8"; then
    echo "✓ cuDNN 8.x installed"
    CUDNN_PATH=$(ldconfig -p | grep "libcudnn.so.8" | awk '{print $NF}' | head -1)
    echo "  Location: $CUDNN_PATH"
else
    echo "✗ cuDNN 8.x not found in library cache"
    echo "  Run 'ldconfig' to update cache or check installation"
fi

echo ""
echo "Checking TensorFlow GPU support..."
if command -v python3 &> /dev/null; then
    python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')" 2>/dev/null || echo "TensorFlow not installed (install with: pip install tensorflow)"
else
    echo "Python3 not found. Install Python and TensorFlow to test GPU support."
fi

# Add to environment
echo ""
echo "============================================================"
echo "Configuring Environment"
echo "============================================================"
echo ""
CUDA_LIB_PATH="/usr/local/cuda-11.8/lib64"

# Check if CUDA libraries exist at the standard location
if [ ! -d "$CUDA_LIB_PATH" ]; then
    echo "Note: CUDA 11.8 libraries not found at $CUDA_LIB_PATH"
    echo "Libraries may be installed in system paths (/usr/lib/x86_64-linux-gnu/)"
    echo "This is normal for package installations and should work fine."
    CUDA_LIB_PATH="/usr/lib/x86_64-linux-gnu"
fi

# Add to current session
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH}"
echo "✓ LD_LIBRARY_PATH set for current session"

# Add to .bashrc for persistence (only if using /usr/local/cuda-11.8)
if [ "$CUDA_LIB_PATH" = "/usr/local/cuda-11.8/lib64" ]; then
    BASHRC="${HOME}/.bashrc"
    if ! grep -q "CUDA 11.8" "${BASHRC}"; then
        echo "" >> "${BASHRC}"
        echo "# CUDA 11.8 for essentia-tensorflow (GPU support)" >> "${BASHRC}"
        echo "export LD_LIBRARY_PATH=\"${CUDA_LIB_PATH}:\${LD_LIBRARY_PATH}\"" >> "${BASHRC}"
        echo "✓ Environment configured in ${BASHRC}"
    else
        echo "✓ Environment already configured in ${BASHRC}"
    fi
else
    echo "✓ System libraries will be used (no .bashrc modification needed)"
fi

# Install ffmpeg for audio processing
echo ""
echo "Installing ffmpeg for audio metadata extraction..."
apt-get install -y ffmpeg

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "✓ CUDA 11.8 runtime libraries installed"
echo "✓ cuDNN 8.x installed"
echo "✓ ffmpeg installed for audio processing"
echo ""
echo "Next Steps:"
echo "------------"
echo ""
echo "1. Install Python dependencies:"
echo "   python3 -m venv .venv"
echo "   source .venv/bin/activate"
echo "   pip install essentia-tensorflow tensorflow numpy tqdm"
echo ""
echo "2. Download Essentia models:"
echo "   The script will auto-download models on first run"
echo "   Or manually download from:"
echo "   https://essentia.upf.edu/models/"
echo ""
echo "3. Test GPU execution:"
echo "   python Training/scripts/genre_filter_test.py"
echo ""
echo "4. Expected GPU behavior:"
echo "   - TensorFlow will detect GPU on startup"
echo "   - GPU utilization during inference: 5-25%"
echo "   - Processing time: ~40s per audio file"
echo ""
echo "Troubleshooting:"
echo "----------------"
echo "If TensorFlow doesn't detect GPU:"
echo "  • Check NVIDIA driver: nvidia-smi"
echo "  • Verify CUDA libraries: ldconfig -p | grep cuda"
echo "  • Verify cuDNN: ldconfig -p | grep cudnn"
echo "  • Check Python TensorFlow: python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo ""
