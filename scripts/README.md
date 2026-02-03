# GPU-Enabled Genre Filter Test - Complete Guide

## Quick Start

```bash
# 1. Navigate to scripts directory
cd /root/workspace/Training/scripts

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install cuDNN (if not in GPU container)
apt-get update && apt-get install -y libcudnn9-cuda-12

# 4. Run the test
python genre_filter_test.py
```

## What This Does

This script performs comprehensive audio analysis using Essentia's GPU-accelerated TensorFlow models. It includes:

### GPU Enforcement
- ✓ **Preflight check**: Fails fast if GPU not detected
- ✓ **Utilization monitoring**: Proves actual GPU compute (not just VRAM)
- ✓ **Pass/fail threshold**: Requires >10% GPU utilization during inference
- ✓ **Exit codes**: Clear success/failure signals

### Analysis Features
- Genre classification (multiple model sets)
- Mood/theme detection
- Instrument recognition  
- BPM/tempo analysis
- Audio quality metrics
- Musical structure analysis

## Exit Codes

| Code | Meaning | Action Required |
|------|---------|----------------|
| 0 | Success - GPU utilized | None |
| 1 | No GPU detected | Check CUDA/cuDNN installation |
| 2 | GPU not utilized | Check Essentia GPU support |

## Files in This Directory

### Main Script
- `genre_filter_test.py` - GPU-enforced audio analysis

### Configuration
- `requirements.txt` - Curated dependencies with documentation
- `requirements.freeze.txt` - Exact pinned environment (auto-generated)

### Documentation
- `README.md` - This file (quick start guide)
- `GPU_REQUIREMENTS.md` - Detailed GPU setup guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

### Testing
- `test_gpu_infrastructure.py` - GPU infrastructure test

## Troubleshooting

### Error: "No GPU detected by TensorFlow"

```bash
# 1. Check GPU is available
nvidia-smi

# 2. Install cuDNN
apt-get install -y libcudnn9-cuda-12

# 3. Verify CUDA libraries
ldconfig -p | grep cuda

# 4. Set library path (if needed)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Error: "GPU utilization never exceeded threshold"

This means GPU was detected but models ran on CPU. Possible causes:

**Cause 1: Essentia-TensorFlow CUDA version mismatch**
- essentia-tensorflow uses TensorFlow 2.x (expects CUDA 11.x)
- Your system has CUDA 12.x

**Solution**: Install CUDA 11.x compatibility libraries
```bash
apt-get install -y cuda-11-8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

**Cause 2: Essentia compiled without GPU support**
- Check if your Essentia build has GPU support

**Solution**: Use essentia-tensorflow package
```bash
pip uninstall essentia
pip install essentia-tensorflow
```

### Best Practice: Use GPU-Ready Container

For RunPod or Docker deployments:

```dockerfile
# Base image with CUDA 11.8 + cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

# Install Python packages
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt

# Copy script
COPY genre_filter_test.py /app/
WORKDIR /app

CMD ["python3", "genre_filter_test.py"]
```

Or use pre-built images:
- `runpod/pytorch:2.0.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- `tensorflow/tensorflow:2.14.0-gpu`

## Testing GPU Infrastructure

Before running the full analysis, test GPU setup:

```bash
python test_gpu_infrastructure.py
```

Expected output:
```
TEST 1: GPU Detection via TensorFlow
✓ PASS: GPU detected

TEST 2: NVML GPU Monitoring  
✓ PASS: NVML monitoring working

TEST 3: TensorFlow GPU Compute Test
✓ PASS: GPU compute verified (util > 10%)

ALL TESTS PASSED
```

## Requirements Explained

### Python Packages (requirements.txt)
- `essentia-tensorflow` - Audio analysis with GPU TensorFlow support
- `pynvml` - GPU utilization monitoring
- `numpy` - Numerical operations

### System Dependencies
- CUDA Toolkit 11.x or 12.x
- cuDNN 8.x (CUDA 11.x) or 9.x (CUDA 12.x)
- ffmpeg (for audio file handling)

## Output Format

On success:
```
================================================================================
GPU CHECK
--------------------------------------------------------------------------------
Python executable: /path/to/python
TensorFlow version: 2.20.0
TensorFlow GPUs detected: [PhysicalDevice(...)]
  ✓ Enabled memory growth
--------------------------------------------------------------------------------

... [analysis results] ...

================================================================================
GPU VERIFICATION SUMMARY
--------------------------------------------------------------------------------
  ✓ GPU detected: 1 device(s)
  ✓ GPU utilization: avg=45.2% max=87.3%
  ✓ Threshold met: True (>10.0%)
  ✓ Analysis completed successfully on GPU
================================================================================
```

## Environment Variables

Optional performance tuning:

```bash
# Disable oneDNN warnings
export TF_ENABLE_ONEDNN_OPTS=0

# Enable TensorFlow GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Set visible GPUs (if multi-GPU)
export CUDA_VISIBLE_DEVICES=0

# Add CUDA libraries to path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Performance Notes

- **First run**: Models download (~2GB), expect 1-2 minutes
- **Subsequent runs**: Use cached models, ~10-30 seconds per audio file
- **GPU utilization**: Typically 30-90% during inference
- **Memory**: ~4GB GPU VRAM typical usage

## Support

For issues:

1. **GPU not detected**: See `GPU_REQUIREMENTS.md` 2. **Technical details**: See `IMPLEMENTATION_SUMMARY.md`
3. **CUDA version issues**: Use compatible container image

## License

This script uses Essentia models which have their own licenses. Check:
- https://essentia.upf.edu/models.html

## Credits

- Essentia audio analysis framework
- MTG Jamendo dataset and models
- TensorFlow for GPU acceleration
