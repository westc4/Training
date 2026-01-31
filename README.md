# AudioCraft Music Generation Training Pipeline

A complete end-to-end pipeline for training music generation models using Meta's AudioCraft framework on the Free Music Archive (FMA) and MTG-Jamendo datasets.

## Overview

This repository provides Jupyter notebooks and scripts for training neural audio compression models and music generation models (MusicGen) on RunPod or similar GPU environments. The pipeline handles everything from dataset preparation to model training with support for multiple dataset sizes.

## Features

- **Multi-dataset Support**: FMA-small, FMA-large, MTG-Jamendo
- **End-to-end Pipeline**: Data download → preprocessing → training → evaluation
- **Reproducible Workflows**: Idempotent Jupyter notebooks that can be run top-to-bottom
- **Flexible Configuration**: Hydra-based configs for easy experimentation
- **GPU Optimized**: Designed for high-VRAM GPUs (96GB+)
- **Dora Integration**: Experiment tracking and management with Dora

## Repository Structure

```
Training/
├── notebooks/
│   └── fma/
│       ├── small/              # FMA-small dataset workflows
│       │   ├── 01_fma_small_mini_setup.ipynb
│       │   ├── 01b_fma_small_mini_downloader.ipynb
│       │   ├── 01c_mtg_jamendo_downloader.ipynb
│       │   ├── 01d_all_files.ipynb
│       │   └── 02_audiocraft_train_compression_debug.ipynb
│       └── large/              # FMA-large dataset workflows
│           ├── 01_fma_large_setup.ipynb
│           ├── 01b_fma_large_downloader.ipynb
│           └── 02_audiocraft_train_compression_debug.ipynb
├── model_config/
│   ├── fma_small_mini.yaml     # Config for small dataset
│   └── fma_large.yaml          # Config for large dataset
├── outputs/
│   ├── Checkpoints Small/      # Training checkpoints
│   └── musicgen_uncond_debug/  # MusicGen outputs
├── scripts/
│   ├── download_script.sh      # Dataset download automation
│   └── startup_script.sh       # Environment setup
├── 03_audiocraft_train_musicgen_uncond_debug.ipynb  # MusicGen training
├── train.ipynb                 # Legacy training notebook
└── runpod_ssh_sync.py          # SSH config sync utility
```

## Quick Start

### Prerequisites

- RunPod instance (or similar) with:
  - NVIDIA GPU with ≥ 24GB VRAM (96GB recommended)
  - CUDA support
  - Python 3.11
  - At least 100GB free disk space

### Installation & Setup

1. **Run Setup Notebook**:
   ```
   notebooks/fma/small/01_fma_small_mini_setup.ipynb
   ```
   This installs system dependencies, Python packages, clones AudioCraft, and sets up the environment.

2. **Download Dataset**:
   ```
   notebooks/fma/small/01b_fma_small_mini_downloader.ipynb
   ```
   Downloads and preprocesses audio data (10s mono segments at 32kHz).

3. **Train Compression Model**:
   ```
   notebooks/fma/small/02_audiocraft_train_compression_debug.ipynb
   ```
   Trains the neural audio compression/codebook model.

4. **Train Music Generator**:
   ```
   03_audiocraft_train_musicgen_uncond_debug.ipynb
   ```
   Trains the MusicGen language model for unconditional music generation.

## Workflows

### FMA-Small Mini Pipeline

For quick experimentation with a small dataset:

1. `01_fma_small_mini_setup.ipynb` - Environment setup
2. `01b_fma_small_mini_downloader.ipynb` - Download ~100 tracks
3. `02_audiocraft_train_compression_debug.ipynb` - Train compression
4. `03_audiocraft_train_musicgen_uncond_debug.ipynb` - Train generator

### FMA-Large Pipeline

For full-scale training:

1. `notebooks/fma/large/01_fma_large_setup.ipynb`
2. `notebooks/fma/large/01b_fma_large_downloader.ipynb`
3. `notebooks/fma/large/02_audiocraft_train_compression_debug.ipynb`
4. `03_audiocraft_train_musicgen_uncond_debug.ipynb`

### MTG-Jamendo Integration

Additional dataset for diversity:

- `01c_mtg_jamendo_downloader.ipynb` - Download MTG-Jamendo tracks
- `01d_all_files.ipynb` - Combine multiple datasets

## Dataset Processing

The pipeline automatically:

1. Downloads audio files from FMA or MTG-Jamendo
2. Converts to mono WAV at 32kHz sample rate
3. Segments into 10-second chunks
4. Creates train/valid splits (90/10 ratio by default)
5. Generates AudioCraft-compatible `data.jsonl` with:
   - File paths
   - Duration metadata
   - Sample rate information
6. Creates symlinked `egs/` directory structure for efficient storage

## Training Configuration

Edit configuration files in `model_config/`:

```yaml
# model_config/fma_small_mini.yaml
datasource:
  max_sample_rate: 32000
  max_channels: 1
  train: /root/workspace/data/fma_small_mini/egs/train
  valid: /root/workspace/data/fma_small_mini/egs/valid
  evaluate: /root/workspace/data/fma_small_mini/egs/valid
  generate: /root/workspace/data/fma_small_mini/egs/valid
```

## Training Outputs

Results are saved to:

- **Dora experiments**: `/root/workspace/experiments/audiocraft/xps/<xp_id>/`
- **Checkpoints**: `outputs/Checkpoints Small/<xp_id>/`
- **Logs**: `solver.log.*` files with training progress
- **Configs**: `.hydra/` directory with full experiment configuration

## Key Parameters

Default hyperparameters (configurable in notebooks):

- **Segment duration**: 10 seconds
- **Sample rate**: 32,000 Hz
- **Channels**: 1 (mono)
- **Batch size**: 4
- **Training epochs**: 1 (debug mode)
- **Workers**: Auto-detected (CPU cores / 4)

## Utilities

- **SSH Sync**: `runpod_ssh_sync.py` - Syncs SSH config for RunPod
- **Download Script**: `scripts/download_script.sh` - Batch dataset downloads
- **Startup Script**: `scripts/startup_script.sh` - Pod initialization

## Environment Details

The setup process installs:

- **System**: ffmpeg, pkg-config, libav* libraries
- **Python**: PyTorch (CUDA 12.1), dora-search, transformers
- **AudioCraft**: Latest from Facebook Research GitHub
- **Optional**: xformers (may be uninstalled if incompatible)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `BATCH_SIZE` in training notebooks
2. **CUDA not available**: Check PyTorch installation with `torch.cuda.is_available()`
3. **ffmpeg errors**: Reinstall with `apt-get install -y ffmpeg`
4. **xformers warnings**: Safe to ignore or uninstall with `pip uninstall xformers`

### Validation Checks

Each notebook includes sanity checks:
- File counts at each stage
- Sample duration validation (should be ~10s)
- First lines of `data.jsonl` inspection
- Configuration file verification

## Development Notes

- All notebooks are designed to be **idempotent** - safe to re-run
- Uses symlinks instead of file copies to save disk space
- Working directory: `/root/workspace` (RunPod standard)
- Experiments tracked via Dora in `/root/workspace/experiments/audiocraft/`

## References

- [AudioCraft GitHub](https://github.com/facebookresearch/audiocraft)
- [Free Music Archive](https://freemusicarchive.org/)
- [MTG-Jamendo Dataset](https://mtg.github.io/mtg-jamendo-dataset/)
- [Dora Documentation](https://github.com/facebookresearch/dora)

## License

This training pipeline is for research and educational purposes. Please respect the licenses of:
- AudioCraft (MIT License)
- FMA dataset (CC BY 4.0)
- MTG-Jamendo dataset (CC BY-NC-SA 4.0)

## Citation

If you use this training pipeline, please cite the original AudioCraft paper:

```bibtex
@article{copet2023simple,
  title={Simple and Controllable Music Generation},
  author={Copet, Jade and Kreuk, Felix and Gat, Itai and Remez, Tal and Kant, David and Synnaeve, Gabriel and Adi, Yossi and Défossez, Alexandre},
  journal={arXiv preprint arXiv:2306.05284},
  year={2023}
}
```

---

**Status**: Active development | Last updated: January 2026
