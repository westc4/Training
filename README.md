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
- **Training Quality Checks**: Automated codec quality verification, stability monitoring, and sample generation
- **Curriculum Learning**: Progressive training from short to long sequences

## Repository Structure

```
Training/
├── notebooks/
│   ├── dora_train.py           # Compression model training with preflight checks
│   ├── musicgen_train.py       # MusicGen LM training script
│   ├── musicgen_generate.py    # Sample generation from trained models
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
├── training_checks/            # Training validation and monitoring tools
│   ├── preflight.py            # Pre-training validation checks
│   ├── codec_quality_check.py  # Codec reconstruction quality verification
│   ├── validate_conditioning.py # Conditioning pipeline validation
│   ├── stability_monitor.py    # INF/NaN gradient monitoring
│   ├── curriculum_train.py     # Progressive sequence length training
│   ├── fixed_seed_sampler.py   # Reproducible sample generation for A/B testing
│   └── curriculum_configs/     # YAML configs for curriculum stages
│       ├── short_10s.yaml
│       └── medium_20s.yaml
├── model_config/
│   ├── fma_small_mini.yaml     # Config for small dataset
│   └── fma_large.yaml          # Config for large dataset
├── outputs/
│   ├── Checkpoints Small/      # Training checkpoints
│   ├── codec_quality_check/    # Codec verification outputs
│   ├── samples/                # Generated sample comparisons
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
  - NVIDIA GPU with ≥ 24GB VRAM (A40)
  - Preffered GPU: 4X H200 SXM
  - Base Image: pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
  - Python 3.11
  - At least 2TB free disk space

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

## Training Quality Checks

The `training_checks/` folder contains tools for monitoring and validating training quality.

### Codec Quality Check

Verifies codec (EnCodec) reconstruction quality by encoding and decoding audio samples.

```bash
# Run standalone
python training_checks/codec_quality_check.py --num-samples 10

# With specific checkpoint
python training_checks/codec_quality_check.py --checkpoint /path/to/checkpoint.th
```

**Integrated mode**: Configure in `dora_train.py`:
```python
CODEC_CHECK_EVERY = 10     # Run every 10 epochs (0=disabled)
CODEC_CHECK_SAMPLES = 5    # Number of samples per check
```

**Output includes**:
- SI-SNR metrics (standard and AudioCraft conventions)
- L1, MSE, mel spectrogram losses
- Original/reconstructed wav pairs for listening tests

### Conditioning Validation

Validates that text conditioning is properly configured for text-to-music training.

```bash
python training_checks/validate_conditioning.py --num-samples 100
```

**Checks**:
- Percentage of samples with text descriptions
- Average description token length
- Conditioner configuration (warns if unconditional)

### Stability Monitor

Tracks gradient instability during training (can be imported into training scripts).

```python
from training_checks.stability_monitor import StabilityMonitor, StabilityConfig

monitor = StabilityMonitor(StabilityConfig(
    max_nan_ratio_per_epoch=0.1,      # Fail if >10% NaN losses
    max_inf_grad_ratio_per_epoch=0.3, # Warn if >30% INF gradients
    enable_adaptive_lr=True,          # Auto-reduce LR on instability
))
```

### Curriculum Training

Progressive training from short to long sequences for better convergence.

```bash
# Auto-detect current stage and continue
python training_checks/curriculum_train.py --auto --dry-run

# Run specific stage
python training_checks/curriculum_train.py --stage short_10s
python training_checks/curriculum_train.py --stage medium_20s --continue-from /path/to/checkpoint.th
```

**Stages**:
| Stage | Segment | Batch Size | Epochs | Purpose |
|-------|---------|------------|--------|--------|
| `short_10s` | 10s | 128 | 50 | Fast initial learning |
| `medium_20s` | 20s | 64 | 50 | Structure learning |
| `full_30s` | 30s | 32 | 100 | Long-form coherence |

### Fixed-Seed Sample Generation

Generates samples with fixed seeds for reproducible A/B comparison across epochs.

```bash
# Generate samples for current checkpoint
python training_checks/fixed_seed_sampler.py --epoch 50

# List available checkpoints
python training_checks/fixed_seed_sampler.py --list-epochs

# Generate HTML comparison index only
python training_checks/fixed_seed_sampler.py --generate-index
```

**Output**: HTML index at `outputs/samples/index.html` with side-by-side audio players.

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

**Status**: Active development | Last updated: February 2026
