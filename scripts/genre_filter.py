#!/usr/bin/env python3
"""
Comprehensive audio analysis using Essentia.
Auto-installs dependencies and downloads models as needed.
"""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set environment variables for optimal GPU usage
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging (0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR)

# Force TensorFlow to use GPU for all operations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'
os.environ['TF_USE_CUDNN'] = '1'

# Try to force TensorFlow operations to GPU
# Soft placement allows fallback to CPU if GPU operation not available
# We want to allow this for Essentia compatibility
os.environ['TF_ENABLE_SOFT_PLACEMENT'] = '1'  # Allow CPU fallback for unsupported ops

# Additional TensorFlow GPU configurations
os.environ['TF_GPU_HOST_MEM_LIMIT_IN_MB'] = '8000'

# For TensorFlow 2.x, try to force XLA compilation which can help GPU utilization
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

# Add CUDA library paths to LD_LIBRARY_PATH
cuda_paths = [
    '/usr/local/cuda-11.8/targets/x86_64-linux/lib',
    '/usr/local/cuda-12.4/targets/x86_64-linux/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib',
]

# Get existing LD_LIBRARY_PATH
existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')

# Add CUDA paths that exist
valid_cuda_paths = [p for p in cuda_paths if Path(p).exists()]
if valid_cuda_paths:
    new_ld_path = ':'.join(valid_cuda_paths)
    required_cuda_path = cuda_paths[0]  # The main CUDA 11.8 path

    # Check if we need to restart with updated LD_LIBRARY_PATH
    # Only restart if the required path is not already in LD_LIBRARY_PATH
    if required_cuda_path not in existing_ld_path and '__LD_LIBRARY_PATH_SET__' not in os.environ:
        # Set marker to prevent infinite restart loop
        os.environ['__LD_LIBRARY_PATH_SET__'] = '1'

        # Update LD_LIBRARY_PATH
        if existing_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{existing_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = new_ld_path

        print(f"Restarting script with updated LD_LIBRARY_PATH for GPU support...")
        print(f"Added paths: {new_ld_path}\n")

        # Restart the script with the updated environment
        os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

    # If already restarted or path is correct, just update the environ
    if existing_ld_path and required_cuda_path not in existing_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{existing_ld_path}"
    elif not existing_ld_path:
        os.environ['LD_LIBRARY_PATH'] = new_ld_path


def ensure_dependencies():
    """Ensure required Python packages are installed."""
    needs_restart = False

    # Check if essentia has TensorFlow support
    try:
        import essentia.standard as es
        # Try to access a TensorFlow-based algorithm
        if not hasattr(es, 'TensorflowPredictEffnetDiscogs'):
            print("Essentia TensorFlow support not found.")
            print("Reinstalling with TensorFlow support...")
            # Uninstall regular essentia and install essentia-tensorflow
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'uninstall', '-y', 'essentia'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--quiet', 'essentia-tensorflow'
            ])
            print("Essentia with TensorFlow support installed successfully!")
            needs_restart = True
    except ImportError:
        # Essentia not installed
        print("Installing essentia-tensorflow...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--quiet', 'essentia-tensorflow'
        ])
        print("Dependencies installed successfully!")
        needs_restart = True

    # Ensure numpy is installed
    try:
        import numpy
    except ImportError:
        print("Installing numpy...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--quiet', 'numpy'
        ])
        print("Numpy installed successfully!")
        needs_restart = True

    if needs_restart:
        print("\nRestarting script to load new dependencies...")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def configure_gpu():
    """Configure TensorFlow to use GPU if available."""
    try:
        import tensorflow as tf

        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print(f"\n{'='*80}")
            print(f"GPU CONFIGURATION")
            print(f"{'='*80}")
            print(f"Found {len(gpus)} GPU(s):")

            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                try:
                    # Enable memory growth for this GPU
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"    ✓ Memory growth enabled")
                except RuntimeError as e:
                    print(f"    ✗ Could not set memory growth: {e}")

            # Set visible devices (use all available GPUs)
            tf.config.set_visible_devices(gpus, 'GPU')

            # Set GPU as default device for all operations
            # This is critical for Essentia's TensorFlow operations
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=22000)]  # 22GB limit
            )

            # Log GPU details
            print(f"\nTensorFlow GPU Configuration:")
            print(f"  TensorFlow version: {tf.__version__}")
            print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
            print(f"  GPU device set as default: /GPU:0")

            # Test GPU computation
            try:
                with tf.device('/GPU:0'):
                    test_tensor = tf.random.normal([1000, 1000])
                    result = tf.matmul(test_tensor, test_tensor)
                print(f"  ✓ GPU computation test successful")
                print(f"\n{'='*80}")
                print(f"✓✓✓ GPU IS CONFIGURED AND READY ✓✓✓")
                print(f"{'='*80}\n")
            except Exception as e:
                print(f"  ✗ GPU test failed: {e}")
                print(f"{'='*80}\n")

            return True  # GPU available

        else:
            print(f"\n{'='*80}")
            print(f"WARNING: No GPU detected - using CPU")
            print(f"TensorFlow version: {tf.__version__}")
            print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
            print(f"{'='*80}\n")
            return False  # No GPU

    except ImportError:
        print("\nTensorFlow not found - GPU configuration skipped")
        print("Note: Essentia-TensorFlow includes TensorFlow\n")
        return False
    except Exception as e:
        print(f"\nWarning: GPU configuration failed: {e}")
        print("Continuing with default configuration...\n")
        return False


def get_gpu_memory_usage():
    """Get current GPU memory usage if available."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # This requires nvidia-smi to be available
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for i, line in enumerate(lines):
                    used, total = line.split(',')
                    gpu_info.append({
                        'gpu_id': i,
                        'used_mb': int(used.strip()),
                        'total_mb': int(total.strip()),
                        'percent': (int(used.strip()) / int(total.strip())) * 100
                    })
                return gpu_info
    except Exception:
        pass
    return None


def download_file(url: str, output_path: Path):
    """Download a file from URL to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  ✓ Already exists: {output_path.name}")
        return

    print(f"  Downloading: {output_path.name}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Downloaded: {output_path.name}")
    except Exception as e:
        print(f"  ✗ Failed to download {output_path.name}: {e}")
        raise


def ensure_models(models_dir: Path):
    """Download required Essentia models if they don't exist."""
    models_dir.mkdir(parents=True, exist_ok=True)

    # Model URLs
    embedding_base_url = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
    genre_base_url = "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/"
    mood_base_url = "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/"
    instrument_base_url = "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/"

    models_to_download = [
        (f"{embedding_base_url}discogs-effnet-bs64-1.pb", models_dir / "discogs-effnet-bs64-1.pb"),
        (f"{genre_base_url}mtg_jamendo_genre-discogs-effnet-1.pb", models_dir / "mtg_jamendo_genre-discogs-effnet-1.pb"),
        (f"{genre_base_url}mtg_jamendo_genre-discogs-effnet-1.json", models_dir / "mtg_jamendo_genre-discogs-effnet-1.json"),
        (f"{mood_base_url}mtg_jamendo_moodtheme-discogs-effnet-1.pb", models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"),
        (f"{mood_base_url}mtg_jamendo_moodtheme-discogs-effnet-1.json", models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.json"),
        (f"{instrument_base_url}mtg_jamendo_instrument-discogs-effnet-1.pb", models_dir / "mtg_jamendo_instrument-discogs-effnet-1.pb"),
        (f"{instrument_base_url}mtg_jamendo_instrument-discogs-effnet-1.json", models_dir / "mtg_jamendo_instrument-discogs-effnet-1.json"),
    ]

    print("Checking/downloading models...")
    for url, path in models_to_download:
        download_file(url, path)
    print("All models ready!\n")


# Ensure dependencies are installed first
ensure_dependencies()

# Configure GPU before importing Essentia (TensorFlow will be configured)
configure_gpu()

# Now import the packages
import numpy as np
import essentia
import essentia.standard as es


def detect_bpm_essentia(path: str, sr: int = 44100):
    """
    BPM detection using Essentia's RhythmExtractor2013.
    Returns (bpm, confidence_raw, confidence_normalized, is_reliable).
    """
    # Load audio (Essentia works well at 44.1k for rhythm)
    audio = es.MonoLoader(filename=path, sampleRate=sr)()

    # Optional: trim leading/trailing silence
    # Skipping for now to avoid parameter type issues
    # trimmer = es.StartStopSilence()
    # audio = trimmer(audio)

    # Rhythm extraction
    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beat_conf, _, _ = rhythm(audio)

    # beat_conf can be a single float or array; handle both cases
    if isinstance(beat_conf, (list, np.ndarray)):
        conf_raw = float(np.mean(beat_conf)) if len(beat_conf) > 0 else 0.0
    else:
        conf_raw = float(beat_conf)

    # Normalize confidence to 0-1 range
    # Essentia's beat confidence is typically in range [0, ~5], with higher being better
    # We'll use a sigmoid-like normalization
    conf_normalized = min(1.0, max(0.0, conf_raw / 5.0))

    # Consider BPM reliable if normalized confidence > 0.5
    is_reliable = conf_normalized > 0.5

    return float(bpm), conf_raw, conf_normalized, is_reliable


def detect_genre_essentia(
    path: str,
    embedding_model_pb: str,
    classifier_model_pb: str,
    labels_json: str,
    topk: int = 5,
):
    """
    Genre inference using Essentia's DiscogEffnet embeddings + genre classifier.

    You must provide:
      - embedding_model_pb: path to discogs-effnet embedding model (.pb)
      - classifier_model_pb: path to genre classifier head (.pb)
      - labels_json: path to a labels file (json/dict with "classes" key)

    Returns:
      - list of (label, score) sorted desc
    """
    # Load mono audio at 16kHz (required for discogs-effnet)
    audio = es.MonoLoader(filename=path, sampleRate=16000)()

    # Extract embeddings using TensorflowPredictEffnetDiscogs
    embedding_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(Path(embedding_model_pb).expanduser()),
        output="PartitionedCall:1"
    )
    embeddings = embedding_model(audio)

    # Run genre classification on embeddings
    classifier = es.TensorflowPredict2D(
        graphFilename=str(Path(classifier_model_pb).expanduser()),
        input="model/Placeholder",
        output="model/Sigmoid"
    )
    predictions = classifier(embeddings)

    # Load labels
    with open(labels_json, "r") as f:
        labels_obj = json.load(f)

    if isinstance(labels_obj, dict) and "classes" in labels_obj:
        labels = labels_obj["classes"]
    elif isinstance(labels_obj, list):
        labels = labels_obj
    else:
        raise RuntimeError("labels_json must have 'classes' key or be a list")

    # Average predictions across time
    predictions_avg = np.mean(predictions, axis=0)

    if len(labels) != len(predictions_avg):
        raise RuntimeError(f"Label count ({len(labels)}) != output size ({len(predictions_avg)})")

    # Top-k
    idx = np.argsort(-predictions_avg)[:topk]
    return [(str(labels[i]), float(predictions_avg[i])) for i in idx]


def detect_mood_theme_essentia(
    path: str,
    embedding_model_pb: str,
    classifier_model_pb: str,
    labels_json: str,
    topk: int = 10,
):
    """
    Mood/theme inference using Essentia's DiscogEffnet embeddings + mood classifier.
    Returns list of (mood_tag, score) sorted desc.
    """
    # Load mono audio at 16kHz (required for discogs-effnet)
    audio = es.MonoLoader(filename=path, sampleRate=16000)()

    # Extract embeddings using TensorflowPredictEffnetDiscogs
    embedding_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(Path(embedding_model_pb).expanduser()),
        output="PartitionedCall:1"
    )
    embeddings = embedding_model(audio)

    # Run mood/theme classification on embeddings
    classifier = es.TensorflowPredict2D(
        graphFilename=str(Path(classifier_model_pb).expanduser()),
        input="model/Placeholder",
        output="model/Sigmoid"
    )
    predictions = classifier(embeddings)

    # Load labels
    with open(labels_json, "r") as f:
        labels_obj = json.load(f)

    if isinstance(labels_obj, dict) and "classes" in labels_obj:
        labels = labels_obj["classes"]
    elif isinstance(labels_obj, list):
        labels = labels_obj
    else:
        raise RuntimeError("labels_json must have 'classes' key or be a list")

    # Average predictions across time
    predictions_avg = np.mean(predictions, axis=0)

    if len(labels) != len(predictions_avg):
        raise RuntimeError(f"Label count ({len(labels)}) != output size ({len(predictions_avg)})")

    # Top-k
    idx = np.argsort(-predictions_avg)[:topk]
    return [(str(labels[i]), float(predictions_avg[i])) for i in idx]


def detect_instruments_essentia(
    path: str,
    embedding_model_pb: str,
    classifier_model_pb: str,
    labels_json: str,
    topk: int = 10,
):
    """
    Instrument inference using Essentia's DiscogEffnet embeddings + instrument classifier.
    Returns list of (instrument, score) sorted desc.
    """
    # Load mono audio at 16kHz (required for discogs-effnet)
    audio = es.MonoLoader(filename=path, sampleRate=16000)()

    # Extract embeddings using TensorflowPredictEffnetDiscogs
    embedding_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(Path(embedding_model_pb).expanduser()),
        output="PartitionedCall:1"
    )
    embeddings = embedding_model(audio)

    # Run instrument classification on embeddings
    classifier = es.TensorflowPredict2D(
        graphFilename=str(Path(classifier_model_pb).expanduser()),
        input="model/Placeholder",
        output="model/Sigmoid"
    )
    predictions = classifier(embeddings)

    # Load labels
    with open(labels_json, "r") as f:
        labels_obj = json.load(f)

    if isinstance(labels_obj, dict) and "classes" in labels_obj:
        labels = labels_obj["classes"]
    elif isinstance(labels_obj, list):
        labels = labels_obj
    else:
        raise RuntimeError("labels_json must have 'classes' key or be a list")

    # Average predictions across time
    predictions_avg = np.mean(predictions, axis=0)

    if len(labels) != len(predictions_avg):
        raise RuntimeError(f"Label count ({len(labels)}) != output size ({len(predictions_avg)})")

    # Top-k
    idx = np.argsort(-predictions_avg)[:topk]
    return [(str(labels[i]), float(predictions_avg[i])) for i in idx]


def detect_musical_structure(path: str, sr: int = 44100) -> Dict:
    """
    Detect musical structure: key, mode, meter, danceability, onset rate.
    """
    audio = es.MonoLoader(filename=path, sampleRate=sr)()

    # Key detection
    key_detector = es.KeyExtractor()
    key, scale, key_strength = key_detector(audio)

    # Normalize key strength to 0-1 (Essentia returns values typically 0-1 already)
    key_confidence = float(key_strength)

    # Meter/time signature estimation (simplified)
    # Essentia doesn't have direct meter detection, but we can estimate from rhythm
    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beat_conf, _, beats_loudness = rhythm(audio)

    # Simple heuristic: analyze beat intervals to guess meter
    if len(beats) > 4:
        beat_intervals = np.diff(beats)
        # Check for patterns that suggest 3/4 vs 4/4
        # This is a simplified heuristic
        meter = "4/4"  # Default to 4/4 for most popular music
    else:
        meter = "4/4"

    # Danceability (using rhythm strength and beat regularity)
    danceability_extractor = es.Danceability()
    danceability, dfa = danceability_extractor(audio)

    # Onset rate (notes per second proxy)
    # Use beat times as a proxy for onset rate
    duration = len(audio) / sr
    onset_rate = len(beats) / duration if duration > 0 else 0.0

    return {
        "key": key,
        "mode": scale,
        "key_confidence": key_confidence,
        "meter": meter,
        "danceability": float(danceability),
        "onset_rate": float(onset_rate),
    }


def analyze_audio_quality(path: str, sr: int = 44100) -> Dict:
    """
    Analyze audio quality: loudness, dynamics, clipping, silence ratio.
    """
    # Load mono audio
    audio = es.MonoLoader(filename=path, sampleRate=sr)()

    # Loudness (RMS-based proxy for LUFS)
    # Compute RMS loudness in dBFS
    rms = np.sqrt(np.mean(audio**2))
    loudness_integrated = 20 * np.log10(rms) if rms > 0 else -96

    # Estimate loudness range using percentiles
    frame_size = int(sr * 0.4)  # 400ms frames
    hop_size = frame_size // 2
    frame_loudnesses = []

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i+frame_size]
        frame_rms = np.sqrt(np.mean(frame**2))
        if frame_rms > 0:
            frame_loudnesses.append(20 * np.log10(frame_rms))

    if frame_loudnesses:
        loudness_range = np.percentile(frame_loudnesses, 95) - np.percentile(frame_loudnesses, 10)
    else:
        loudness_range = 0.0

    # Peak dBFS (clipping detection)
    peak = np.max(np.abs(audio))
    peak_dbfs = 20 * np.log10(peak) if peak > 0 else -np.inf

    # Dynamic range (crest factor)
    rms = np.sqrt(np.mean(audio**2))
    crest_factor = peak / rms if rms > 0 else 0
    dynamic_range_db = 20 * np.log10(crest_factor) if crest_factor > 0 else 0

    # Silence ratio
    # Use energy-based silence detection
    frame_size = 2048
    hop_size = 1024
    silence_threshold = 1e-4

    frames = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)
    silent_frames = 0
    total_frames = 0

    for frame in frames:
        total_frames += 1
        energy = np.sum(frame**2)
        if energy < silence_threshold:
            silent_frames += 1

    silence_ratio = silent_frames / total_frames if total_frames > 0 else 0

    return {
        "lufs_integrated": float(loudness_integrated),
        "loudness_range": float(loudness_range),
        "peak_dbfs": float(peak_dbfs),
        "dynamic_range_db": float(dynamic_range_db),
        "crest_factor": float(crest_factor),
        "silence_ratio": float(silence_ratio),
        "is_clipped": peak_dbfs > -0.1,  # Near 0dBFS indicates clipping
        "is_too_quiet": loudness_integrated < -40,  # Very quiet audio
    }


def get_technical_metadata(path: str) -> Dict:
    """
    Extract technical metadata using ffprobe.
    """
    try:
        # Run ffprobe to get file info
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return {
                "duration_sec": None,
                "sample_rate_hz": None,
                "channels": None,
                "bitrate_kbps": None,
                "codec_name": None,
                "container": None,
                "source_format": None,
                "error": "ffprobe failed"
            }

        data = json.loads(result.stdout)

        # Extract format info
        format_info = data.get("format", {})
        duration = float(format_info.get("duration", 0))
        bitrate = int(format_info.get("bit_rate", 0)) // 1000  # Convert to kbps
        container = format_info.get("format_name", "unknown")

        # Extract audio stream info
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if audio_stream:
            sample_rate = int(audio_stream.get("sample_rate", 0))
            channels = int(audio_stream.get("channels", 0))
            codec_name = audio_stream.get("codec_name", "unknown")
        else:
            sample_rate = None
            channels = None
            codec_name = None

        # Get source format from file extension
        source_format = Path(path).suffix.lstrip(".")

        return {
            "duration_sec": duration,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "bitrate_kbps": bitrate,
            "codec_name": codec_name,
            "container": container,
            "source_format": source_format,
        }

    except subprocess.TimeoutExpired:
        return {
            "duration_sec": None,
            "sample_rate_hz": None,
            "channels": None,
            "bitrate_kbps": None,
            "codec_name": None,
            "container": None,
            "source_format": None,
            "error": "ffprobe timeout"
        }
    except Exception as e:
        return {
            "duration_sec": None,
            "sample_rate_hz": None,
            "channels": None,
            "bitrate_kbps": None,
            "codec_name": None,
            "container": None,
            "source_format": None,
            "error": str(e)
        }


def comprehensive_audio_analysis(
    path: str,
    embedding_model_pb: str,
    genre_model_pb: str,
    genre_labels_json: str,
    mood_model_pb: str,
    mood_labels_json: str,
    instrument_model_pb: str,
    instrument_labels_json: str,
) -> Dict:
    """
    Comprehensive audio analysis returning all requested features.
    """
    # BPM Detection (with fixed confidence normalization)
    bpm, bpm_conf_raw, bpm_conf_norm, bpm_is_reliable = detect_bpm_essentia(path)

    # Musical Structure
    musical_structure = detect_musical_structure(path)

    # Audio Quality
    audio_quality = analyze_audio_quality(path)

    # Technical Metadata
    tech_metadata = get_technical_metadata(path)

    # Genre Detection
    genre_results = detect_genre_essentia(
        path, embedding_model_pb, genre_model_pb, genre_labels_json, topk=5
    )

    # Mood/Theme Detection
    mood_results = detect_mood_theme_essentia(
        path, embedding_model_pb, mood_model_pb, mood_labels_json, topk=10
    )

    # Instrument Detection
    instrument_results = detect_instruments_essentia(
        path, embedding_model_pb, instrument_model_pb, instrument_labels_json, topk=10
    )

    # Derive energy level from mood tags
    energy_indicators = {
        "energetic": 1.0, "fast": 0.9, "powerful": 0.9, "upbeat": 0.8,
        "epic": 0.8, "party": 0.8, "action": 0.7, "sport": 0.7,
        "relaxing": 0.2, "calm": 0.1, "slow": 0.2, "soft": 0.2,
        "meditative": 0.1, "ballad": 0.3
    }
    energy_scores = [
        score * energy_indicators.get(tag, 0.5)
        for tag, score in mood_results
        if tag in energy_indicators
    ]
    energy_value = np.mean(energy_scores) if energy_scores else 0.5

    if energy_value < 0.35:
        energy_level = "low"
    elif energy_value < 0.65:
        energy_level = "medium"
    else:
        energy_level = "high"

    # Derive valence (happy/sad) from mood tags
    valence_map = {
        "happy": 0.9, "fun": 0.85, "funny": 0.8, "uplifting": 0.85,
        "positive": 0.8, "upbeat": 0.75, "groovy": 0.7, "party": 0.8,
        "sad": 0.1, "melancholic": 0.2, "dark": 0.2, "dramatic": 0.3,
        "emotional": 0.4, "calm": 0.5, "relaxing": 0.6
    }
    valence_scores = [
        score * valence_map.get(tag, 0.5)
        for tag, score in mood_results
        if tag in valence_map
    ]
    valence = np.mean(valence_scores) if valence_scores else 0.5

    # Derive arousal from mood tags
    arousal_map = {
        "energetic": 0.9, "powerful": 0.85, "epic": 0.8, "dramatic": 0.75,
        "fast": 0.8, "action": 0.75, "party": 0.8,
        "calm": 0.2, "relaxing": 0.15, "meditative": 0.1, "slow": 0.25,
        "soft": 0.3, "ballad": 0.35
    }
    arousal_scores = [
        score * arousal_map.get(tag, 0.5)
        for tag, score in mood_results
        if tag in arousal_map
    ]
    arousal = np.mean(arousal_scores) if arousal_scores else 0.5

    # Detect vocals
    voice_score = next((score for inst, score in instrument_results if inst == "voice"), 0.0)
    has_vocals = voice_score > 0.05
    instrumental = voice_score < .05

    # Vocal type (simplified heuristic based on genre/mood)
    vocal_type = "unknown"
    if has_vocals:
        # Check genre for rap indicators
        top_genres = [g.lower() for g, _ in genre_results[:3]]
        if any(g in top_genres for g in ["hiphop", "rap"]):
            vocal_type = "rap"
        elif voice_score > 0.5:
            # High voice presence, default to mixed or predominant
            vocal_type = "mixed"  # Could be male/female/choir
        else:
            vocal_type = "mixed"

    # Acoustic vs Electronic
    electronic_instruments = ["synthesizer", "computer", "drummachine", "sampler", "pad"]
    acoustic_instruments = [
        "acousticguitar", "acousticbassguitar", "piano", "drums",
        "violin", "cello", "flute", "trumpet", "saxophone"
    ]

    electronic_score = sum(
        score for inst, score in instrument_results if inst in electronic_instruments
    )
    acoustic_score = sum(
        score for inst, score in instrument_results if inst in acoustic_instruments
    )

    total = electronic_score + acoustic_score
    if total > 0:
        acoustic_vs_electronic = acoustic_score / total  # 0=electronic, 1=acoustic
    else:
        acoustic_vs_electronic = 0.5

    # ============================================================================
    # CLUBBINESS SCORES: Overall + Genre-Specific Sub-Club Scores
    # ============================================================================

    def compute_sub_club_score(
        target_genres: dict,
        optimal_bpm_range: tuple,
        good_bpm_range: tuple,
        target_moods: dict,
        energy_weight: float = 0.5,
        vocal_preference: float = 0.5,  # 0=prefer instrumental, 1=prefer vocals, 0.5=neutral
        instrumentation_pref: dict = None,  # Instrument preferences
    ) -> float:
        """
        Compute a genre-specific club score.
        """
        # Genre match score
        genre_match = 0.0
        genre_total_weight = 0.0
        for genre, score in genre_results[:5]:
            genre_lower = genre.lower()
            weight = score
            if genre_lower in target_genres:
                genre_match += target_genres[genre_lower] * weight
                genre_total_weight += weight
            else:
                genre_total_weight += weight * 0.2  # Small weight for other genres

        genre_score = genre_match / max(genre_total_weight, 0.01)

        # Mood match score
        mood_match = 0.0
        mood_total_weight = 0.0
        for mood, score in mood_results[:10]:
            mood_lower = mood.lower()
            weight = score
            if mood_lower in target_moods:
                mood_match += target_moods[mood_lower] * weight
                mood_total_weight += weight
            else:
                mood_total_weight += weight * 0.3

        mood_score = mood_match / max(mood_total_weight, 0.01)

        # BPM score based on scene-specific ranges
        optimal_min, optimal_max = optimal_bpm_range
        good_min, good_max = good_bpm_range

        if optimal_min <= bpm <= optimal_max:
            bpm_score = 1.0
        elif good_min <= bpm <= good_max:
            bpm_score = 0.7
        else:
            # Calculate distance from optimal range
            if bpm < optimal_min:
                distance = optimal_min - bpm
            else:
                distance = bpm - optimal_max
            bpm_score = max(0.0, 0.5 - (distance / 50.0))

        # Instrumentation preference score
        inst_score = 0.5  # Default neutral
        if instrumentation_pref:
            inst_match = 0.0
            inst_total = 0.0
            for inst, score in instrument_results[:10]:
                inst_lower = inst.lower()
                if inst_lower in instrumentation_pref:
                    inst_match += instrumentation_pref[inst_lower] * score
                    inst_total += score
                else:
                    inst_total += score * 0.3
            inst_score = inst_match / max(inst_total, 0.01)

        # Energy score (some scenes prefer high energy, others groove)
        energy_contrib = energy_value * energy_weight

        # Vocal match
        if has_vocals:
            vocal_score = vocal_preference
        else:
            vocal_score = 1.0 - vocal_preference

        # Danceability
        dance_score = min(1.0, musical_structure["danceability"] / 3.0)

        # Weighted combination
        sub_score = (
            genre_score * 0.30 +
            mood_score * 0.20 +
            bpm_score * 0.20 +
            dance_score * 0.12 +
            inst_score * 0.10 +
            energy_contrib * 0.06 +
            vocal_score * 0.02
        )

        return max(0.0, min(1.0, sub_score))

    # ============================================================================
    # GENERAL CLUBBINESS SCORE
    # ============================================================================

    # Genre contribution to clubbiness
    club_genres = {
        "dance": 1.0, "house": 1.0, "techno": 1.0, "trance": 0.95, "edm": 1.0,
        "club": 1.0, "dubstep": 0.9, "drumnbass": 0.9, "electronic": 0.85,
        "eurodance": 0.95, "deephouse": 0.95, "electropop": 0.8, "disco": 0.85,
        "electronica": 0.75, "breakbeat": 0.8, "progressive": 0.7
    }
    anti_club_genres = {
        "classical": -0.5, "folk": -0.4, "acoustic": -0.3, "country": -0.3,
        "ballad": -0.5, "ambient": -0.4, "newage": -0.4, "medieval": -0.5,
        "singersongwriter": -0.3, "orchestral": -0.4
    }

    genre_club_score = 0.0
    genre_weight_total = 0.0
    for genre, score in genre_results[:5]:  # Top 5 genres
        genre_lower = genre.lower()
        weight = score  # Use prediction score as weight

        if genre_lower in club_genres:
            genre_club_score += club_genres[genre_lower] * weight
            genre_weight_total += weight
        elif genre_lower in anti_club_genres:
            genre_club_score += anti_club_genres[genre_lower] * weight
            genre_weight_total += weight
        else:
            # Neutral genres don't contribute
            genre_weight_total += weight * 0.5

    genre_contribution = genre_club_score / max(genre_weight_total, 0.01)

    # Mood contribution to clubbiness
    club_moods = {
        "party": 1.0, "energetic": 0.95, "upbeat": 0.9, "groovy": 0.95,
        "fast": 0.85, "dance": 1.0, "fun": 0.8, "powerful": 0.7,
        "uplifting": 0.75, "happy": 0.7, "sexy": 0.8, "cool": 0.7
    }
    anti_club_moods = {
        "calm": -0.4, "relaxing": -0.5, "slow": -0.6, "sad": -0.3,
        "melancholic": -0.4, "ballad": -0.6, "meditative": -0.6,
        "soft": -0.4, "dramatic": -0.2
    }

    mood_club_score = 0.0
    mood_weight_total = 0.0
    for mood, score in mood_results[:10]:  # Top 10 moods
        mood_lower = mood.lower()
        weight = score

        if mood_lower in club_moods:
            mood_club_score += club_moods[mood_lower] * weight
            mood_weight_total += weight
        elif mood_lower in anti_club_moods:
            mood_club_score += anti_club_moods[mood_lower] * weight
            mood_weight_total += weight
        else:
            mood_weight_total += weight * 0.5

    mood_contribution = mood_club_score / max(mood_weight_total, 0.01)

    # Musical features contribution
    # Optimal club BPM range is typically 120-140 BPM
    bpm_score = 0.0
    if 118 <= bpm <= 142:
        bpm_score = 1.0  # Perfect club range
    elif 110 <= bpm <= 150:
        bpm_score = 0.8  # Good range
    elif 100 <= bpm <= 160:
        bpm_score = 0.5  # Acceptable
    elif 90 <= bpm <= 170:
        bpm_score = 0.3  # Marginal
    else:
        bpm_score = 0.1  # Too slow or too fast

    # Danceability contribution (normalize to 0-1 if needed)
    dance_score = min(1.0, musical_structure["danceability"] / 3.0)

    # Energy contribution
    energy_score = energy_value

    # Electronic instrumentation favors clubbiness
    electronic_contribution = 1.0 - acoustic_vs_electronic  # Invert so 1=electronic

    # Vocal presence (club music can be vocal or instrumental)
    # Slight preference for vocals but not strongly penalize instrumental
    vocal_contribution = 0.6 if has_vocals else 0.5

    # Combine all factors with weights
    clubbiness = (
        genre_contribution * 0.30 +      # Genre is important
        mood_contribution * 0.25 +       # Mood is important
        bpm_score * 0.15 +               # BPM range matters
        dance_score * 0.15 +             # Danceability is key
        energy_score * 0.10 +            # Energy level matters
        electronic_contribution * 0.03 + # Electronic sound
        vocal_contribution * 0.02        # Vocals (minor factor)
    )

    # Normalize to 0-1 range and clamp
    clubbiness = max(0.0, min(1.0, clubbiness))

    # Determine clubbiness category
    if clubbiness >= 0.75:
        clubbiness_category = "very high"
    elif clubbiness >= 0.55:
        clubbiness_category = "high"
    elif clubbiness >= 0.35:
        clubbiness_category = "moderate"
    elif clubbiness >= 0.15:
        clubbiness_category = "low"
    else:
        clubbiness_category = "very low"

    # ============================================================================
    # SUB-CLUB SCORES: Genre-Specific Club Scene Scores
    # ============================================================================

    # EDM / House Club Scene
    club_edm = compute_sub_club_score(
        target_genres={"house": 1.0, "techno": 1.0, "trance": 0.95, "edm": 1.0,
                      "dance": 1.0, "dubstep": 0.9, "drumnbass": 0.9, "electronic": 0.85,
                      "eurodance": 0.95, "deephouse": 0.95, "electropop": 0.8},
        optimal_bpm_range=(120, 140),
        good_bpm_range=(115, 150),
        target_moods={"energetic": 1.0, "party": 1.0, "upbeat": 0.9, "powerful": 0.8,
                     "fast": 0.85, "uplifting": 0.8, "groovy": 0.9},
        energy_weight=0.8,
        vocal_preference=0.5,
        instrumentation_pref={"synthesizer": 1.0, "computer": 0.9, "drummachine": 0.9,
                             "sampler": 0.85, "keyboard": 0.7}
    )

    # Hip-Hop Club Scene
    club_hiphop = compute_sub_club_score(
        target_genres={"hiphop": 1.0, "rap": 1.0, "rnb": 0.8, "trap": 0.95,
                      "urban": 0.85, "soul": 0.6},
        optimal_bpm_range=(80, 105),
        good_bpm_range=(70, 115),
        target_moods={"groovy": 1.0, "cool": 0.9, "dark": 0.7, "powerful": 0.8,
                     "sexy": 0.8, "party": 0.85, "upbeat": 0.7},
        energy_weight=0.4,  # Hip-hop values groove over high energy
        vocal_preference=0.9,  # Strong preference for vocals/rap
        instrumentation_pref={"bass": 1.0, "drums": 0.9, "synthesizer": 0.7,
                             "voice": 1.0, "sampler": 0.8}
    )

    # Latin Club Scene
    club_latin = compute_sub_club_score(
        target_genres={"latin": 1.0, "reggae": 0.8, "reggaeton": 1.0, "salsa": 0.95,
                      "bachata": 0.9, "merengue": 0.9, "tropical": 0.85},
        optimal_bpm_range=(95, 115),
        good_bpm_range=(85, 125),
        target_moods={"party": 1.0, "sexy": 0.9, "groovy": 0.95, "happy": 0.8,
                     "upbeat": 0.9, "summer": 0.85, "dance": 1.0},
        energy_weight=0.7,
        vocal_preference=0.8,
        instrumentation_pref={"percussion": 1.0, "brass": 0.8, "guitar": 0.7,
                             "voice": 0.9, "trumpet": 0.8}
    )

    # Rock / Alternative Club Scene
    club_rock = compute_sub_club_score(
        target_genres={"rock": 1.0, "alternative": 0.95, "alternativerock": 0.95,
                      "indie": 0.85, "punkrock": 0.9, "hardrock": 0.85,
                      "grunge": 0.8, "poprock": 0.8},
        optimal_bpm_range=(120, 145),
        good_bpm_range=(110, 155),
        target_moods={"energetic": 1.0, "powerful": 0.95, "fast": 0.9, "action": 0.85,
                     "party": 0.8, "upbeat": 0.85, "groovy": 0.75},
        energy_weight=0.9,  # Rock clubs value high energy
        vocal_preference=0.7,
        instrumentation_pref={"electricguitar": 1.0, "guitar": 0.9, "drums": 1.0,
                             "bass": 0.9, "voice": 0.8}
    )

    # Pop Club Scene
    club_pop = compute_sub_club_score(
        target_genres={"pop": 1.0, "poprock": 0.85, "electropop": 0.9, "synthpop": 0.9,
                      "dance": 0.8, "disco": 0.85, "instrumentalpop": 0.7},
        optimal_bpm_range=(115, 135),
        good_bpm_range=(105, 145),
        target_moods={"happy": 1.0, "upbeat": 1.0, "party": 0.95, "fun": 0.9,
                     "uplifting": 0.85, "positive": 0.85, "groovy": 0.8},
        energy_weight=0.6,
        vocal_preference=0.85,  # Pop strongly prefers vocals
        instrumentation_pref={"synthesizer": 0.8, "keyboard": 0.7, "drums": 0.8,
                             "voice": 1.0, "bass": 0.7}
    )

    # Reggae / Dancehall Club Scene
    club_reggae = compute_sub_club_score(
        target_genres={"reggae": 1.0, "dub": 0.9, "dancehall": 1.0, "ska": 0.7},
        optimal_bpm_range=(85, 105),
        good_bpm_range=(75, 115),
        target_moods={"groovy": 1.0, "chill": 0.8, "party": 0.8, "cool": 0.85,
                     "relaxing": 0.6, "summer": 0.8},
        energy_weight=0.3,  # Reggae values groove and vibe over energy
        vocal_preference=0.75,
        instrumentation_pref={"bass": 1.0, "drums": 0.9, "guitar": 0.8,
                             "voice": 0.9, "organ": 0.7}
    )

    # Determine best club scene match
    club_scenes = {
        "EDM/House": club_edm,
        "Hip-Hop": club_hiphop,
        "Latin": club_latin,
        "Rock": club_rock,
        "Pop": club_pop,
        "Reggae": club_reggae,
    }

    # Find top 2 club scenes
    sorted_scenes = sorted(club_scenes.items(), key=lambda x: x[1], reverse=True)
    best_club_scene = sorted_scenes[0][0]
    best_club_scene_score = sorted_scenes[0][1]

    return {
        # BPM (fixed confidence normalization)
        "bpm": bpm,
        "bpm_conf_raw": bpm_conf_raw,
        "bpm_conf_norm": bpm_conf_norm,
        "bpm_is_reliable": bpm_is_reliable,

        # Musical Structure
        "key": musical_structure["key"],
        "mode": musical_structure["mode"],
        "key_confidence": musical_structure["key_confidence"],
        "meter": musical_structure["meter"],
        "danceability": musical_structure["danceability"],
        "onset_rate": musical_structure["onset_rate"],

        # Genre/Style
        "genre_top1": genre_results[0][0] if genre_results else "unknown",
        "genre_topk": genre_results,

        # Mood/Vibe
        "mood_tags": mood_results,
        "mood_tags_dict": {tag: score for tag, score in mood_results},
        "energy_level": energy_level,
        "energy_value": float(energy_value),
        "valence": float(valence),
        "arousal": float(arousal),

        # Overall Scores
        "clubbiness": float(clubbiness),
        "clubbiness_category": clubbiness_category,

        # Sub-Club Scores (genre-specific club scenes)
        "club_edm": float(club_edm),
        "club_hiphop": float(club_hiphop),
        "club_latin": float(club_latin),
        "club_rock": float(club_rock),
        "club_pop": float(club_pop),
        "club_reggae": float(club_reggae),
        "best_club_scene": best_club_scene,
        "best_club_scene_score": float(best_club_scene_score),

        # Instrumentation
        "primary_instruments": instrument_results,
        "acoustic_vs_electronic": float(acoustic_vs_electronic),

        # Vocals
        "has_vocals": has_vocals,
        "vocal_score": float(voice_score),
        "vocal_type": vocal_type,
        "instrumental": instrumental,

        # Audio Quality
        "lufs_integrated": audio_quality["lufs_integrated"],
        "loudness_range": audio_quality["loudness_range"],
        "peak_dbfs": audio_quality["peak_dbfs"],
        "dynamic_range_db": audio_quality["dynamic_range_db"],
        "crest_factor": audio_quality["crest_factor"],
        "silence_ratio": audio_quality["silence_ratio"],
        "is_clipped": audio_quality["is_clipped"],
        "is_too_quiet": audio_quality["is_too_quiet"],

        # Technical Metadata
        "duration_sec": tech_metadata["duration_sec"],
        "sample_rate_hz": tech_metadata["sample_rate_hz"],
        "channels": tech_metadata["channels"],
        "bitrate_kbps": tech_metadata["bitrate_kbps"],
        "codec_name": tech_metadata["codec_name"],
        "container": tech_metadata["container"],
        "source_format": tech_metadata["source_format"],
    }


# ---- Example usage ----
if __name__ == "__main__":
    # Setup paths
    workspace_root = Path("/root/workspace")
    audio_dir = workspace_root / "data" / "jamendo" / "downloads"
    models_dir = workspace_root / "data" / "models"

    # Ensure models are downloaded
    ensure_models(models_dir)

    # Find sample audio files
    audio_files = sorted(audio_dir.glob("*.mp3"))[:3]  # Use first 3 MP3 files
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        sys.exit(1)

    test = str(audio_files[0])  # Use first file as test

    # Model paths
    embedding_model = str(models_dir / "discogs-effnet-bs64-1.pb")
    genre_model = str(models_dir / "mtg_jamendo_genre-discogs-effnet-1.pb")
    genre_labels = str(models_dir / "mtg_jamendo_genre-discogs-effnet-1.json")
    mood_model = str(models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    mood_labels = str(models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.json")
    instrument_model = str(models_dir / "mtg_jamendo_instrument-discogs-effnet-1.pb")
    instrument_labels = str(models_dir / "mtg_jamendo_instrument-discogs-effnet-1.json")

    print("=" * 80)
    print("COMPREHENSIVE AUDIO ANALYSIS")
    print("=" * 80)
    print(f"File: {test}")
    print()

    # Show GPU memory before analysis
    gpu_mem_before = get_gpu_memory_usage()
    if gpu_mem_before:
        print(f"GPU Memory (before analysis):")
        for gpu in gpu_mem_before:
            print(f"  GPU {gpu['gpu_id']}: {gpu['used_mb']} MB / {gpu['total_mb']} MB ({gpu['percent']:.1f}%)")
        print()

    # Run comprehensive analysis
    print("Running comprehensive audio analysis...")
    print("(This may take a minute... GPU will be used for TensorFlow inference)\n")

    results = comprehensive_audio_analysis(
        path=test,
        embedding_model_pb=embedding_model,
        genre_model_pb=genre_model,
        genre_labels_json=genre_labels,
        mood_model_pb=mood_model,
        mood_labels_json=mood_labels,
        instrument_model_pb=instrument_model,
        instrument_labels_json=instrument_labels,
    )

    # Show GPU memory after analysis
    gpu_mem_after = get_gpu_memory_usage()
    if gpu_mem_after:
        print("\nGPU Memory (after analysis):")
        for gpu in gpu_mem_after:
            print(f"  GPU {gpu['gpu_id']}: {gpu['used_mb']} MB / {gpu['total_mb']} MB ({gpu['percent']:.1f}%)")
        print()

    # Display results
    print("TECHNICAL METADATA")
    print("-" * 80)
    print(f"  Duration: {results['duration_sec']:.2f}s")
    print(f"  Sample Rate: {results['sample_rate_hz']} Hz")
    print(f"  Channels: {results['channels']}")
    print(f"  Bitrate: {results['bitrate_kbps']} kbps")
    print(f"  Codec: {results['codec_name']}")
    print(f"  Container: {results['container']}")
    print(f"  Format: {results['source_format']}")
    print()

    print("TEMPO & RHYTHM")
    print("-" * 80)
    print(f"  BPM: {results['bpm']:.2f}")
    print(f"    Raw Confidence: {results['bpm_conf_raw']:.3f}")
    print(f"    Normalized Confidence: {results['bpm_conf_norm']:.3f} (0-1)")
    print(f"    Reliable: {results['bpm_is_reliable']}")
    print(f"  Meter: {results['meter']}")
    print(f"  Danceability: {results['danceability']:.3f}")
    print(f"  Onset Rate: {results['onset_rate']:.2f} notes/sec")
    print()

    print("KEY & MODE")
    print("-" * 80)
    print(f"  Key: {results['key']}")
    print(f"  Mode: {results['mode']}")
    print(f"  Confidence: {results['key_confidence']:.3f}")
    print()

    print("GENRE / STYLE")
    print("-" * 80)
    print(f"  Primary Genre: {results['genre_top1']}")
    print(f"  Top 5 Genres:")
    for i, (genre, score) in enumerate(results['genre_topk'], 1):
        print(f"    {i}. {genre:25s} {score:.4f}")
    print()

    print("MOOD / VIBE")
    print("-" * 80)
    print(f"  Energy Level: {results['energy_level']} ({results['energy_value']:.3f})")
    print(f"  Valence (happy/sad): {results['valence']:.3f} (0=sad, 1=happy)")
    print(f"  Arousal (calm/intense): {results['arousal']:.3f} (0=calm, 1=intense)")
    print(f"  Top Mood Tags:")
    for i, (mood, score) in enumerate(results['mood_tags'][:8], 1):
        print(f"    {i}. {mood:20s} {score:.4f}")
    print()

    print("OVERALL SCORES")
    print("-" * 80)
    print(f"  General Clubbiness: {results['clubbiness']:.3f} ({results['clubbiness_category']})")
    print(f"    (Combined metric: genre + mood + BPM + danceability + energy)")
    print()
    print(f"  Best Club Scene: {results['best_club_scene']} ({results['best_club_scene_score']:.3f})")
    print()
    print(f"  Sub-Club Scores (genre-specific scenes):")
    print(f"    EDM/House Club:  {results['club_edm']:.3f}")
    print(f"    Hip-Hop Club:    {results['club_hiphop']:.3f}")
    print(f"    Latin Club:      {results['club_latin']:.3f}")
    print(f"    Rock Club:       {results['club_rock']:.3f}")
    print(f"    Pop Club:        {results['club_pop']:.3f}")
    print(f"    Reggae Club:     {results['club_reggae']:.3f}")
    print()

    print("INSTRUMENTATION")
    print("-" * 80)
    print(f"  Acoustic vs Electronic: {results['acoustic_vs_electronic']:.3f} (0=electronic, 1=acoustic)")
    print(f"  Primary Instruments:")
    for i, (inst, score) in enumerate(results['primary_instruments'][:8], 1):
        print(f"    {i}. {inst:25s} {score:.4f}")
    print()

    print("VOCALS")
    print("-" * 80)
    print(f"  Has Vocals: {results['has_vocals']}")
    print(f"  Vocal Score: {results['vocal_score']:.4f}")
    print(f"  Vocal Type: {results['vocal_type']}")
    print(f"  Instrumental: {results['instrumental']}")
    print()

    print("AUDIO QUALITY")
    print("-" * 80)
    print(f"  LUFS Integrated: {results['lufs_integrated']:.2f} LUFS")
    print(f"  Loudness Range: {results['loudness_range']:.2f} LU")
    print(f"  Peak: {results['peak_dbfs']:.2f} dBFS")
    print(f"  Dynamic Range: {results['dynamic_range_db']:.2f} dB")
    print(f"  Crest Factor: {results['crest_factor']:.2f}")
    print(f"  Silence Ratio: {results['silence_ratio']:.3f}")
    print(f"  Is Clipped: {results['is_clipped']}")
    print(f"  Is Too Quiet: {results['is_too_quiet']}")
    print()

    print("=" * 80)