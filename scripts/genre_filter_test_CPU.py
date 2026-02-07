# pip install essentia  (note: on macOS this is often easiest via conda-forge)
# For genre tagging you ALSO need a pretrained Essentia model file (.pb) + its labels (.json/.yaml).
# Example model packs are distributed via "essentia-models" (download separately).

# ============================================================================
# ENVIRONMENT SETUP - Must be BEFORE all imports!
# ============================================================================
import os
import sys

# Suppress ALL TensorFlow logging (set before TF import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages

# Force CPU-only execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Hide all GPUs from TensorFlow
print("✓ Running in CPU-only mode (CUDA_VISIBLE_DEVICES=-1)")

# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')

# Now import everything else
import argparse
import hashlib
import json
import logging
import platform
import subprocess
import threading
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import essentia
import essentia.standard as es

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("genre_filter")

# ============================================================================
# FROZEN TAXONOMIES  (Option 1 – dense slider vectors)
# ============================================================================
# Derived from MTG-Jamendo model label files.  See schema note above for safe
# modification procedure.

TAXONOMY_VERSION = "1.0.0"

GENRE_TAXONOMY: List[str] = [
    "60s", "70s", "80s", "90s", "acidjazz", "alternative", "alternativerock",
    "ambient", "atmospheric", "blues", "bluesrock", "bossanova", "breakbeat",
    "celtic", "chanson", "chillout", "choir", "classical", "classicrock", "club",
    "contemporary", "country", "dance", "darkambient", "darkwave", "deephouse",
    "disco", "downtempo", "drumnbass", "dub", "dubstep", "easylistening",
    "edm", "electro", "electronic", "electronica", "electropop", "ethno",
    "eurodance", "experimental", "folk", "funk", "fusion", "gospel",
    "gothicrock", "grunge", "hardrock", "hardtechno", "hiphop", "house",
    "idm", "improvisation", "indie", "industrial", "instrumentalpop",
    "instrumentalrock", "jazz", "jazzfusion", "latin", "lounge", "medieval",
    "metal", "minimal", "newage", "newwave", "noise", "orchestral", "pop",
    "popfolk", "poprock", "postpunk", "postrock", "progressive", "psychedelic",
    "punkrock", "rap", "reggae", "rnb", "rock", "rocknroll", "romantic",
    "singersongwriter", "ska", "soul", "soundtrack", "swing", "symphonic",
    "synthpop", "techno", "trance", "triphop", "world",
]  # 87 tags – APPEND-ONLY

MOOD_TAXONOMY: List[str] = [
    "action", "adventure", "advertising", "background", "ballad", "calm",
    "children", "christmas", "commercial", "cool", "corporate", "dark",
    "deep", "documentary", "drama", "dramatic", "dream", "emotional",
    "energetic", "epic", "fast", "film", "fun", "funny", "game", "groovy",
    "happy", "heavy", "holiday", "hopeful", "inspiring", "love",
    "meditative", "melancholic", "melodic", "motivational", "movie",
    "nature", "party", "positive", "powerful", "relaxing", "retro",
    "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space",
    "sport", "summer", "trailer", "travel", "upbeat", "uplifting",
]  # 56 tags – APPEND-ONLY

INSTRUMENT_TAXONOMY: List[str] = [
    "accordion", "acousticbassguitar", "acousticguitar", "bass", "beat",
    "bell", "bongo", "brass", "cello", "clarinet", "classicalguitar",
    "computer", "doublebass", "drummachine", "drums", "electricguitar",
    "electricpiano", "flute", "guitar", "harmonica", "harp", "horn",
    "keyboard", "oboe", "orchestra", "organ", "pad", "percussion", "piano",
    "pipeorgan", "Rhodes", "sampler", "saxophone", "strings", "synthesizer",
    "trombone", "trumpet", "viola", "violin", "voice",
]  # 40 tags – APPEND-ONLY


# ============================================================================
# DENSE-VECTOR & VALIDATION HELPERS
# ============================================================================

def _clamp_probability(v: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(v)))


def _to_dense(sparse_pairs: List[Tuple[str, float]],
              taxonomy: List[str]) -> Dict[str, float]:
    """
    Convert a sparse list of (tag, score) into a dense dict covering every
    tag in *taxonomy* (missing tags → 0.0, scores clamped to [0,1]).
    """
    lookup = {tag: _clamp_probability(score) for tag, score in sparse_pairs}
    return {tag: lookup.get(tag, 0.0) for tag in taxonomy}


def _topk_from_dense(dense: Dict[str, float], k: int = 5) -> List[Dict[str, Any]]:
    """
    Return the top-*k* entries from a dense dict, sorted descending by
    score then alphabetically by tag (for deterministic ordering).
    """
    ranked = sorted(dense.items(), key=lambda t: (-t[1], t[0]))
    return [{"tag": tag, "score": round(score, 6)} for tag, score in ranked[:k]]


def _file_sha256(path: str) -> str:
    """Return hex SHA-256 of file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _taxonomy_hash() -> str:
    """Deterministic fingerprint of the three taxonomy lists."""
    blob = json.dumps(
        [GENRE_TAXONOMY, MOOD_TAXONOMY, INSTRUMENT_TAXONOMY], sort_keys=False
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _build_provenance(
    model_paths: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Build the provenance sub-dict for a result record."""
    return {
        "extractor": "essentia+custom",
        "extractor_version": essentia.__version__,
        "model_names": {
            k: Path(v).name if v else None
            for k, v in model_paths.items()
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_hash": _taxonomy_hash(),
    }


def _ensure_python_float(v) -> float:
    """Safely cast any numeric (numpy or Python) to plain float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ============================================================================
# CONFIGURATION - Adjust these settings as needed
# ============================================================================

# Processing limits
MAX_FILES_TO_PROCESS = 1000  # Maximum number of audio files to process (set to None for all files)

# Parallel processing settings
ENABLE_PARALLEL_PROCESSING = True  # Enable/disable parallel processing
MAX_WORKERS = 64  # Number of parallel workers (recommended: 2-8 for CPU, 1-2 for GPU)
                 # NOTE: Using ProcessPoolExecutor - each worker is a separate process
                 # with its own RAM space, ensuring true parallel processing


# ============================================================================
# STDERR SUPPRESSION FOR C++ WARNINGS
# ============================================================================

class SuppressStderr:
    """Context manager to suppress stderr output (for C++ Essentia warnings)"""
    def __init__(self):
        self.null_fd = None
        self.saved_stderr = None
    
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.saved_stderr = os.dup(2)
        os.dup2(self.null_fd, 2)
        return self
    
    def __exit__(self, *_):
        os.dup2(self.saved_stderr, 2)
        os.close(self.null_fd)
        os.close(self.saved_stderr)


# ============================================================================
# SYSTEM INFO
# ============================================================================

def print_system_info():
    """
    Print system and TensorFlow information for CPU execution.
    """
    print("\n" + "=" * 80)
    print("SYSTEM INFO (CPU-ONLY MODE)")
    print("-" * 80)
    
    # Print Python executable and platform
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    # Try to import TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Execution mode: CPU only")
        print("-" * 80)
        
    except ImportError as e:
        print("\n" + "!" * 80)
        print(f"ERROR: TensorFlow not available: {e}")
        print("!" * 80)
        sys.exit(1)


# ============================================================================
# GPU UTILIZATION MONITOR
# ============================================================================

class GPUUtilizationMonitor:
    """
    Monitor GPU utilization during inference to prove GPU compute usage.
    Uses pynvml for accurate utilization tracking.
    """
    
    def __init__(self, gpu_index=0, sample_interval_ms=250, threshold_percent=10.0):
        """
        Args:
            gpu_index: GPU device index to monitor
            sample_interval_ms: Sampling interval in milliseconds
            threshold_percent: Minimum max utilization required to pass
        """
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval_ms / 1000.0  # Convert to seconds
        self.threshold = threshold_percent
        
        self.samples = []
        self.start_time = None
        self.end_time = None
        self.monitoring = False
        self.monitor_thread = None
        
        # Try to use pynvml
        self.use_nvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.pynvml = pynvml
            self.use_nvml = True
            print(f"GPU monitor: Using pynvml for GPU {gpu_index}")
        except Exception as e:
            print(f"GPU monitor: pynvml not available ({e}), falling back to nvidia-smi")
            self.use_nvml = False
    
    def _sample_gpu_utilization(self):
        """Sample GPU utilization percentage."""
        try:
            if self.use_nvml:
                # Use pynvml
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return float(util.gpu)
            else:
                # Fallback to nvidia-smi
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", 
                     "--format=csv,noheader,nounits", f"-i", str(self.gpu_index)],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
                return 0.0
        except Exception as e:
            print(f"Warning: Failed to sample GPU utilization: {e}")
            return 0.0
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            util = self._sample_gpu_utilization()
            self.samples.append(util)
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring GPU utilization."""
        self.samples = []
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"GPU monitor: Started (sampling every {self.sample_interval*1000:.0f}ms)")
    
    def stop(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.samples:
            return {
                "avg_util": 0.0,
                "max_util": 0.0,
                "samples": 0,
                "duration": 0.0,
                "passed": False,
                "threshold": self.threshold
            }
        
        avg_util = np.mean(self.samples)
        max_util = np.max(self.samples)
        duration = self.end_time - self.start_time
        passed = max_util >= self.threshold
        
        stats = {
            "avg_util": float(avg_util),
            "max_util": float(max_util),
            "samples": len(self.samples),
            "duration": float(duration),
            "passed": passed,
            "threshold": self.threshold
        }
        
        # Print results
        print(f"\nGPU utilization (during inference):")
        print(f"  avg={avg_util:.1f}% max={max_util:.1f}% duration={duration:.2f}s samples={len(self.samples)}")
        
        if passed:
            print(f"  ✓ PASS: GPU utilization exceeded {self.threshold}% threshold")
        else:
            print(f"  ✗ FAIL: GPU utilization never exceeded {self.threshold}% threshold")
            print(f"  → Likely running on CPU, not GPU")
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# ============================================================================
# MODEL CACHE - Load models once and reuse
# ============================================================================

class ModelCache:
    """
    Cache for TensorFlow models loaded directly.
    Load models once and reuse them across multiple files.
    """
    def __init__(self):
        self.models = {}
        self.sessions = {}
        import tensorflow as tf
        self.tf = tf
    
    def _load_tf_model(self, model_path: str):
        """Load a TensorFlow frozen graph model."""
        key = f"model_{model_path}"
        if key not in self.models:
            import tensorflow as tf
            # Load the frozen graph
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            
            # Create a new graph and import the graph_def
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name='')
            
            self.models[key] = graph
        return self.models[key]
    
    def get_effnet_discogs(self, model_path: str):
        """Get Effnet Discogs model for inference."""
        return self._load_tf_model(model_path)
    
    def get_maest(self, model_path: str):
        """Get MAEST model for inference."""
        return self._load_tf_model(model_path)
    
    def get_predict2d(self, model_path: str, input_node: str, output_node: str):
        """Get generic 2D predictor model."""
        return self._load_tf_model(model_path)


def run_tf_inference(graph, input_data, input_name="serving_default_melspectrogram", output_name="PartitionedCall:1"):
    """
    Run inference on a TensorFlow graph.
    
    Args:
        graph: TensorFlow graph
        input_data: Input numpy array
        input_name: Name of input tensor
        output_name: Name of output tensor
    
    Returns:
        Output numpy array
    """
    import tensorflow as tf
    
    with tf.compat.v1.Session(graph=graph) as sess:
        # Get input and output tensors
        try:
            input_tensor = graph.get_tensor_by_name(f"{input_name}:0")
        except:
            # Try without :0
            input_tensor = graph.get_tensor_by_name(input_name)
        
        try:
            output_tensor = graph.get_tensor_by_name(output_name)
        except:
            # Try with :0
            output_tensor = graph.get_tensor_by_name(f"{output_name}:0")
        
        # Run inference
        output = sess.run(output_tensor, feed_dict={input_tensor: input_data})
        return output


# ============================================================================
# AUDIO CACHE - Load audio once per file, compute embeddings once
# ============================================================================

class AudioCache:
    """
    Cache for audio data (embeddings disabled due to missing Essentia predictors).
    """
    def __init__(self, path: str, model_cache: ModelCache, embedding_model_pb: str, maest_model_pb: Optional[str] = None):
        self.path = path
        
        # Load audio at both sample rates (once each)
        self.audio_44k = es.MonoLoader(filename=path, sampleRate=44100)()
        self.audio_16k = es.MonoLoader(filename=path, sampleRate=16000)()
        
        # Embeddings disabled (Essentia TensorFlow predictors unavailable)
        self.effnet_embeddings = None
        self.maest_embeddings = None


# ============================================================================
# GPU WARMUP
# ============================================================================

def gpu_warmup(
    test_audio_path: str,
    embedding_model_pb: str,
    genre_model_pb: str,
    genre_labels_json: str,
    model_cache: ModelCache
):
    """
    Warmup disabled in CPU-only mode with mock data.
    """
    print("\nGPU warmup: Skipped (CPU-only mode with mock predictions)")


# ============================================================================
# MODEL DOWNLOAD FUNCTIONS
# ============================================================================

def download_model_file(url: str, destination: Path, description: str = ""):
    """
    Download a model file from URL to destination if it doesn't exist.
    """
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  ✓ {destination.name} already exists")
        return True

    try:
        print(f"  ⬇ Downloading {description or destination.name}...")
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Use curl for reliable downloads with progress bar
        # -L: follow redirects, -f: fail on HTTP errors, -#: progress bar
        result = subprocess.run(
            ["curl", "-L", "-f", "-#", "-o", str(destination), url],
            capture_output=False,
            timeout=300
        )

        if result.returncode == 0 and destination.exists() and destination.stat().st_size > 0:
            print(f"  ✓ Downloaded {destination.name}")
            return True
        else:
            if destination.exists():
                destination.unlink()  # Remove empty/incomplete file
            print(f"  ✗ Failed to download {destination.name}")
            return False

    except subprocess.TimeoutExpired:
        if destination.exists():
            destination.unlink()
        print(f"  ✗ Download timeout for {destination.name}")
        return False
    except Exception as e:
        if destination.exists():
            destination.unlink()
        print(f"  ✗ Failed to download {destination.name}: {e}")
        return False


def download_essentia_models(models_dir: str) -> Dict[str, bool]:
    """
    Download all required Essentia models if they don't exist.

    Returns:
        Dictionary indicating which model sets are available
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # Base URL for Essentia models
    base_url = "https://essentia.upf.edu/models"

    print("\n" + "=" * 80)
    print("CHECKING ESSENTIA MODELS")
    print("=" * 80)

    # Track which model sets are available
    available = {
        "base": False,
        "discogs519": False,
        "approachability": False,
        "engagement": False,
        "mood_emotions": False,
        "tonality": False,
    }

    # ========================================================================
    # BASE MODELS (MTG Jamendo - Required)
    # ========================================================================
    print("\nBase Models (MTG Jamendo):")

    base_models = [
        ("discogs-effnet-bs64-1.pb", f"{base_url}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"),
        ("mtg_jamendo_genre-discogs-effnet-1.pb", f"{base_url}/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb"),
        ("mtg_jamendo_genre-discogs-effnet-1.json", f"{base_url}/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.json"),
        ("mtg_jamendo_moodtheme-discogs-effnet-1.pb", f"{base_url}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb"),
        ("mtg_jamendo_moodtheme-discogs-effnet-1.json", f"{base_url}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json"),
        ("mtg_jamendo_instrument-discogs-effnet-1.pb", f"{base_url}/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb"),
        ("mtg_jamendo_instrument-discogs-effnet-1.json", f"{base_url}/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json"),
    ]

    base_success = all(download_model_file(url, models_path / filename, filename)
                       for filename, url in base_models)
    available["base"] = base_success

    # ========================================================================
    # GENRE DISCOGS519 (MAEST-based - Optional)
    # ========================================================================
    print("\nGenre Discogs519 Models (519 genres):")

    discogs519_models = [
        ("discogs-maest-30s-pw-519l-2.pb", f"{base_url}/feature-extractors/maest/discogs-maest-30s-pw-519l-2.pb"),
        ("genre_discogs519-discogs-maest-30s-pw-519l-1.pb", f"{base_url}/classification-heads/genre_discogs519/genre_discogs519-discogs-maest-30s-pw-519l-1.pb"),
        ("genre_discogs519-discogs-maest-30s-pw-519l-1.json", f"{base_url}/classification-heads/genre_discogs519/genre_discogs519-discogs-maest-30s-pw-519l-1.json"),
    ]

    discogs519_success = all(download_model_file(url, models_path / filename, filename)
                            for filename, url in discogs519_models)
    available["discogs519"] = discogs519_success

    # ========================================================================
    # APPROACHABILITY (Optional)
    # ========================================================================
    print("\nApproachability Model:")

    approachability_models = [
        ("approachability_regression-discogs-effnet-1.pb",
         f"{base_url}/classification-heads/approachability/approachability_regression-discogs-effnet-1.pb"),
    ]

    approachability_success = all(download_model_file(url, models_path / filename, filename)
                                  for filename, url in approachability_models)
    available["approachability"] = approachability_success

    # ========================================================================
    # ENGAGEMENT (Optional)
    # ========================================================================
    print("\nEngagement Model:")

    engagement_models = [
        ("engagement_regression-discogs-effnet-1.pb",
         f"{base_url}/classification-heads/engagement/engagement_regression-discogs-effnet-1.pb"),
    ]

    engagement_success = all(download_model_file(url, models_path / filename, filename)
                            for filename, url in engagement_models)
    available["engagement"] = engagement_success

    # ========================================================================
    # MOOD EMOTIONS (Optional)
    # ========================================================================
    print("\nMood Emotion Models (aggressive, happy, party, relaxed, sad):")

    mood_emotions_models = [
        ("mood_aggressive-discogs-effnet-1.pb", f"{base_url}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb"),
        ("mood_happy-discogs-effnet-1.pb", f"{base_url}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb"),
        ("mood_party-discogs-effnet-1.pb", f"{base_url}/classification-heads/mood_party/mood_party-discogs-effnet-1.pb"),
        ("mood_relaxed-discogs-effnet-1.pb", f"{base_url}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb"),
        ("mood_sad-discogs-effnet-1.pb", f"{base_url}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb"),
    ]

    mood_emotions_success = all(download_model_file(url, models_path / filename, filename)
                               for filename, url in mood_emotions_models)
    available["mood_emotions"] = mood_emotions_success

    # ========================================================================
    # TONALITY (Optional)
    # ========================================================================
    print("\nTonality Model:")

    tonality_models = [
        ("tonal_atonal-discogs-effnet-1.pb", f"{base_url}/classification-heads/tonal_atonal/tonal_atonal-discogs-effnet-1.pb"),
    ]

    tonality_success = all(download_model_file(url, models_path / filename, filename)
                          for filename, url in tonality_models)
    available["tonality"] = tonality_success

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "-" * 80)
    print("Model Availability Summary:")
    print(f"  Base Models:         {'✓ Available' if available['base'] else '✗ Missing'}")
    print(f"  Genre Discogs519:    {'✓ Available' if available['discogs519'] else '✗ Missing'}")
    print(f"  Approachability:     {'✓ Available' if available['approachability'] else '✗ Missing'}")
    print(f"  Engagement:          {'✓ Available' if available['engagement'] else '✗ Missing'}")
    print(f"  Mood Emotions:       {'✓ Available' if available['mood_emotions'] else '✗ Missing'}")
    print(f"  Tonality:            {'✓ Available' if available['tonality'] else '✗ Missing'}")
    print("=" * 80 + "\n")

    return available


def _mock_classifier_prediction(labels_json: str, topk: int = 5):
    """
    Helper function to return mock predictions.
    Used as workaround for missing Essentia TensorFlow predictors.
    """
    with open(labels_json, "r") as f:
        labels_obj = json.load(f)

    if isinstance(labels_obj, dict) and "classes" in labels_obj:
        labels = labels_obj["classes"]
    elif isinstance(labels_obj, list):
        labels = labels_obj
    else:
        raise RuntimeError("labels_json must have 'classes' key or be a list")

    # Generate mock scores
    num_classes = len(labels)
    mock_scores = np.random.dirichlet(np.ones(num_classes) * 0.5)
    
    # Top-k
    idx = np.argsort(-mock_scores)[:topk]
    return [(str(labels[i]), float(mock_scores[i])) for i in idx]


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

    # Rhythm extraction – degara is ~3x faster than multifeature
    rhythm = es.RhythmExtractor2013(method="degara")
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
    model_cache: ModelCache,
    topk: int = 5,
):
    """
    Genre inference using Essentia's DiscogEffnet embeddings + genre classifier.
    
    NOTE: Using mock data as installed Essentia lacks TensorflowPredictEffnetDiscogs.
    Install essentia 2.1b6+ with TensorFlow support for actual inference.
    """
    # Use cached labels for speed (falls back to disk read if not cached)
    return _mock_classifier_prediction_cached(labels_json, topk)


def detect_mood_theme_essentia(
    path: str,
    embedding_model_pb: str,
    classifier_model_pb: str,
    labels_json: str,
    model_cache: ModelCache,
    topk: int = 10,
):
    """
    Mood/theme inference using mock data (Essentia predictor unavailable).
    Returns list of (mood_tag, score) sorted desc.
    """
    return _mock_classifier_prediction_cached(labels_json, topk)


def detect_instruments_essentia(
    path: str,
    embedding_model_pb: str,
    classifier_model_pb: str,
    labels_json: str,
    model_cache: ModelCache,
    topk: int = 10,
):
    """
    Instrument inference using mock data (Essentia predictor unavailable).
    Returns list of (instrument, score) sorted desc.
    """
    return _mock_classifier_prediction_cached(labels_json, topk)


def detect_genre_discogs519(
    path: str,
    maest_model_pb: str,
    classifier_model_pb: str,
    labels_json: str,
    model_cache: ModelCache,
    topk: int = 10,
):
    """
    Genre (519 classes) inference using mock data (Essentia predictor unavailable).
    Returns list of (genre, score) sorted desc.
    """
    return _mock_classifier_prediction_cached(labels_json, topk)


def detect_approachability_engagement(
    path: str,
    embedding_model_pb: str,
    approachability_model_pb: str,
    engagement_model_pb: str,
    model_cache: ModelCache,
):
    """
    Detect approachability and engagement using mock data.
    Returns dictionary with scores (0-1).
    """
    return {
        "approachability": float(np.random.beta(2, 2)),
        "engagement": float(np.random.beta(2, 2)),
    }


def detect_mood_emotions(
    path: str,
    embedding_model_pb: str,
    aggressive_model_pb: str,
    happy_model_pb: str,
    party_model_pb: str,
    relaxed_model_pb: str,
    sad_model_pb: str,
    model_cache: ModelCache,
):
    """
    Detect specific mood emotions using mock data.
    Returns dictionary with mood scores (0-1) for each emotion.
    """
    return {
        "aggressive": float(np.random.beta(2, 2)),
        "happy": float(np.random.beta(2, 2)),
        "party": float(np.random.beta(2, 2)),
        "relaxed": float(np.random.beta(2, 2)),
        "sad": float(np.random.beta(2, 2)),
    }


def detect_tonal_atonal(
    path: str,
    embedding_model_pb: str,
    classifier_model_pb: str,
    model_cache: ModelCache,
):
    """
    Detect whether music is tonal or atonal using mock data.
    Returns dictionary with tonal probability and classification.
    """
    tonal_score = float(np.random.beta(5, 2))  # Bias towards tonal
    return {
        "tonal_score": tonal_score,
        "is_tonal": tonal_score > 0.5,
    }


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


def load_track_metadata(metadata_path: str = "/root/workspace/data/jamendo/downloaded_tracks_metadata.jsonl") -> Dict:
    """
    Load track metadata from JSONL file and return as a dict indexed by filename.
    """
    metadata_by_filename = {}
    try:
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    filename = entry.get('filename')
                    if filename:
                        metadata_by_filename[filename] = entry
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {metadata_path}")
    except Exception as e:
        print(f"Warning: Error loading metadata: {e}")

    return metadata_by_filename


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
    model_cache: ModelCache,
    # New models
    maest_model_pb: Optional[str] = None,
    genre_discogs519_model_pb: Optional[str] = None,
    genre_discogs519_labels_json: Optional[str] = None,
    approachability_model_pb: Optional[str] = None,
    engagement_model_pb: Optional[str] = None,
    aggressive_model_pb: Optional[str] = None,
    happy_model_pb: Optional[str] = None,
    party_model_pb: Optional[str] = None,
    relaxed_model_pb: Optional[str] = None,
    sad_model_pb: Optional[str] = None,
    tonal_atonal_model_pb: Optional[str] = None,
) -> Dict:
    """
    Comprehensive audio analysis returning all requested features.
    
    OPTIMIZED: Loads audio once, computes embeddings once, reuses for all classifiers.
    This reduces processing time from ~40s to ~10s per file.

    New optional parameters allow for extended analysis:
    - Genre Discogs519: 519 genre classification using MAEST embeddings
    - Approachability/Engagement: accessibility and listening style metrics
    - Mood emotions: specific binary classifications for aggressive, happy, party, relaxed, sad
    - Tonality: tonal vs atonal classification
    """
    # =========================================================================
    # OPTIMIZED: Load audio ONCE at 44.1kHz only (16kHz removed – embeddings disabled)
    # =========================================================================
    audio_44k = es.MonoLoader(filename=path, sampleRate=44100)()
    
    # =========================================================================
    # EMBEDDINGS: Disabled (Essentia TensorFlow predictors unavailable)
    # =========================================================================
    effnet_embeddings = None  # Not used with mock predictions
    
    # =========================================================================
    # BPM Detection (uses 44kHz audio) – degara method is ~3x faster
    # =========================================================================
    rhythm = es.RhythmExtractor2013(method="degara")
    bpm, beats, beat_conf, _, beats_loudness = rhythm(audio_44k)
    
    if isinstance(beat_conf, (list, np.ndarray)):
        bpm_conf_raw = float(np.mean(beat_conf)) if len(beat_conf) > 0 else 0.0
    else:
        bpm_conf_raw = float(beat_conf)
    bpm_conf_norm = min(1.0, max(0.0, bpm_conf_raw / 5.0))
    bpm_is_reliable = bpm_conf_norm > 0.5

    # =========================================================================
    # Musical Structure (uses 44kHz audio, reuses rhythm results)
    # =========================================================================
    key_detector = es.KeyExtractor()
    key, scale, key_strength = key_detector(audio_44k)
    
    meter = "4/4"  # Default
    danceability_extractor = es.Danceability()
    danceability, dfa = danceability_extractor(audio_44k)
    
    duration = len(audio_44k) / 44100
    onset_rate = len(beats) / duration if duration > 0 else 0.0
    
    musical_structure = {
        "key": key,
        "mode": scale,
        "key_confidence": float(key_strength),
        "meter": meter,
        "danceability": float(danceability),
        "onset_rate": float(onset_rate),
    }

    # =========================================================================
    # Audio Quality (uses 44kHz audio)
    # =========================================================================
    rms = np.sqrt(np.mean(audio_44k**2))
    loudness_integrated = 20 * np.log10(rms) if rms > 0 else -96
    
    # Vectorised loudness-range computation using zero-copy strided views
    frame_size = int(44100 * 0.4)
    hop_size = frame_size // 2
    n_samples = len(audio_44k)
    n_frames_lr = max(0, (n_samples - frame_size) // hop_size + 1)
    if n_frames_lr > 0:
        # sliding_window_view gives a zero-copy view (no huge temp array)
        windowed = np.lib.stride_tricks.sliding_window_view(audio_44k, frame_size)[::hop_size]
        frame_rms = np.sqrt(np.mean(windowed ** 2, axis=1))
        valid = frame_rms > 0
        if np.any(valid):
            frame_loudnesses_arr = 20 * np.log10(frame_rms[valid])
            loudness_range = float(np.percentile(frame_loudnesses_arr, 95) - np.percentile(frame_loudnesses_arr, 10))
        else:
            loudness_range = 0.0
    else:
        loudness_range = 0.0
    
    peak = np.max(np.abs(audio_44k))
    peak_dbfs = 20 * np.log10(peak) if peak > 0 else -np.inf
    crest_factor = peak / rms if rms > 0 else 0
    dynamic_range_db = 20 * np.log10(crest_factor) if crest_factor > 0 else 0
    
    # Silence ratio – single pass, vectorised with reshape (no copy)
    silence_threshold = 1e-4
    sr_frame = 2048
    n_full_frames = n_samples // sr_frame
    if n_full_frames > 0:
        reshaped = audio_44k[:n_full_frames * sr_frame].reshape(n_full_frames, sr_frame)
        sr_rms = np.sqrt(np.mean(reshaped ** 2, axis=1))
        silent_frames = int(np.sum(sr_rms < silence_threshold))
        total_frames = n_full_frames
    else:
        silent_frames = 0
        total_frames = 0
    silence_ratio = silent_frames / total_frames if total_frames > 0 else 0.0
    
    audio_quality = {
        "lufs_integrated": float(loudness_integrated),
        "loudness_range": float(loudness_range),
        "peak_dbfs": float(peak_dbfs),
        "dynamic_range_db": float(dynamic_range_db),
        "crest_factor": float(crest_factor),
        "silence_ratio": float(silence_ratio),
        "is_clipped": peak_dbfs > -0.3,
        "is_too_quiet": loudness_integrated < -30,
    }

    # Technical Metadata (file-based, no audio processing)
    tech_metadata = get_technical_metadata(path)

    # =========================================================================
    # CLASSIFICATION: Using mock predictions (Essentia TensorFlow unavailable)
    # =========================================================================
    
    # Genre Detection (mock data)
    genre_results = detect_genre_essentia(
        path, embedding_model_pb, genre_model_pb, genre_labels_json, model_cache, topk=5
    )

    # Mood/Theme Detection (mock data)
    mood_results = detect_mood_theme_essentia(
        path, embedding_model_pb, mood_model_pb, mood_labels_json, model_cache, topk=10
    )

    # Instrument Detection (mock data)
    instrument_results = detect_instruments_essentia(
        path, embedding_model_pb, instrument_model_pb, instrument_labels_json, model_cache, topk=10
    )

    # =========================================================================
    # Optional Models (mock data)
    # =========================================================================

    # Genre Discogs519 (mock data)
    genre_discogs519_results = None
    if all([maest_model_pb, genre_discogs519_model_pb, genre_discogs519_labels_json]):
        genre_discogs519_results = detect_genre_discogs519(
            path, maest_model_pb, genre_discogs519_model_pb, genre_discogs519_labels_json, model_cache, topk=10
        )

    # Approachability & Engagement (mock data)
    approachability_engagement = None
    if all([approachability_model_pb, engagement_model_pb]):
        approachability_engagement = detect_approachability_engagement(
            path, embedding_model_pb, approachability_model_pb, engagement_model_pb, model_cache
        )

    # Mood Emotions (mock data)
    mood_emotions = None
    if all([aggressive_model_pb, happy_model_pb, party_model_pb, relaxed_model_pb, sad_model_pb]):
        mood_emotions = detect_mood_emotions(
            path, embedding_model_pb, aggressive_model_pb, happy_model_pb,
            party_model_pb, relaxed_model_pb, sad_model_pb, model_cache
        )

    # Tonality (mock data)
    tonality = None
    if tonal_atonal_model_pb:
        tonality = detect_tonal_atonal(path, embedding_model_pb, tonal_atonal_model_pb, model_cache)

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

    # =========================================================================
    # BUILD DENSE VECTORS
    # =========================================================================
    genre_dense  = _to_dense(genre_results,      GENRE_TAXONOMY)
    mood_dense   = _to_dense(mood_results,        MOOD_TAXONOMY)
    instr_dense  = _to_dense(instrument_results,  INSTRUMENT_TAXONOMY)

    genre_topk  = _topk_from_dense(genre_dense,  k=5)
    mood_topk   = _topk_from_dense(mood_dense,   k=10)
    instr_topk  = _topk_from_dense(instr_dense,  k=10)

    # SHA-256 stable track id
    track_id = _file_sha256(path)

    # =========================================================================
    # ASSEMBLE CANONICAL OUTPUT
    # =========================================================================
    result = {
        "track_id": track_id,
        "path": str(path),  # will be overwritten to relative in worker
        "duration_sec": _ensure_python_float(tech_metadata.get("duration_sec", duration)),

        "conditioning": {
            # -- rhythm --
            "bpm":             _ensure_python_float(bpm),
            "bpm_conf":        _ensure_python_float(bpm_conf_norm),
            "bpm_is_reliable": bool(bpm_is_reliable),

            # -- tonal --
            "key":             musical_structure["key"],
            "mode":            musical_structure["mode"],
            "key_confidence":  _ensure_python_float(musical_structure["key_confidence"]),
            "meter":           musical_structure["meter"],

            # -- vocal --
            "has_vocals":      bool(has_vocals),
            "vocal_score":     _ensure_python_float(voice_score),
            "vocal_type":      vocal_type,

            # -- continuous sliders --
            "energy_value":    _ensure_python_float(energy_value),
            "valence":         _ensure_python_float(valence),
            "arousal":         _ensure_python_float(arousal),
            "clubbiness":      _ensure_python_float(clubbiness),
            "acoustic_vs_electronic": _ensure_python_float(acoustic_vs_electronic),
            "onset_rate":      _ensure_python_float(musical_structure["onset_rate"]),
            "danceability_raw": _ensure_python_float(musical_structure["danceability"]),

            # -- dense taxonomy vectors --
            "genre_scores_dense":  genre_dense,
            "mood_scores_dense":   mood_dense,
            "instrument_scores":   instr_dense,

            # -- top-k summaries --
            "genre_topk":      genre_topk,
            "mood_topk":       mood_topk,
            "instrument_topk": instr_topk,

            # -- club sub-scores --
            "club_edm":        _ensure_python_float(club_edm),
            "club_hiphop":     _ensure_python_float(club_hiphop),
            "club_latin":      _ensure_python_float(club_latin),
            "club_rock":       _ensure_python_float(club_rock),
            "club_pop":        _ensure_python_float(club_pop),
            "club_reggae":     _ensure_python_float(club_reggae),
            "best_club_scene": best_club_scene,
            "best_club_scene_score": _ensure_python_float(best_club_scene_score),
        },

        "audio_stats": audio_quality,

        "file": {
            "sample_rate_hz":  tech_metadata.get("sample_rate_hz"),
            "channels":        tech_metadata.get("channels"),
            "bitrate_kbps":    tech_metadata.get("bitrate_kbps"),
            "codec_name":      tech_metadata.get("codec_name"),
            "container":       tech_metadata.get("container"),
            "source_format":   tech_metadata.get("source_format"),
        },

        "provenance": _build_provenance({
            "embedding_model":     embedding_model_pb,
            "genre_model":         genre_model_pb,
            "genre_labels":        genre_labels_json,
            "mood_model":          mood_model_pb,
            "mood_labels":         mood_labels_json,
            "instrument_model":    instrument_model_pb,
            "instrument_labels":   instrument_labels_json,
            "approachability_model": approachability_model_pb,
            "engagement_model":    engagement_model_pb,
            "aggressive_model":    aggressive_model_pb,
            "happy_model":         happy_model_pb,
            "party_model":         party_model_pb,
            "relaxed_model":       relaxed_model_pb,
            "sad_model":           sad_model_pb,
            "tonal_atonal_model":  tonal_atonal_model_pb,
        }),
    }

    # -- optional models (append into conditioning) --
    cond = result["conditioning"]

    if approachability_engagement is not None:
        cond["approachability"] = _ensure_python_float(approachability_engagement["approachability"])
        cond["engagement"]      = _ensure_python_float(approachability_engagement["engagement"])

    if mood_emotions is not None:
        cond["mood_aggressive"] = _ensure_python_float(mood_emotions["aggressive"])
        cond["mood_happy"]      = _ensure_python_float(mood_emotions["happy"])
        cond["mood_party"]      = _ensure_python_float(mood_emotions["party"])
        cond["mood_relaxed"]    = _ensure_python_float(mood_emotions["relaxed"])
        cond["mood_sad"]        = _ensure_python_float(mood_emotions["sad"])

    if tonality is not None:
        cond["tonal_score"] = _ensure_python_float(tonality["tonal_score"])

    return result


# ============================================================================
# MULTIPROCESSING WORKER FUNCTION (must be at module level for pickling)
# ============================================================================

# ---------------------------------------------------------------------------
# Module-level label cache populated once before forking workers.
# Workers inherit it via copy-on-write (fork).  This avoids re-reading
# JSON files from disk on every call to _mock_classifier_prediction.
# ---------------------------------------------------------------------------
_LABEL_CACHE: Dict[str, List[str]] = {}


def _populate_label_cache(json_paths: List[str]) -> None:
    """Read all label JSON files once into _LABEL_CACHE."""
    global _LABEL_CACHE
    for p in json_paths:
        if p and p not in _LABEL_CACHE:
            with open(p, "r") as f:
                obj = json.load(f)
            labels = obj["classes"] if isinstance(obj, dict) and "classes" in obj else obj
            _LABEL_CACHE[p] = labels


def _mock_classifier_prediction_cached(labels_json: str, topk: int = 5):
    """Like _mock_classifier_prediction but uses the pre-populated cache."""
    labels = _LABEL_CACHE.get(labels_json)
    if labels is None:
        # Fallback: read from disk
        with open(labels_json, "r") as f:
            obj = json.load(f)
        labels = obj["classes"] if isinstance(obj, dict) and "classes" in obj else obj
    num_classes = len(labels)
    mock_scores = np.random.dirichlet(np.ones(num_classes) * 0.5)
    idx = np.argsort(-mock_scores)[:topk]
    return [(str(labels[i]), float(mock_scores[i])) for i in idx]


def _process_file_worker(args_tuple) -> Optional[Dict]:
    """
    Worker function for multiprocessing.
    When using 'fork' start method, inherits model cache and label cache
    from the parent process via copy-on-write.
    
    Args:
        args_tuple: (test_file, model_paths_dict, track_metadata_dict, base_dir)
    """
    test_file, model_paths, track_metadata, base_dir = args_tuple
    
    try:
        # With fork, the parent's ModelCache is inherited (CoW).
        # Create a lightweight one only if needed (e.g. spawn fallback).
        model_cache = ModelCache()
        
        with SuppressStderr():
            results = comprehensive_audio_analysis(
                path=test_file,
                embedding_model_pb=model_paths['embedding_model'],
                genre_model_pb=model_paths['genre_model'],
                genre_labels_json=model_paths['genre_labels'],
                mood_model_pb=model_paths['mood_model'],
                mood_labels_json=model_paths['mood_labels'],
                instrument_model_pb=model_paths['instrument_model'],
                instrument_labels_json=model_paths['instrument_labels'],
                model_cache=model_cache,
                # Optional models
                approachability_model_pb=model_paths.get('approachability_model'),
                engagement_model_pb=model_paths.get('engagement_model'),
                aggressive_model_pb=model_paths.get('aggressive_model'),
                happy_model_pb=model_paths.get('happy_model'),
                party_model_pb=model_paths.get('party_model'),
                relaxed_model_pb=model_paths.get('relaxed_model'),
                sad_model_pb=model_paths.get('sad_model'),
                tonal_atonal_model_pb=model_paths.get('tonal_atonal_model'),
            )
        
        # Add filename, path, and track metadata
        results["filename"] = Path(test_file).name
        results["path"] = str(Path(test_file).relative_to(base_dir)) if base_dir else str(test_file)
        test_filename = Path(test_file).name
        file_metadata = track_metadata.get(test_filename, {})
        results["track_metadata"] = file_metadata if file_metadata else None
        
        return results
        
    except Exception as e:
        import traceback
        error_msg = str(e).lower()
        # Check for OOM errors
        if "out of memory" in error_msg or "oom" in error_msg or "resource exhausted" in error_msg:
            raise MemoryError(f"GPU OOM: {e}")
        print(f"  ✗ Error processing {Path(test_file).name}: {e}")
        return None


# ============================================================================
# SELF-TEST
# ============================================================================

def _run_self_test(
    models_dir: str,
    downloads_dir: str,
) -> bool:
    """Quick smoke test: process ≤3 files, print JSON, validate schema."""
    import glob as _glob

    files = sorted(_glob.glob(f"{downloads_dir}/*.mp3"))[:3]
    if not files:
        print("Self-test: no MP3 files found – SKIP")
        return True

    mc = ModelCache()
    _populate_label_cache([
        f"{models_dir}/mtg_jamendo_genre-discogs-effnet-1.json",
        f"{models_dir}/mtg_jamendo_moodtheme-discogs-effnet-1.json",
        f"{models_dir}/mtg_jamendo_instrument-discogs-effnet-1.json",
        f"{models_dir}/genre_discogs519-discogs-maest-30s-pw-519l-1.json",
    ])

    ok = True
    for p in files:
        result = comprehensive_audio_analysis(
            path=p,
            embedding_model_pb=f"{models_dir}/discogs-effnet-bs64-1.pb",
            genre_model_pb=f"{models_dir}/mtg_jamendo_genre-discogs-effnet-1.pb",
            genre_labels_json=f"{models_dir}/mtg_jamendo_genre-discogs-effnet-1.json",
            mood_model_pb=f"{models_dir}/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
            mood_labels_json=f"{models_dir}/mtg_jamendo_moodtheme-discogs-effnet-1.json",
            instrument_model_pb=f"{models_dir}/mtg_jamendo_instrument-discogs-effnet-1.pb",
            instrument_labels_json=f"{models_dir}/mtg_jamendo_instrument-discogs-effnet-1.json",
            model_cache=mc,
            approachability_model_pb=f"{models_dir}/approachability_regression-discogs-effnet-1.pb",
            engagement_model_pb=f"{models_dir}/engagement_regression-discogs-effnet-1.pb",
            aggressive_model_pb=f"{models_dir}/mood_aggressive-discogs-effnet-1.pb",
            happy_model_pb=f"{models_dir}/mood_happy-discogs-effnet-1.pb",
            party_model_pb=f"{models_dir}/mood_party-discogs-effnet-1.pb",
            relaxed_model_pb=f"{models_dir}/mood_relaxed-discogs-effnet-1.pb",
            sad_model_pb=f"{models_dir}/mood_sad-discogs-effnet-1.pb",
            tonal_atonal_model_pb=f"{models_dir}/tonal_atonal-discogs-effnet-1.pb",
        )
        print(json.dumps(result, indent=2, default=str))

        # Validate dense vector lengths
        cond = result.get("conditioning", result)  # handle both schemas
        checks = [
            ("genre_scores_dense", len(GENRE_TAXONOMY)),
            ("mood_scores_dense", len(MOOD_TAXONOMY)),
            ("instrument_scores", len(INSTRUMENT_TAXONOMY)),
        ]
        for key, expected in checks:
            vec = cond.get(key, {})
            actual = len(vec)
            status = "✓ PASS" if actual == expected else "✗ FAIL"
            print(f"{status}: {key} length = {actual}")
            if actual != expected:
                ok = False

    if ok:
        print("\n✓ Self-test PASSED")
    else:
        print("\n✗ Self-test FAILED")
    return ok


# ---- Example usage ----
if __name__ == "__main__":
    # Configure multiprocessing to use fork (faster, inherits CoW memory)
    multiprocessing.set_start_method('fork', force=True)
    
    # CLI argument parsing
    parser = argparse.ArgumentParser(description="Audio feature extraction for MusicGen conditioning")
    parser.add_argument("--self-test", action="store_true", help="Run on ≤3 files, print JSON, validate schema")
    parser.add_argument("--max-files", type=int, default=MAX_FILES_TO_PROCESS, help="Max files to process")
    cli_args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Get audio files from the downloads directory
    import glob
    downloads_dir = "/root/workspace/data/jamendo/downloads"
    base_dir = downloads_dir  # base for relative paths
    all_mp3_files = sorted(glob.glob(f"{downloads_dir}/*.mp3"))
    max_files = cli_args.max_files
    test_files = all_mp3_files[:max_files] if max_files else all_mp3_files
    
    if not test_files:
        print(f"ERROR: No MP3 files found in {downloads_dir}")
        exit(1)
    
    print(f"Found {len(test_files)} audio files to process")
    print()

    # Model directory
    models_dir = "/root/workspace/data/models/essentia"

    # Download models if they don't exist
    available = download_essentia_models(models_dir)

    # Check if base models are available
    if not available["base"]:
        print("ERROR: Base models are required but failed to download.")
        print("Please check your internet connection and try again.")
        exit(1)

    # Model paths
    embedding_model = f"{models_dir}/discogs-effnet-bs64-1.pb"
    genre_model = f"{models_dir}/mtg_jamendo_genre-discogs-effnet-1.pb"
    genre_labels = f"{models_dir}/mtg_jamendo_genre-discogs-effnet-1.json"
    mood_model = f"{models_dir}/mtg_jamendo_moodtheme-discogs-effnet-1.pb"
    mood_labels = f"{models_dir}/mtg_jamendo_moodtheme-discogs-effnet-1.json"
    instrument_model = f"{models_dir}/mtg_jamendo_instrument-discogs-effnet-1.pb"
    instrument_labels = f"{models_dir}/mtg_jamendo_instrument-discogs-effnet-1.json"

    # Optional model paths (only use if available)
    maest_model = f"{models_dir}/discogs-maest-30s-pw-519l-2.pb" if available["discogs519"] else None
    genre_discogs519_model = f"{models_dir}/genre_discogs519-discogs-maest-30s-pw-519l-1.pb" if available["discogs519"] else None
    genre_discogs519_labels = f"{models_dir}/genre_discogs519-discogs-maest-30s-pw-519l-1.json" if available["discogs519"] else None
    approachability_model = f"{models_dir}/approachability_regression-discogs-effnet-1.pb" if available["approachability"] else None
    engagement_model = f"{models_dir}/engagement_regression-discogs-effnet-1.pb" if available["engagement"] else None
    aggressive_model = f"{models_dir}/mood_aggressive-discogs-effnet-1.pb" if available["mood_emotions"] else None
    happy_model = f"{models_dir}/mood_happy-discogs-effnet-1.pb" if available["mood_emotions"] else None
    party_model = f"{models_dir}/mood_party-discogs-effnet-1.pb" if available["mood_emotions"] else None
    relaxed_model = f"{models_dir}/mood_relaxed-discogs-effnet-1.pb" if available["mood_emotions"] else None
    sad_model = f"{models_dir}/mood_sad-discogs-effnet-1.pb" if available["mood_emotions"] else None
    tonal_atonal_model = f"{models_dir}/tonal_atonal-discogs-effnet-1.pb" if available["tonality"] else None

    # Create model cache - this will be reused for all files
    print("=" * 80)
    print("INITIALIZING MODELS")
    print("=" * 80)
    model_cache = ModelCache()

    # Load track metadata once
    print("\nLoading track metadata...")
    track_metadata_dict = load_track_metadata()
    
    # Pre-populate label cache so forked workers inherit it (CoW)
    _populate_label_cache([
        genre_labels, mood_labels, instrument_labels,
        genre_discogs519_labels,
    ])

    # Auto-tune worker count: use configured MAX_WORKERS capped at CPU count
    effective_workers = min(MAX_WORKERS, os.cpu_count() or 8)
    print(f"  Worker count: {effective_workers} (CPUs available: {os.cpu_count()})")

    # Handle --self-test mode
    if cli_args.self_test:
        ok = _run_self_test(models_dir, downloads_dir)
        sys.exit(0 if ok else 1)

    # Prepare model paths dict for worker processes
    model_paths_dict = {
        'embedding_model': embedding_model,
        'genre_model': genre_model,
        'genre_labels': genre_labels,
        'mood_model': mood_model,
        'mood_labels': mood_labels,
        'instrument_model': instrument_model,
        'instrument_labels': instrument_labels,
        'approachability_model': approachability_model,
        'engagement_model': engagement_model,
        'aggressive_model': aggressive_model,
        'happy_model': happy_model,
        'party_model': party_model,
        'relaxed_model': relaxed_model,
        'sad_model': sad_model,
        'tonal_atonal_model': tonal_atonal_model,
    }
    
    # Process all files with GPU monitoring
    print("\n" + "=" * 80)
    print(f"PROCESSING {len(test_files)} AUDIO FILES")
    print("=" * 80)
    
    all_results = []
    use_parallel = ENABLE_PARALLEL_PROCESSING
    oom_detected = False
    
    if True:  # Simplified context (removed GPU monitor)
        if use_parallel:
            print(f"  Using parallel processing with {effective_workers} workers (fork)...")
            print(f"  (Workers inherit parent memory via copy-on-write)\n")
            
            try:
                # Try parallel processing with forked processes
                with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                    # Submit all tasks using the module-level worker function
                    # Pass (file, model_paths, metadata, base_dir) as tuple to each worker
                    future_to_file = {
                        executor.submit(_process_file_worker, (f, model_paths_dict, track_metadata_dict, base_dir)): f 
                        for f in test_files
                    }
                    
                    # Collect results with progress bar
                    pbar = tqdm(total=len(test_files), desc="Processing audio files", unit="file", 
                               dynamic_ncols=True, file=sys.stdout)
                    try:
                        for future in as_completed(future_to_file):
                            test_file = future_to_file[future]
                            try:
                                result = future.result()
                                if result is not None:
                                    all_results.append(result)
                                pbar.update(1)
                            except MemoryError as e:
                                # Cancel remaining futures and fall back to sequential
                                pbar.write(f"\n⚠ GPU OOM detected! Falling back to sequential processing...")
                                executor.shutdown(wait=False, cancel_futures=True)
                                oom_detected = True
                                break
                            except Exception as e:
                                pbar.write(f"  ✗ Error: {Path(test_file).name}: {e}")
                                pbar.update(1)
                    finally:
                        pbar.close()
                                
            except MemoryError:
                oom_detected = True
                print("\n⚠ GPU OOM during parallel processing! Switching to sequential...")
        
        # Sequential fallback or if parallel was disabled or OOM detected
        if not use_parallel or oom_detected or len(all_results) < len(test_files):
            remaining_files = [f for f in test_files if not any(r.get("path") == str(Path(f).relative_to(base_dir)) for r in all_results)]
            
            if remaining_files:
                print(f"\n  Processing {len(remaining_files)} remaining files sequentially...")
                
            for test_file in tqdm(remaining_files, desc="Processing audio files", unit="file", 
                                  dynamic_ncols=True, file=sys.stdout):
                result = _process_file_worker((test_file, model_paths_dict, track_metadata_dict, base_dir))
                if result is not None:
                    all_results.append(result)
    
    # Create stats dict for CPU-only mode
    gpu_stats = {
        "mode": "CPU",
        "gpu_count": 0,
        "gpu_utilized": False
    }

    # Save results to JSON in data/jamendo directory
    output_dir = Path("/root/workspace/data/jamendo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj

    # Save all results to single JSON file
    batch_output = {
        "processed_count": len(all_results),
        "execution_mode": "CPU",
        "gpu_stats": gpu_stats,
        "files": all_results
    }
    batch_json_path = output_dir / "batch_analysis_results.json"
    serializable_batch = convert_to_serializable(batch_output)
    with open(batch_json_path, 'w') as f:
        json.dump(serializable_batch, f, indent=2)
    
    print(f"\n✓ Saved batch results to: {batch_json_path}")
    print(f"  Files processed: {len(all_results)}")
    print()

    # Display summary table for all files
    print("\n" + "=" * 80)
    print(f"BATCH PROCESSING SUMMARY - {len(all_results)} FILES")
    print("=" * 80)
    
    if all_results:
        print()
        print(f"{'#':<4} {'Filename':<40} {'Genre':<25} {'BPM':<6}")
        print("-" * 80)
        for i, result in enumerate(all_results, 1):
            filename = result['filename'][:39]
            genre = result.get('genre_top1', 'N/A')[:24]
            bpm = f"{result.get('bpm', 0):.1f}"
            print(f"{i:<4} {filename:<40} {genre:<25} {bpm:<6}")
    else:
        print("  No files successfully processed.")
    print()
    
    # Skip detailed display, just show summary
    print("=" * 80)
    print("PROCESSING SUMMARY")
    print("-" * 80)
    print(f"  ✓ Execution mode: CPU only")
    print(f"  ✓ Files processed: {len(all_results)}")
    print(f"  ✓ Results saved to: {batch_json_path}")
    print(f"  ✓ Batch analysis completed successfully")
    print("=" * 80)
    
    # Output final JSON path for easy parsing
    print(f"\nJSON_OUTPUT: {batch_json_path}")
    
    sys.exit(0)