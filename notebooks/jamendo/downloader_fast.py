#!/usr/bin/env python3
"""
Fast Jamendo downloader - optimized version with NO per-track API calls.
Reads audiodownload URLs directly from filtered_tracks.json.
"""

import httpx
import json
from pathlib import Path
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import subprocess
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bar...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'tqdm'])
    from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

# Download configuration
DOWNLOAD_DIR = Path("/root/workspace/data/jamendo/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Master overwrite flag - set to True to redownload all files
OVERWRITE = False

# Parallel download workers (can be higher now since no API rate limits!)
MAX_WORKERS = 64

# Download settings
CHUNK_SIZE = 512 * 1024  # 512KB chunks for faster streaming
DOWNLOAD_TIMEOUT = 120  # Timeout for downloads in seconds
MAX_RETRIES = 3  # Retry failed downloads
RETRY_DELAY = 2  # Initial retry delay

# Optional analysis mode (set to True to run ffprobe/ffmpeg after downloads)
ANALYZE = False
ANALYSIS_WORKERS = 4  # Concurrent analysis jobs

# Progress batching (write every N tracks to reduce disk I/O)
PROGRESS_BATCH_SIZE = 50

# Testing: Limit number of tracks (None = all tracks)
LIMIT = None  # Set to 200 for quick testing

# Data files
FULL_TRACK_INFO_FILE = Path("/root/workspace/data/jamendo/full_track_info.json")
FILTERED_OUTPUT_FILE = Path("/root/workspace/data/jamendo/final_filtered_tracks.json")
PROGRESS_FILE = Path("/root/workspace/data/jamendo/download_progress.json")
METADATA_OUTPUT_FILE = Path("/root/workspace/data/jamendo/downloaded_tracks_metadata.jsonl")
FAILED_TRACKS_LOG = Path("/root/workspace/data/jamendo/failed_tracks_log.jsonl")
ANALYSIS_OUTPUT_FILE = Path("/root/workspace/data/jamendo/analysis_results.jsonl")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def format_time(seconds):
    """Format time in a human-readable way (days, hours, minutes, seconds)."""
    if seconds is None or seconds < 0:
        return "?"

    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def format_bytes(bytes_val):
    """Format bytes in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def load_progress():
    """Load download progress from file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {'completed': [], 'failed': []}
    return {'completed': [], 'failed': []}

def save_progress(progress):
    """Save download progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def filter_tracks(tracks):
    """
    Filter tracks based on license requirements:
    - Require ccnc == "false" (allow commercial use)
    - Require ccnd == "false" (allow derivatives)
    """
    filtered = []

    for track in tracks:
        licenses = track.get('licenses', {})

        # Check ccnc and ccnd flags (stored as strings)
        ccnc = licenses.get('ccnc', 'false')
        ccnd = licenses.get('ccnd', 'false')

        # Convert to boolean
        if isinstance(ccnc, str):
            ccnc_bool = ccnc.lower() == 'true'
        else:
            ccnc_bool = bool(ccnc)

        if isinstance(ccnd, str):
            ccnd_bool = ccnd.lower() == 'true'
        else:
            ccnd_bool = bool(ccnd)

        # Only accept if both are false (no restrictions)
        if not ccnc_bool and not ccnd_bool:
            filtered.append(track)

    return filtered

# ============================================================
# FAST DOWNLOAD FUNCTION (NO API CALLS)
# ============================================================

def download_track_fast(track):
    """
    Fast download using audiodownload URL directly from track data.
    NO API calls - URL must be in track['audiodownload'].

    Optimizations:
    - Large chunk streaming (512KB)
    - Incremental sha256 computation while downloading
    - Reuses httpx client per thread
    - No ffprobe/ffmpeg during download

    Returns: dict with status and minimal metadata
    """
    track_id = track.get('id')

    try:
        # Validate audiodownload URL exists
        download_url = track.get('audiodownload', '').strip()

        if not download_url:
            return {
                'status': 'failed',
                'track_id': track_id,
                'error': 'Missing audiodownload URL'
            }

        # Generate filename
        safe_name = "".join(c for c in track.get('name', 'track') if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_artist = "".join(c for c in track.get('artist_name', 'artist') if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{track_id}_{safe_artist}_{safe_name}.mp3"
        filepath = DOWNLOAD_DIR / filename

        # Retry loop
        for attempt in range(MAX_RETRIES):
            try:
                # Create client with connection pooling
                with httpx.Client(
                    timeout=DOWNLOAD_TIMEOUT,
                    follow_redirects=True,
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
                ) as client:

                    start_time = time.time()

                    # Stream download with large chunks
                    sha256_hash = hashlib.sha256()
                    file_size = 0

                    with client.stream("GET", download_url) as response:
                        response.raise_for_status()

                        # Stream to file and compute hash simultaneously
                        with open(filepath, 'wb', buffering=1024*1024) as f:  # 1MB write buffer
                            for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                                f.write(chunk)
                                sha256_hash.update(chunk)
                                file_size += len(chunk)

                    elapsed = time.time() - start_time

                    # Return minimal metadata (no ffprobe/ffmpeg yet)
                    return {
                        'status': 'success',
                        'track_id': track_id,
                        'track_name': track.get('name'),
                        'artist_name': track.get('artist_name'),
                        'artist_id': track.get('artist_id'),
                        'album_name': track.get('album_name'),
                        'album_id': track.get('album_id'),
                        'duration_api': track.get('duration'),
                        'filename': filename,
                        'file_path': str(filepath),
                        'file_size_bytes': file_size,
                        'download_time_sec': elapsed,
                        'sha256': sha256_hash.hexdigest(),
                    }

            except httpx.TimeoutException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    continue
                else:
                    return {
                        'status': 'failed',
                        'track_id': track_id,
                        'error': f'Timeout after {MAX_RETRIES} retries'
                    }

            except httpx.HTTPStatusError as e:
                return {
                    'status': 'failed',
                    'track_id': track_id,
                    'error': f'HTTP {e.response.status_code}'
                }

            except Exception as e:
                return {
                    'status': 'failed',
                    'track_id': track_id,
                    'error': str(e)
                }

    except Exception as e:
        return {
            'status': 'failed',
            'track_id': track_id,
            'error': str(e)
        }

# ============================================================
# OPTIONAL ANALYSIS FUNCTIONS (separate from download)
# ============================================================

def analyze_track_ffprobe(filepath):
    """Extract technical audio metadata using ffprobe."""
    try:
        result = subprocess.run([
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(filepath)
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return {'error': 'ffprobe failed'}

        data = json.loads(result.stdout)

        # Find audio stream
        audio_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break

        if not audio_stream:
            return {'error': 'no audio stream'}

        format_info = data.get('format', {})

        return {
            'sample_rate_hz': int(audio_stream.get('sample_rate', 0)) if audio_stream.get('sample_rate') else None,
            'channels': audio_stream.get('channels'),
            'bitrate': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
            'codec_name': audio_stream.get('codec_name'),
            'duration_sec_actual': float(format_info.get('duration', 0)) if format_info.get('duration') else None,
        }

    except Exception as e:
        return {'error': str(e)}

# ============================================================
# MAIN DOWNLOAD LOGIC
# ============================================================

def main():
    print("="*60)
    print("FAST JAMENDO DOWNLOADER (NO API CALLS)")
    print("="*60)
    print(f"Configuration:")
    print(f"  - OVERWRITE: {OVERWRITE}")
    print(f"  - Workers: {MAX_WORKERS}")
    print(f"  - Chunk size: {format_bytes(CHUNK_SIZE)}")
    print(f"  - Analysis mode: {ANALYZE}")
    print(f"  - Test limit: {LIMIT if LIMIT else 'None (full download)'}")
    print()

    # Load tracks (try filtered first, then full with filtering)
    if FILTERED_OUTPUT_FILE.exists():
        print(f"Loading tracks from {FILTERED_OUTPUT_FILE.name}...")
        with open(FILTERED_OUTPUT_FILE, 'r') as f:
            all_tracks = json.load(f)
        print(f"✓ Loaded {len(all_tracks):,} pre-filtered tracks")
    elif FULL_TRACK_INFO_FILE.exists():
        print(f"Loading tracks from {FULL_TRACK_INFO_FILE.name}...")
        with open(FULL_TRACK_INFO_FILE, 'r') as f:
            raw_tracks = json.load(f)
        print(f"✓ Loaded {len(raw_tracks):,} tracks")

        print("Applying license filters (ccnc=false, ccnd=false)...")
        all_tracks = filter_tracks(raw_tracks)
        print(f"✓ Filtered to {len(all_tracks):,} tracks ({len(all_tracks)/len(raw_tracks)*100:.1f}%)")
    else:
        print(f"✗ Error: No track data found!")
        print(f"  Looked for: {FILTERED_OUTPUT_FILE}")
        print(f"  Looked for: {FULL_TRACK_INFO_FILE}")
        print("Please run the fetch script first.")
        return

    # Apply test limit
    if LIMIT:
        all_tracks = all_tracks[:LIMIT]
        print(f"⚠️  Test mode: limited to {LIMIT} tracks")

    # Validate audiodownload URLs
    print("\nValidating audiodownload URLs...")
    missing_url_count = 0
    valid_tracks = []

    for track in all_tracks:
        if not track.get('audiodownload', '').strip():
            missing_url_count += 1
        else:
            valid_tracks.append(track)

    print(f"  Total tracks: {len(all_tracks):,}")
    print(f"  Missing audiodownload: {missing_url_count:,}")
    print(f"  Valid for download: {len(valid_tracks):,}")

    if missing_url_count > 0:
        print(f"  ⚠️  {missing_url_count} tracks will be skipped (no URL)")

    # Load progress
    progress = load_progress()
    completed_ids = set(progress.get('completed', []))
    failed_ids = set(progress.get('failed', []))

    print(f"\nProgress status:")
    print(f"  Previously completed: {len(completed_ids):,}")
    print(f"  Previously failed: {len(failed_ids):,}")

    # Determine tracks to download
    if OVERWRITE:
        tracks_to_download = valid_tracks
        print(f"  OVERWRITE enabled: downloading all {len(tracks_to_download):,} tracks")
    else:
        tracks_to_download = [t for t in valid_tracks if t.get('id') not in completed_ids]
        skipped = len(valid_tracks) - len(tracks_to_download)
        print(f"  Skipping {skipped:,} completed tracks")
        print(f"  Remaining: {len(tracks_to_download):,}")

    if not tracks_to_download:
        print("\n✓ All tracks already downloaded!")
        return

    # Download phase
    print(f"\n{'='*60}")
    print(f"DOWNLOADING {len(tracks_to_download):,} TRACKS")
    print(f"{'='*60}\n")

    download_stats = {
        'success': 0,
        'failed': 0,
        'total_bytes': 0,
    }

    # Thread-safe locks
    metadata_lock = Lock()
    progress_lock = Lock()
    stats_lock = Lock()

    # Batch progress tracking
    pending_completed = []
    pending_failed = []
    last_progress_write = time.time()

    def flush_progress():
        """Flush pending progress updates to disk."""
        nonlocal pending_completed, pending_failed, last_progress_write

        with progress_lock:
            if pending_completed:
                completed_ids.update(pending_completed)
                progress['completed'] = list(completed_ids)
                pending_completed = []

            if pending_failed:
                failed_ids.update(pending_failed)
                progress['failed'] = list(failed_ids)
                pending_failed = []

            save_progress(progress)
            last_progress_write = time.time()

    def process_result(result):
        """Process download result (thread-safe with batched progress writes)."""
        nonlocal pending_completed, pending_failed, last_progress_write

        if result and result.get('status') == 'success':
            track_id = result.get('track_id')

            # Save metadata immediately
            with metadata_lock:
                with open(METADATA_OUTPUT_FILE, 'a') as f:
                    f.write(json.dumps(result) + '\n')

            # Batch progress updates
            pending_completed.append(track_id)

            # Update stats
            with stats_lock:
                download_stats['success'] += 1
                download_stats['total_bytes'] += result.get('file_size_bytes', 0)

            # Flush progress if batch full or time elapsed
            if len(pending_completed) >= PROGRESS_BATCH_SIZE or (time.time() - last_progress_write) > 5:
                flush_progress()

        else:
            # Track failed
            track_id = result.get('track_id') if result else None

            # Log failure
            with metadata_lock:
                with open(FAILED_TRACKS_LOG, 'a') as f:
                    f.write(json.dumps(result) + '\n')

            # Batch failure tracking
            if track_id:
                pending_failed.append(track_id)

            with stats_lock:
                download_stats['failed'] += 1

            # Flush if needed
            if len(pending_failed) >= PROGRESS_BATCH_SIZE or (time.time() - last_progress_write) > 5:
                flush_progress()

    # Start download with progress bar
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_track = {executor.submit(download_track_fast, track): track for track in tracks_to_download}

            with tqdm(total=len(tracks_to_download),
                     desc="Downloading",
                     unit="track",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}] {postfix}') as pbar:

                for future in as_completed(future_to_track):
                    result = future.result()
                    process_result(result)
                    pbar.update(1)

                    # Calculate ETA
                    elapsed = time.time() - start_time
                    if pbar.n > 0:
                        rate = pbar.n / elapsed
                        remaining = (pbar.total - pbar.n) / rate if rate > 0 else 0
                        eta = format_time(remaining)
                    else:
                        eta = "?"

                    pbar.set_postfix_str(f"✓{download_stats['success']} ✗{download_stats['failed']} | ETA: {eta}")

        # Final progress flush
        flush_progress()

    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted!")
        flush_progress()
        print("Progress saved. Run again to resume.")
        return

    # Calculate stats
    total_time = time.time() - start_time
    tracks_per_sec = download_stats['success'] / total_time if total_time > 0 else 0
    mb_per_sec = (download_stats['total_bytes'] / 1024 / 1024) / total_time if total_time > 0 else 0

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Results:")
    print(f"  ✓ Success: {download_stats['success']:,}")
    print(f"  ✗ Failed: {download_stats['failed']:,}")
    print(f"  Total downloaded: {format_bytes(download_stats['total_bytes'])}")
    print(f"  Time: {format_time(total_time)}")
    print(f"  Throughput: {tracks_per_sec:.2f} tracks/sec, {mb_per_sec:.2f} MB/sec")
    print(f"\nOutput:")
    print(f"  Downloads: {DOWNLOAD_DIR}")
    print(f"  Metadata: {METADATA_OUTPUT_FILE}")
    print(f"  Failed log: {FAILED_TRACKS_LOG}")

    # Optional analysis phase
    if ANALYZE and download_stats['success'] > 0:
        print(f"\n{'='*60}")
        print(f"RUNNING ANALYSIS (ffprobe on {download_stats['success']} files)")
        print(f"{'='*60}\n")

        # Load successful downloads
        successful_files = []
        with open(METADATA_OUTPUT_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if entry.get('status') == 'success':
                            successful_files.append(entry)
                    except:
                        pass

        def analyze_file(entry):
            filepath = Path(entry['file_path'])
            if filepath.exists():
                analysis = analyze_track_ffprobe(filepath)
                return {
                    'track_id': entry['track_id'],
                    'filename': entry['filename'],
                    **analysis
                }
            return None

        with ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            futures = [executor.submit(analyze_file, entry) for entry in successful_files]

            with tqdm(total=len(successful_files), desc="Analyzing", unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        with open(ANALYSIS_OUTPUT_FILE, 'a') as f:
                            f.write(json.dumps(result) + '\n')
                    pbar.update(1)

        print(f"✓ Analysis complete: {ANALYSIS_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
