# Test: Download a single audio file and extract comprehensive metadata
import httpx
import json
from pathlib import Path
import time
import subprocess
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bar...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'tqdm'])
    from tqdm import tqdm

# Configuration
JAMENDO_CLIENT_ID1 = "48ecf016"
JAMENDO_CLIENT_ID2 = "9a3294a7"
JAMENDO_CLIENT_ID3 = "e19dbc03"
JAMENDO_CLIENT_ID4 = "cbf907a0"

# List of client IDs to rotate through
JAMENDO_CLIENT_IDS = [JAMENDO_CLIENT_ID1, JAMENDO_CLIENT_ID2, JAMENDO_CLIENT_ID3, JAMENDO_CLIENT_ID4]

JAMENDO_API_BASE = "https://api.jamendo.com/v3.0"

# Download configuration
DOWNLOAD_DIR = Path("/root/workspace/data/jamendo/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Master overwrite flag - set to True to redownload all files
OVERWRITE = False

# Parallel download workers (keep low to respect API rate limits)
# WARNING: Jamendo API has rate limits - don't set too high!
MAX_WORKERS = 32

# Rate limiting settings (to avoid API throttling)
REQUEST_DELAY = 0.2  # Minimum seconds between API requests
MAX_RETRIES = 5  # Number of retries for failed requests
RETRY_DELAY = 2  # Initial retry delay in seconds (exponential backoff)

# Data files
CHECKPOINT_FILE = Path("/root/workspace/data/jamendo/tracks_checkpoint.jsonl")
OUTPUT_FILE = Path("/root/workspace/data/jamendo/full_track_info.json")
FILTERED_OUTPUT_FILE = Path("/root/workspace/data/jamendo/final_filtered_tracks.json")
PROGRESS_FILE = Path("/root/workspace/data/jamendo/download_progress.json")
METADATA_OUTPUT_FILE = Path("/root/workspace/data/jamendo/downloaded_tracks_metadata.jsonl")
FAILED_TRACKS_LOG = Path("/root/workspace/data/jamendo/failed_tracks_log.jsonl")

# ============================================================
# RATE LIMITER CLASS
# ============================================================
class RateLimiter:
    """Thread-safe rate limiter to ensure minimum delay between API requests."""

    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.last_request_time = 0
        self.lock = Lock()

    def wait(self):
        """Wait if necessary to maintain minimum interval between requests."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self.last_request_time = time.time()

# Global rate limiter instance
rate_limiter = RateLimiter(REQUEST_DELAY)

def get_client_id():
    """Get a random client ID from the pool to distribute load."""
    return random.choice(JAMENDO_CLIENT_IDS)

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

def extract_audio_metadata_ffprobe(filepath):
    """
    Extract technical audio metadata using ffprobe.
    Returns dict with: sample_rate_hz, channels, bitrate, codec_name, duration_sec_actual
    """
    try:
        # Run ffprobe to get JSON output
        result = subprocess.run([
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(filepath)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {
                'sample_rate_hz': None,
                'channels': None,
                'bitrate': None,
                'codec_name': None,
                'duration_sec_actual': None,
                'error': 'ffprobe failed'
            }
        
        data = json.loads(result.stdout)
        
        # Find audio stream
        audio_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break
        
        if not audio_stream:
            return {
                'sample_rate_hz': None,
                'channels': None,
                'bitrate': None,
                'codec_name': None,
                'duration_sec_actual': None,
                'error': 'no audio stream found'
            }
        
        # Extract format info
        format_info = data.get('format', {})
        
        return {
            'sample_rate_hz': int(audio_stream.get('sample_rate', 0)) if audio_stream.get('sample_rate') else None,
            'channels': audio_stream.get('channels'),
            'bitrate': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
            'codec_name': audio_stream.get('codec_name'),
            'duration_sec_actual': float(format_info.get('duration', 0)) if format_info.get('duration') else None,
        }
    
    except subprocess.TimeoutExpired:
        return {
            'sample_rate_hz': None,
            'channels': None,
            'bitrate': None,
            'codec_name': None,
            'duration_sec_actual': None,
            'error': 'ffprobe timeout'
        }
    except Exception as e:
        return {
            'sample_rate_hz': None,
            'channels': None,
            'bitrate': None,
            'codec_name': None,
            'duration_sec_actual': None,
            'error': str(e)
        }

def analyze_audio_quality(filepath):
    """
    Analyze audio quality metrics using ffmpeg.
    Returns dict with: peak_dbfs, silence_ratio
    """
    try:
        # Use ffmpeg volumedetect filter to get peak volume
        result = subprocess.run([
            'ffmpeg',
            '-i', str(filepath),
            '-af', 'volumedetect',
            '-f', 'null',
            '-'
        ], capture_output=True, text=True, timeout=60)
        
        # Parse output for peak volume
        peak_dbfs = None
        for line in result.stderr.split('\n'):
            if 'max_volume:' in line:
                try:
                    # Extract value like "max_volume: -23.5 dB"
                    peak_dbfs = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
        
        # Use ffmpeg silencedetect filter to detect silence
        result_silence = subprocess.run([
            'ffmpeg',
            '-i', str(filepath),
            '-af', 'silencedetect=noise=-50dB:d=0.1',
            '-f', 'null',
            '-'
        ], capture_output=True, text=True, timeout=60)
        
        # Parse silence detection output
        silence_duration = 0.0
        total_duration = 0.0
        
        for line in result_silence.stderr.split('\n'):
            if 'silence_duration:' in line:
                try:
                    duration = float(line.split('silence_duration:')[1].strip().split()[0])
                    silence_duration += duration
                except:
                    pass
            if 'Duration:' in line and total_duration == 0:
                try:
                    # Extract duration from "Duration: 00:03:45.67"
                    time_str = line.split('Duration:')[1].strip().split(',')[0].strip()
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        hours, minutes, seconds = parts
                        total_duration = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                except:
                    pass
        
        silence_ratio = (silence_duration / total_duration) if total_duration > 0 else 0.0
        
        return {
            'peak_dbfs': peak_dbfs,
            'silence_ratio': silence_ratio
        }
    
    except subprocess.TimeoutExpired:
        return {
            'peak_dbfs': None,
            'silence_ratio': None,
            'error': 'quality analysis timeout'
        }
    except Exception as e:
        return {
            'peak_dbfs': None,
            'silence_ratio': None,
            'error': str(e)
        }

def compute_file_hash(filepath):
    """
    Compute SHA256 hash of file for deduplication.
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        return None

def filter_tracks_by_license(tracks):
    """
    Filter tracks based on Creative Commons license requirements:
    - Require ccnc == "false" (allow commercial use)
    - Require ccnd == "false" (allow derivatives)

    Returns: list of filtered tracks
    """
    filtered = []
    stats = {
        'total': len(tracks),
        'no_cc_license': 0,
        'has_nc': 0,
        'has_nd': 0,
        'passed': 0
    }

    for track in tracks:
        # REQUIREMENT 1: Must have a non-empty license_ccurl (CC license URL)
        # COMMENTED OUT - accepting all tracks regardless of license_ccurl
        # license_ccurl = track.get('license_ccurl', '')
        # has_cc_license = bool(license_ccurl and license_ccurl.strip())
        #
        # if not has_cc_license:
        #     stats['no_cc_license'] += 1
        #     continue

        # REQUIREMENT 2: Get license flags
        licenses = track.get('licenses', {})

        # Check ccnc and ccnd flags (they're stored as strings "true"/"false")
        ccnc = licenses.get('ccnc', 'false')
        ccnd = licenses.get('ccnd', 'false')

        # Convert to boolean (handle both string and bool types)
        if isinstance(ccnc, str):
            ccnc_bool = ccnc.lower() == 'true'
        else:
            ccnc_bool = bool(ccnc)

        if isinstance(ccnd, str):
            ccnd_bool = ccnd.lower() == 'true'
        else:
            ccnd_bool = bool(ccnd)

        # Reject if has NC (non-commercial) or ND (no-derivatives) restrictions
        if ccnc_bool:
            stats['has_nc'] += 1
            continue

        if ccnd_bool:
            stats['has_nd'] += 1
            continue

        # Track passed all filters
        filtered.append(track)
        stats['passed'] += 1

    return filtered, stats

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

def download_track(track):
    """
    Download a single track and extract all metadata with rate limiting and retry logic.
    Returns: dict with status and metadata, or None if failed
    Thread-safe: creates its own httpx client and respects global rate limiter
    """
    track_id = track.get('id')

    # Try with retries and exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            # Create client for this thread
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                # RATE LIMIT: Wait before making API request
                rate_limiter.wait()

                # Get download URL from API (using random client ID from pool)
                client_id = get_client_id()
                response = client.get(f"{JAMENDO_API_BASE}/tracks/", params={
                    "client_id": client_id,
                    "format": "json",
                    "id": track_id,
                    "audiodownload": "true"
                })
                response.raise_for_status()
                track_data = response.json()

                if not track_data.get("results"):
                    return {'status': 'failed', 'error': 'No results from API', 'track_id': track_id}

                track_info = track_data["results"][0]
                download_url = track_info.get("audiodownload")

                if not download_url:
                    return {'status': 'failed', 'error': 'No download URL', 'track_id': track_id}

                # Generate filename
                safe_name = "".join(c for c in track.get('name', 'track') if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_artist = "".join(c for c in track.get('artist_name', 'artist') if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"{track_id}_{safe_artist}_{safe_name}.mp3"
                filepath = DOWNLOAD_DIR / filename

                # Download the file (no rate limit on download URL, only API calls)
                start_time = time.time()
                with client.stream("GET", download_url) as r:
                    r.raise_for_status()
                    with open(filepath, 'wb') as f:
                        for chunk in r.iter_bytes(chunk_size=8192):
                            f.write(chunk)

                elapsed = time.time() - start_time
                file_size = filepath.stat().st_size

                # Extract metadata
                tech_metadata = extract_audio_metadata_ffprobe(filepath)
                quality_metadata = analyze_audio_quality(filepath)
                file_hash = compute_file_hash(filepath)

                # Compile complete metadata
                complete_metadata = {
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
                    **tech_metadata,
                    **quality_metadata,
                    'sha256': file_hash,
                }

                return complete_metadata

        except httpx.HTTPStatusError as e:
            # Handle rate limiting (429) with exponential backoff
            if e.response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'status': 'failed',
                        'track_id': track_id,
                        'error': f'Rate limited after {MAX_RETRIES} retries'
                    }
            else:
                return {
                    'status': 'failed',
                    'track_id': track_id,
                    'error': f'HTTP {e.response.status_code}: {str(e)}'
                }

        except httpx.TimeoutException:
            # Retry on timeout
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                return {
                    'status': 'failed',
                    'track_id': track_id,
                    'error': f'Timeout after {MAX_RETRIES} retries'
                }

        except Exception as e:
            # For other errors, don't retry
            return {
                'status': 'failed',
                'track_id': track_id,
                'error': str(e)
            }

    # Should not reach here, but just in case
    return {
        'status': 'failed',
        'track_id': track_id,
        'error': 'Failed after all retries'
    }

print("="*60)
print("LOADING AND FILTERING TRACK DATA")
print("="*60)

# Step 1: Load all tracks
if not OUTPUT_FILE.exists():
    print(f"✗ Track data file not found: {OUTPUT_FILE}")
    print("Please run the fetch script first to generate full_track_info.json")
    test_track = None
else:
    print(f"Loading tracks from: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'r') as f:
        all_tracks = json.load(f)
    print(f"✓ Loaded {len(all_tracks)} tracks")

    # Step 2: Filter tracks by license requirements
    print("\nApplying license filters:")
    print("  - Require: ccnc = false (allow commercial use)")
    print("  - Require: ccnd = false (allow derivatives)")

    filtered_tracks, filter_stats = filter_tracks_by_license(all_tracks)

    print(f"\n" + "="*60)
    print("FILTER RESULTS")
    print("="*60)
    print(f"  Original track count:        {filter_stats['total']:,}")
    print(f"  Rejected (no CC license):    {filter_stats['no_cc_license']:,}")
    print(f"  Rejected (NC - no commercial): {filter_stats['has_nc']:,}")
    print(f"  Rejected (ND - no derivatives): {filter_stats['has_nd']:,}")
    print(f"  {'─'*58}")
    print(f"  ✓ Tracks passing filters:    {filter_stats['passed']:,}")

    # Calculate percentages
    pass_rate = 0.0
    reject_rate = 0.0
    if filter_stats['total'] > 0:
        pass_rate = (filter_stats['passed'] / filter_stats['total']) * 100
        reject_rate = 100 - pass_rate
        print(f"\n  Pass rate: {pass_rate:.1f}% | Reject rate: {reject_rate:.1f}%")

    # Show total rejected
    total_rejected = filter_stats['no_cc_license'] + filter_stats['has_nc'] + filter_stats['has_nd']
    print(f"  Total rejected: {total_rejected:,} tracks")

    # Step 3: Save filtered tracks
    if filtered_tracks:
        print(f"\nSaving filtered tracks to: {FILTERED_OUTPUT_FILE}")
        FILTERED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FILTERED_OUTPUT_FILE, 'w') as f:
            json.dump(filtered_tracks, f, indent=2)
        print(f"✓ Saved {len(filtered_tracks)} filtered tracks")

        # Show sample of filtered tracks
        print(f"\n" + "="*60)
        print("SAMPLE OF FILTERED TRACKS (first 5)")
        print("="*60)
        for i, track in enumerate(filtered_tracks[:5], 1):
            licenses = track.get('licenses', {})
            print(f"{i}. ID: {track.get('id')} | {track.get('artist_name')} - {track.get('name')}")
            print(f"   License: ccnc={licenses.get('ccnc')}, ccnd={licenses.get('ccnd')}, ccsa={licenses.get('ccsa')}")
            print(f"   Duration: {track.get('duration')}s | Album: {track.get('album_name')}")
            if i < 5:
                print()

        # Proceed to download all filtered tracks
        print(f"\n{'='*60}")
        print("STARTING FULL DOWNLOAD PROCESS")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  - OVERWRITE mode: {OVERWRITE}")
        print(f"  - Parallel workers: {MAX_WORKERS}")
        print(f"  - API clients: {len(JAMENDO_CLIENT_IDS)} client IDs (load balancing)")
        print(f"  - Rate limiting: {REQUEST_DELAY}s minimum delay between API requests")
        print(f"  - Max retries: {MAX_RETRIES} (with exponential backoff)")
        print(f"  - Download directory: {DOWNLOAD_DIR}")
        print(f"  - Progress file: {PROGRESS_FILE}")
        print(f"  - Metadata output: {METADATA_OUTPUT_FILE}")

        # Load progress
        progress = load_progress()
        completed_ids = set(progress.get('completed', []))
        failed_ids = set(progress.get('failed', []))

        print(f"\nProgress status:")
        print(f"  - Previously completed: {len(completed_ids)}")
        print(f"  - Previously failed: {len(failed_ids)}")

        # Determine tracks to download
        if OVERWRITE:
            tracks_to_download = filtered_tracks
            print(f"  - OVERWRITE enabled: downloading all {len(tracks_to_download)} tracks")
        else:
            tracks_to_download = [t for t in filtered_tracks if t.get('id') not in completed_ids]
            skipped_count = len(filtered_tracks) - len(tracks_to_download)
            print(f"  - Skipping {skipped_count} already completed tracks")
            print(f"  - Remaining to download: {len(tracks_to_download)}")

        if not tracks_to_download:
            print("\n✓ All tracks already downloaded!")
        else:
            # Start downloading with parallel workers
            print(f"\n{'='*60}")
            print(f"DOWNLOADING {len(tracks_to_download)} TRACKS")
            print(f"{'='*60}")
            print(f"Using {MAX_WORKERS} parallel workers with rate limiting")
            print(f"Estimated time: ~{len(tracks_to_download) * REQUEST_DELAY / 60:.1f} minutes (API calls only)\n")

            download_stats = {
                'success': 0,
                'failed': 0,
                'clipping': 0,
                'high_silence': 0
            }

            # Thread-safe locks for writing
            metadata_lock = Lock()
            progress_lock = Lock()
            stats_lock = Lock()

            def process_result(result):
                """Process completed download result (thread-safe)"""
                if result and result.get('status') == 'success':
                    track_id = result.get('track_id')

                    # Save metadata to JSONL file (thread-safe)
                    with metadata_lock:
                        with open(METADATA_OUTPUT_FILE, 'a') as f:
                            f.write(json.dumps(result) + '\n')

                    # Update progress (thread-safe)
                    with progress_lock:
                        if track_id not in completed_ids:
                            completed_ids.add(track_id)
                            progress['completed'] = list(completed_ids)
                            if track_id in failed_ids:
                                failed_ids.remove(track_id)
                                progress['failed'] = list(failed_ids)
                            save_progress(progress)

                    # Update stats (thread-safe)
                    with stats_lock:
                        download_stats['success'] += 1
                        if result.get('peak_dbfs') is not None and result.get('peak_dbfs') > -1.0:
                            download_stats['clipping'] += 1
                        if result.get('silence_ratio') is not None and result.get('silence_ratio') > 0.20:
                            download_stats['high_silence'] += 1

                else:
                    # Track failed
                    track_id = result.get('track_id') if result else None

                    # Log failure details (thread-safe)
                    with metadata_lock:
                        with open(FAILED_TRACKS_LOG, 'a') as f:
                            f.write(json.dumps(result) + '\n')

                    with stats_lock:
                        download_stats['failed'] += 1

                    # Update failed list (thread-safe)
                    if track_id:
                        with progress_lock:
                            if track_id not in failed_ids:
                                failed_ids.add(track_id)
                                progress['failed'] = list(failed_ids)
                                save_progress(progress)

            try:
                # Use ThreadPoolExecutor for parallel downloads
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Submit all download tasks
                    future_to_track = {executor.submit(download_track, track): track for track in tracks_to_download}

                    # Process completed downloads with progress bar with enhanced time display
                    with tqdm(total=len(tracks_to_download),
                             desc="Downloading",
                             unit="track",
                             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}] {postfix}') as pbar:

                        # Store start time for custom remaining calculation
                        start_time = time.time()

                        for future in as_completed(future_to_track):
                            result = future.result()
                            process_result(result)
                            pbar.update(1)

                            # Calculate remaining time with better formatting
                            elapsed = time.time() - start_time
                            if pbar.n > 0:
                                rate = pbar.n / elapsed
                                remaining_items = pbar.total - pbar.n
                                remaining_seconds = remaining_items / rate if rate > 0 else 0
                                remaining_fmt = format_time(remaining_seconds)
                            else:
                                remaining_fmt = "?"

                            # Update progress bar postfix with stats and custom time
                            pbar.set_postfix_str(f"✓{download_stats['success']} ✗{download_stats['failed']} | ETA: {remaining_fmt}")

            except KeyboardInterrupt:
                print("\n\n⚠️  Download interrupted by user!")
                print("Progress has been saved. Run the script again to resume.")

            # Final summary
            print(f"\n{'='*60}")
            print("DOWNLOAD COMPLETE")
            print(f"{'='*60}")
            print(f"Results:")
            print(f"  ✓ Successfully downloaded: {download_stats['success']}")
            print(f"  ✗ Failed: {download_stats['failed']}")
            print(f"  ⚠️  Files with clipping: {download_stats['clipping']}")
            print(f"  ⚠️  Files with high silence: {download_stats['high_silence']}")
            print(f"\nOutput files:")
            print(f"  - Downloads: {DOWNLOAD_DIR}")
            print(f"  - Metadata: {METADATA_OUTPUT_FILE}")
            print(f"  - Progress: {PROGRESS_FILE}")
            print(f"  - Filtered tracks: {FILTERED_OUTPUT_FILE}")

    else:
        print("✗ No tracks passed the filters")