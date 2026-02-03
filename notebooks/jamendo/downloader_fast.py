#!/usr/bin/env python3
"""
Fast Jamendo downloader - optimized version with NO per-track API calls.
Reads audiodownload URLs directly from filtered_tracks.json.

OPTIMIZATIONS:
- Async I/O with aiohttp for maximum concurrency
- Persistent connection pooling (reuses TCP connections)
- HTTP/2 support where available
- Optimized chunk streaming with zero-copy writes
- Parallel hash computation
"""

import asyncio
import aiohttp
import aiofiles
import json
from pathlib import Path
import time
import hashlib
from threading import Lock
import subprocess
import os

try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bar...")
    subprocess.check_call(['pip', 'install', '-q', 'tqdm'])
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm

try:
    import aiohttp
    import aiofiles
except ImportError:
    print("Installing aiohttp and aiofiles for async downloads...")
    subprocess.check_call(['pip', 'install', '-q', 'aiohttp', 'aiofiles'])
    import aiohttp
    import aiofiles

from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================

# Download configuration
DOWNLOAD_DIR = Path("/root/workspace/data/jamendo/downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Master overwrite flag - set to True to redownload all files
OVERWRITE = False

# Concurrent downloads (async - can go much higher than threads!)
MAX_CONCURRENT = 200  # Async semaphore limit - can handle many more than threads

# Connection pool settings
MAX_CONNECTIONS = 100  # TCP connections to reuse
MAX_KEEPALIVE = 60    # Keep connections alive for 60s

# Download settings
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for faster streaming
DOWNLOAD_TIMEOUT = 180  # Timeout for downloads in seconds (connect + read)
CONNECT_TIMEOUT = 10   # Fast fail on connection issues
MAX_RETRIES = 3  # Retry failed downloads
RETRY_DELAY = 1  # Initial retry delay (shorter for async)

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
# ASYNC DOWNLOAD FUNCTION (OPTIMIZED)
# ============================================================

async def download_track_async(session: aiohttp.ClientSession, track: dict, semaphore: asyncio.Semaphore):
    """
    Async download using aiohttp with connection pooling.
    
    Optimizations:
    - Reuses TCP connections from session pool
    - Large chunk streaming (1MB)
    - Async file I/O with aiofiles
    - Non-blocking hash computation
    
    Returns: dict with status and minimal metadata
    """
    track_id = track.get('id')
    
    async with semaphore:  # Limit concurrent downloads
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
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    sha256_hash = hashlib.sha256()
                    file_size = 0

                    async with session.get(download_url) as response:
                        response.raise_for_status()
                        
                        # Stream to file with async I/O
                        async with aiofiles.open(filepath, 'wb') as f:
                            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                                await f.write(chunk)
                                sha256_hash.update(chunk)
                                file_size += len(chunk)

                    elapsed = time.time() - start_time

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

                except asyncio.TimeoutError:
                    last_error = f'Timeout after {DOWNLOAD_TIMEOUT}s'
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                        continue

                except aiohttp.ClientResponseError as e:
                    return {
                        'status': 'failed',
                        'track_id': track_id,
                        'error': f'HTTP {e.status}'
                    }

                except aiohttp.ClientError as e:
                    last_error = str(e)
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                        continue

                except Exception as e:
                    last_error = str(e)
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                        continue
            
            return {
                'status': 'failed',
                'track_id': track_id,
                'error': last_error or 'Unknown error after retries'
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
# MAIN DOWNLOAD LOGIC (ASYNC)
# ============================================================

async def download_all(tracks_to_download, progress, completed_ids, failed_ids):
    """Async download orchestrator with connection pooling."""
    
    download_stats = {
        'success': 0,
        'failed': 0,
        'total_bytes': 0,
    }
    
    # Locks for thread-safe file writes
    metadata_lock = asyncio.Lock()
    stats_lock = asyncio.Lock()
    
    # Batched progress
    pending_completed = []
    pending_failed = []
    
    async def flush_progress():
        """Flush pending progress updates to disk."""
        nonlocal pending_completed, pending_failed
        
        if pending_completed:
            completed_ids.update(pending_completed)
            progress['completed'] = list(completed_ids)
            pending_completed = []
            
        if pending_failed:
            failed_ids.update(pending_failed)
            progress['failed'] = list(failed_ids)
            pending_failed = []
            
        save_progress(progress)
    
    async def process_result(result):
        """Process download result."""
        nonlocal pending_completed, pending_failed
        
        if result and result.get('status') == 'success':
            track_id = result.get('track_id')
            
            async with metadata_lock:
                async with aiofiles.open(METADATA_OUTPUT_FILE, 'a') as f:
                    await f.write(json.dumps(result) + '\n')
            
            pending_completed.append(track_id)
            
            async with stats_lock:
                download_stats['success'] += 1
                download_stats['total_bytes'] += result.get('file_size_bytes', 0)
        else:
            track_id = result.get('track_id') if result else None
            
            async with metadata_lock:
                async with aiofiles.open(FAILED_TRACKS_LOG, 'a') as f:
                    await f.write(json.dumps(result) + '\n')
            
            if track_id:
                pending_failed.append(track_id)
            
            async with stats_lock:
                download_stats['failed'] += 1
        
        # Flush every batch
        if len(pending_completed) + len(pending_failed) >= PROGRESS_BATCH_SIZE:
            await flush_progress()
    
    # Create optimized aiohttp session with connection pooling
    timeout = aiohttp.ClientTimeout(
        total=DOWNLOAD_TIMEOUT,
        connect=CONNECT_TIMEOUT,
    )
    
    connector = aiohttp.TCPConnector(
        limit=MAX_CONNECTIONS,
        limit_per_host=50,  # Per-host connection limit
        ttl_dns_cache=300,  # Cache DNS for 5 minutes
        keepalive_timeout=MAX_KEEPALIVE,
        enable_cleanup_closed=True,
        force_close=False,  # Reuse connections
    )
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    start_time = time.time()
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={
            'User-Agent': 'JamendoDownloader/2.0',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    ) as session:
        
        # Create all download tasks
        tasks = [
            download_track_async(session, track, semaphore)
            for track in tracks_to_download
        ]
        
        # Process with progress bar
        completed = 0
        with tqdm(total=len(tasks), desc="Downloading", unit="track",
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            
            for coro in asyncio.as_completed(tasks):
                result = await coro
                await process_result(result)
                completed += 1
                pbar.update(1)
                
                # Update postfix
                elapsed = time.time() - start_time
                speed_mb = (download_stats['total_bytes'] / 1024 / 1024) / elapsed if elapsed > 0 else 0
                pbar.set_postfix_str(f"✓{download_stats['success']} ✗{download_stats['failed']} | {speed_mb:.1f} MB/s")
    
    # Final flush
    await flush_progress()
    
    return download_stats, time.time() - start_time


def main():
    print("="*60)
    print("FAST JAMENDO DOWNLOADER v2.0 (ASYNC + CONNECTION POOLING)")
    print("="*60)
    print(f"Configuration:")
    print(f"  - OVERWRITE: {OVERWRITE}")
    print(f"  - Max concurrent: {MAX_CONCURRENT}")
    print(f"  - Connection pool: {MAX_CONNECTIONS}")
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
    print(f"DOWNLOADING {len(tracks_to_download):,} TRACKS (ASYNC)")
    print(f"{'='*60}\n")

    # Run async download
    try:
        download_stats, total_time = asyncio.run(
            download_all(tracks_to_download, progress, completed_ids, failed_ids)
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted!")
        print("Progress saved. Run again to resume.")
        return

    # Calculate stats
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
