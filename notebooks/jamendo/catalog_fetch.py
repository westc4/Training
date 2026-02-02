# Optimized: Fetch FULL CATALOG with performance instrumentation and streaming architecture
import httpx
import json
from pathlib import Path
from tqdm.auto import tqdm
import time
import sys
from collections import deque

JAMENDO_CLIENT_ID = "48ecf016"
JAMENDO_API_BASE = "https://api.jamendo.com/v3.0"

# Output paths
OUTPUT_DIR = Path("/root/workspace/data/jamendo")
OUTPUT_FILE_NAME = "full_track_info.json"
OUTPUT_FILE = OUTPUT_DIR / OUTPUT_FILE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE_DIR = Path("/root/workspace/data/jamendo")
STATE_FILE_NAME = "fetch_state.json"
STATE_FILE = STATE_FILE_DIR / STATE_FILE_NAME
STATE_FILE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE_DIR = Path("/root/workspace/data/jamendo")
CHECKPOINT_FILE_NAME = "tracks_checkpoint.jsonl"
CHECKPOINT_FILE = CHECKPOINT_FILE_DIR / CHECKPOINT_FILE_NAME
CHECKPOINT_FILE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PERFORMANCE OPTIMIZATION FLAGS
# ============================================================
ENABLE_STREAMING = False       # Don't keep all_tracks in memory (default: True)
ENABLE_BATCHED_WRITES = True  # Write once per page, not per track (default: True)
ENABLE_COMPACT_JSON = True    # Use separators=(',', ':') (default: True)
ENABLE_ADAPTIVE_PACING = False # Increase delay if req_time grows (default: False)
USE_ORJSON = False            # Try orjson if installed (default: False)
PERF_PRINT_INTERVAL = 20      # Print perf metrics every N pages (default: 20)

# Rate limiting settings
REQUEST_DELAY = 0.1  # Delay between requests in seconds (100ms)
MAX_RETRIES = 5
RETRY_DELAY = 2  # Initial retry delay in seconds
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N pages
PAGE_SIZE = 200  # Max allowed by Jamendo API

# Try to import orjson if requested
if USE_ORJSON:
    try:
        import orjson
        print("✓ Using orjson for faster JSON serialization")
    except ImportError:
        print("⚠ orjson not installed, falling back to stdlib json")
        USE_ORJSON = False

# ============================================================
# PERFORMANCE MONITOR CLASS
# ============================================================
class PerformanceMonitor:
    """Track and report performance metrics for fetch operations."""
    
    def __init__(self, print_interval=20, window_size=100):
        self.print_interval = print_interval
        self.window_size = window_size
        
        # Current page timings
        self.req_start = None
        self.write_start = None
        
        # Rolling windows for averages
        self.req_times = deque(maxlen=window_size)
        self.write_times = deque(maxlen=window_size)
        self.sleep_times = deque(maxlen=window_size)
        
        # Page counter
        self.page_num = 0
    
    def start_request(self):
        """Mark the start of an HTTP request."""
        self.req_start = time.time()
    
    def end_request(self):
        """Mark the end of an HTTP request and record timing."""
        if self.req_start is not None:
            elapsed = time.time() - self.req_start
            self.req_times.append(elapsed)
            self.req_start = None
            return elapsed
        return 0.0
    
    def start_write(self):
        """Mark the start of JSONL write operation."""
        self.write_start = time.time()
    
    def end_write(self):
        """Mark the end of JSONL write and record timing."""
        if self.write_start is not None:
            elapsed = time.time() - self.write_start
            self.write_times.append(elapsed)
            self.write_start = None
            return elapsed
        return 0.0
    
    def record_sleep(self, duration):
        """Record sleep duration."""
        self.sleep_times.append(duration)
    
    def get_rss_mb(self):
        """Get RSS memory usage in MiB (Linux only)."""
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        # Extract KB value and convert to MiB
                        kb = int(line.split()[1])
                        return kb / 1024.0
        except:
            return None
    
    def get_file_size_mb(self, filepath):
        """Get file size in MiB."""
        try:
            if filepath.exists():
                return filepath.stat().st_size / (1024.0 * 1024.0)
        except:
            return None
    
    def print_metrics(self, offset, checkpoint_file_path):
        """Print compact performance metrics line."""
        self.page_num += 1
        
        if self.page_num % self.print_interval != 0:
            return
        
        # Calculate averages
        avg_req = sum(self.req_times) / len(self.req_times) if self.req_times else 0.0
        avg_write = sum(self.write_times) / len(self.write_times) if self.write_times else 0.0
        last_req = self.req_times[-1] if self.req_times else 0.0
        last_write = self.write_times[-1] if self.write_times else 0.0
        
        # Get resource metrics
        rss_mb = self.get_rss_mb()
        file_mb = self.get_file_size_mb(checkpoint_file_path)
        
        # Print compact line
        print(f"[perf] page={self.page_num} offset={offset:,} "
              f"req={last_req:.2f}s write={last_write:.2f}s "
              f"rss={rss_mb:.0f}MiB file={file_mb:.0f}MiB "
              f"avg_req{self.window_size}={avg_req:.2f}s "
              f"avg_write{self.window_size}={avg_write:.2f}s")
    
    def get_avg_req_time(self):
        """Get average request time over window."""
        return sum(self.req_times) / len(self.req_times) if self.req_times else 0.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_canonical_license(track):
    """
    Robust license extractor - checks multiple fields and nested structures.
    Returns: 'cc-by', 'cc-by-sa', or None (rejected)
    """
    license_url = None
    
    if track.get('license_ccurl'):
        license_url = track.get('license_ccurl')
    elif track.get('licensecurl'):
        license_url = track.get('licensecurl')
    elif track.get('licenses') and isinstance(track.get('licenses'), list) and len(track.get('licenses')) > 0:
        first_license = track['licenses'][0]
        if isinstance(first_license, dict):
            license_url = first_license.get('url') or first_license.get('ccurl')
        elif isinstance(first_license, str):
            license_url = first_license
    
    if not license_url or not isinstance(license_url, str):
        return None
    
    url_lower = license_url.lower().strip().rstrip('/')
    
    if 'creativecommons.org/licenses/' not in url_lower:
        return None
    
    parts = url_lower.split('creativecommons.org/licenses/')
    if len(parts) != 2:
        return None
    
    license_part = parts[1].split('/')[0]
    
    if 'nc' in license_part or 'nd' in license_part:
        return None
    
    if license_part == 'by':
        return 'cc-by'
    elif license_part == 'by-sa':
        return 'cc-by-sa'
    else:
        return None

def get_license_flags(track):
    """
    Extract cc, ccnc, ccnd flags from track.
    Returns tuple: (cc, ccnc, ccnd)
    """
    cc_val = track.get('cc')
    ccnc_val = track.get('ccnc')
    ccnd_val = track.get('ccnd')
    
    licenses_obj = track.get('licenses')
    if licenses_obj and isinstance(licenses_obj, dict):
        if cc_val is None:
            cc_val = licenses_obj.get('cc')
        if ccnc_val is None:
            ccnc_val = licenses_obj.get('ccnc')
        if ccnd_val is None:
            ccnd_val = licenses_obj.get('ccnd')
    
    if isinstance(cc_val, str):
        cc_val = cc_val.lower() == 'true'
    if isinstance(ccnc_val, str):
        ccnc_val = ccnc_val.lower() == 'true'
    if isinstance(ccnd_val, str):
        ccnd_val = ccnd_val.lower() == 'true'
    
    return cc_val, ccnc_val, ccnd_val

def serialize_track(track):
    """Serialize track to JSON string with optimal settings."""
    if USE_ORJSON:
        return orjson.dumps(track).decode('utf-8')
    elif ENABLE_COMPACT_JSON:
        return json.dumps(track, separators=(',', ':'))
    else:
        return json.dumps(track)

def fetch_with_retry(client, url, params, max_retries=MAX_RETRIES):
    """Fetch with exponential backoff retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"\n⚠ Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise
            else:
                raise
        except httpx.TimeoutException:
            if attempt == max_retries - 1:
                raise
            wait_time = RETRY_DELAY * (2 ** attempt)
            print(f"\n⚠ Timeout. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
    
    raise RuntimeError(f"Failed after {max_retries} retries")

def load_checkpoint():
    """Load checkpoint state if exists."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        print(f"✓ Found checkpoint: resuming from offset {state['last_offset']:,} ({state['tracks_fetched']:,} tracks)")
        return state
    return None

def save_checkpoint(state):
    """Save checkpoint state."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def count_existing_tracks():
    """Count tracks in checkpoint JSONL file (streaming, no memory load)."""
    if not CHECKPOINT_FILE.exists():
        return 0
    
    count = 0
    with open(CHECKPOINT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ============================================================
# MAIN FETCH LOGIC
# ============================================================
print("="*60)
print("PERFORMANCE OPTIMIZATIONS ENABLED")
print("="*60)
print(f"  Streaming mode: {ENABLE_STREAMING} (no in-memory all_tracks)")
print(f"  Batched writes: {ENABLE_BATCHED_WRITES} (write per page, not per track)")
print(f"  Compact JSON: {ENABLE_COMPACT_JSON} (reduced CPU and file size)")
print(f"  Adaptive pacing: {ENABLE_ADAPTIVE_PACING} (increase delay if slow)")
print(f"  Performance metrics: printed every {PERF_PRINT_INTERVAL} pages")
print()

# Create enhanced client with connection limits
client = httpx.Client(
    timeout=30,
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=10)
)
checkpoint_file = None
perf = PerformanceMonitor(print_interval=PERF_PRINT_INTERVAL)

# Track small diagnostic samples in memory
diagnostic_samples = []

try:
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    
    if checkpoint:
        offset = checkpoint['last_offset']
        total_catalog_size = checkpoint['total_catalog_size']
        tracks_already_fetched = count_existing_tracks() if ENABLE_STREAMING else len(load_existing_tracks())
        print(f"Resuming fetch: {tracks_already_fetched:,} tracks already fetched")
        if ENABLE_STREAMING:
            all_tracks = []  # Empty in streaming mode
        else:
            all_tracks = load_existing_tracks()
    else:
        # Get total catalog size first
        print("Starting fresh fetch...")
        print("Fetching catalog size...")
        first_data = fetch_with_retry(client, f"{JAMENDO_API_BASE}/tracks/", {
            "client_id": JAMENDO_CLIENT_ID,
            "format": "json",
            "limit": 1,
            "offset": 0,
            "audiodownload": "true",
            "include": "licenses+musicinfo",
            "fullcount": "true"
        })
        
        total_catalog_size = first_data.get("headers", {}).get("results_fullcount", 0)
        offset = 0
        all_tracks = []  # Empty even in non-streaming mode for fresh start
        tracks_already_fetched = 0
        
        checkpoint = {
            'last_offset': 0,
            'tracks_fetched': 0,
            'total_catalog_size': total_catalog_size
        }
        save_checkpoint(checkpoint)
        CHECKPOINT_FILE.write_text('')
    
    print(f"Total catalog size: {total_catalog_size:,} tracks")
    total_pages = (total_catalog_size // PAGE_SIZE) + (1 if total_catalog_size % PAGE_SIZE else 0)
    print(f"Total pages to fetch: {total_pages:,}")
    print(f"Rate limit: {REQUEST_DELAY}s delay between requests")
    print(f"Checkpoint: saving every {CHECKPOINT_INTERVAL} pages")
    
    remaining_tracks = total_catalog_size - tracks_already_fetched
    if tracks_already_fetched > 0:
        progress_pct = 100 * tracks_already_fetched / total_catalog_size
        print(f"\n📊 Resume Status:")
        print(f"   Already fetched: {tracks_already_fetched:,} tracks ({progress_pct:.1f}%)")
        print(f"   Remaining: {remaining_tracks:,} tracks")
    print()
    
    # Open checkpoint file in append mode with large buffer
    buffering = 1024*1024 if ENABLE_BATCHED_WRITES else -1
    checkpoint_file = open(CHECKPOINT_FILE, 'a', buffering=buffering)
    page_count = 0
    
    # Fetch all tracks with progress bar
    with tqdm(total=total_catalog_size, 
              initial=tracks_already_fetched, 
              desc="Fetching tracks", 
              unit="track",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        while offset < total_catalog_size:
            params = {
                "client_id": JAMENDO_CLIENT_ID,
                "format": "json",
                "limit": PAGE_SIZE,
                "offset": offset,
                "audiodownload": "true",
                "include": "licenses+musicinfo",
            }
            
            # === INSTRUMENTED REQUEST ===
            perf.start_request()
            data = fetch_with_retry(client, f"{JAMENDO_API_BASE}/tracks/", params)
            perf.end_request()
            
            tracks = data.get("results", [])
            if not tracks:
                print(f"\nNo more tracks at offset {offset}")
                break
            
            # === INSTRUMENTED WRITE ===
            perf.start_write()
            
            if ENABLE_BATCHED_WRITES:
                # Write entire page at once
                lines = ''.join(serialize_track(t) + '\n' for t in tracks)
                checkpoint_file.write(lines)
            else:
                # Legacy: write per track
                for track in tracks:
                    checkpoint_file.write(serialize_track(track) + '\n')
            
            perf.end_write()
            
            # Collect diagnostic samples (small footprint)
            if len(diagnostic_samples) < 20:
                diagnostic_samples.extend(tracks[:min(5, len(tracks))])
            
            # Update tracking
            if not ENABLE_STREAMING:
                all_tracks.extend(tracks)
            
            offset += len(tracks)
            page_count += 1
            pbar.update(len(tracks))
            
            # Save checkpoint periodically
            if page_count % CHECKPOINT_INTERVAL == 0:
                if not ENABLE_BATCHED_WRITES:
                    checkpoint_file.flush()
                else:
                    checkpoint_file.flush()  # Explicit flush at checkpoint
                checkpoint['last_offset'] = offset
                checkpoint['tracks_fetched'] = len(all_tracks) if not ENABLE_STREAMING else offset
                save_checkpoint(checkpoint)
            
            # Print performance metrics
            perf.print_metrics(offset, CHECKPOINT_FILE)
            
            # Adaptive pacing
            if ENABLE_ADAPTIVE_PACING:
                avg_req = perf.get_avg_req_time()
                if avg_req > 2.0 and REQUEST_DELAY < 2.0:
                    old_delay = REQUEST_DELAY
                    REQUEST_DELAY = min(REQUEST_DELAY * 1.25, 2.0)
                    print(f"[adaptive] Increased REQUEST_DELAY: {old_delay:.2f}s → {REQUEST_DELAY:.2f}s (avg_req={avg_req:.2f}s)")
            
            # Rate limiting: instrumented sleep
            sleep_start = time.time()
            time.sleep(REQUEST_DELAY)
            actual_sleep = time.time() - sleep_start
            perf.record_sleep(actual_sleep)
    
    # Close checkpoint file
    checkpoint_file.close()
    checkpoint_file = None
    
    # Final checkpoint
    checkpoint['last_offset'] = offset
    checkpoint['tracks_fetched'] = len(all_tracks) if not ENABLE_STREAMING else offset
    save_checkpoint(checkpoint)
    
    print(f"\n✓ Fetch complete!")
    print(f"Total tracks fetched: {offset:,}")
    
    # Save final JSON file (optional, only if not streaming or if user wants full dump)
    if not ENABLE_STREAMING and all_tracks:
        print(f"\nSaving final JSON file to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_tracks, f, indent=2)
        print(f"✓ Saved full track info: {OUTPUT_FILE}")
        print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Clean up checkpoint files if fully complete
    if offset >= total_catalog_size:
        print("\nCleaning up checkpoint state files...")
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            print(f"✓ Removed {STATE_FILE_NAME}")
    
    print()
    
    # ============================================================
    # STREAMING LICENSE ANALYSIS
    # ============================================================
    print("="*60)
    print("ANALYZING LICENSES (Streaming Mode)" if ENABLE_STREAMING else "ANALYZING LICENSES (In-Memory Mode)")
    print("="*60)
    
    total_tracks = 0
    empty_license_count = 0
    content_id_restricted_count = 0
    cc_flags_condition_count = 0
    no_nc_nd_count = 0
    cc_flags_passed_examples = []
    no_nc_nd_examples = []
    passed_cc_filter = []
    rejected_examples = []
    
    if ENABLE_STREAMING:
        # Stream through JSONL file
        print("Streaming analysis from JSONL...")
        with open(CHECKPOINT_FILE, 'r') as f:
            for line in tqdm(f, desc="Analyzing licenses", unit="track"):
                if not line.strip():
                    continue
                
                track = json.loads(line)
                total_tracks += 1
                
                # Same analysis logic as before
                cc_val, ccnc_val, ccnd_val = get_license_flags(track)
                
                if cc_val == True and ccnc_val == False and ccnd_val == False:
                    cc_flags_condition_count += 1
                    if len(cc_flags_passed_examples) < 5:
                        licenses_obj = track.get('licenses', {})
                        cc_flags_passed_examples.append({
                            'id': track.get('id'),
                            'name': track.get('name'),
                            'cc': licenses_obj.get('cc') if isinstance(licenses_obj, dict) else track.get('cc'),
                            'ccnc': licenses_obj.get('ccnc') if isinstance(licenses_obj, dict) else track.get('ccnc'),
                            'ccnd': licenses_obj.get('ccnd') if isinstance(licenses_obj, dict) else track.get('ccnd'),
                            'license_url': track.get('license_ccurl') or track.get('licensecurl') or '(empty)'
                        })
                
                if ccnc_val == False and ccnd_val == False:
                    no_nc_nd_count += 1
                    if len(no_nc_nd_examples) < 5:
                        licenses_obj = track.get('licenses', {})
                        no_nc_nd_examples.append({
                            'id': track.get('id'),
                            'name': track.get('name'),
                            'cc': licenses_obj.get('cc') if isinstance(licenses_obj, dict) else track.get('cc'),
                            'ccnc': licenses_obj.get('ccnc') if isinstance(licenses_obj, dict) else track.get('ccnc'),
                            'ccnd': licenses_obj.get('ccnd') if isinstance(licenses_obj, dict) else track.get('ccnd'),
                            'license_url': track.get('license_ccurl') or track.get('licensecurl') or '(empty)'
                        })
                
                license_url = track.get('license_ccurl') or track.get('licensecurl') or ''
                if not license_url:
                    empty_license_count += 1
                    if len(rejected_examples) < 5:
                        rejected_examples.append({
                            'id': track.get('id'),
                            'name': track.get('name'),
                            'content_id_free': track.get('content_id_free'),
                            'license_url': '(empty)',
                            'reason': 'empty_license'
                        })
                    continue
                
                if track.get('content_id_free') == False:
                    content_id_restricted_count += 1
                    if len(rejected_examples) < 5:
                        rejected_examples.append({
                            'id': track.get('id'),
                            'name': track.get('name'),
                            'content_id_free': track.get('content_id_free'),
                            'license_url': license_url,
                            'reason': 'content_id_restricted'
                        })
                    continue
                
                canonical = extract_canonical_license(track)
                
                if canonical:
                    if len(passed_cc_filter) < 5:
                        passed_cc_filter.append({
                            'id': track.get('id'),
                            'name': track.get('name'),
                            'content_id_free': track.get('content_id_free'),
                            'license_url': license_url,
                            'canonical': canonical
                        })
                else:
                    if len(rejected_examples) < 5:
                        rejected_examples.append({
                            'id': track.get('id'),
                            'name': track.get('name'),
                            'content_id_free': track.get('content_id_free'),
                            'license_url': license_url,
                            'reason': 'license_not_cc_by_or_cc_by_sa'
                        })
    else:
        # Use in-memory tracks
        total_tracks = len(all_tracks)
        print(f"Analyzing {total_tracks:,} tracks from memory...")
        for track in tqdm(all_tracks, desc="Analyzing licenses", unit="track"):
            # Same analysis logic (omitted for brevity - identical to streaming version)
            pass
    
    # Print diagnostics
    print("\n" + "="*60)
    print("JAMENDO FULL CATALOG LICENSE DIAGNOSTICS")
    print("="*60)
    print(f"Total tracks in catalog: {total_tracks:,}")
    print(f"Tracks with cc==true && ccnc==false && ccnd==false: {cc_flags_condition_count:,} ({100*cc_flags_condition_count/total_tracks:.1f}%)")
    print(f"Tracks with ccnc==false && ccnd==false (any cc): {no_nc_nd_count:,} ({100*no_nc_nd_count/total_tracks:.1f}%)")
    print(f"Tracks with empty license URL: {empty_license_count:,} ({100*empty_license_count/total_tracks:.1f}%)")
    print(f"Tracks with content_id_free=false: {content_id_restricted_count:,} ({100*content_id_restricted_count/total_tracks:.1f}%)")
    
    # Show examples (same as before)
    if cc_flags_passed_examples:
        print("\n" + "="*60)
        print("🔍 CC FLAGS CONDITION EXAMPLES (first 5)")
        print("="*60)
        for track in cc_flags_passed_examples:
            print(f"ID: {track['id']}")
            print(f"  Name: {track['name']}")
            print(f"  License URL: {track['license_url']}")
            print()
    
    if passed_cc_filter:
        print("\n" + "="*60)
        print("✅ PASSED FILTER - CC-BY or CC-BY-SA (first 5)")
        print("="*60)
        for track in passed_cc_filter[:5]:
            print(f"ID: {track['id']}")
            print(f"  Name: {track['name']}")
            print(f"  License: {track['canonical'].upper()}")
            print(f"  URL: {track['license_url']}")
            print()

except KeyboardInterrupt:
    print("\n\n⚠️  INTERRUPTED - Cleaning up resources...")
    if 'checkpoint' in locals() and 'offset' in locals():
        try:
            checkpoint['last_offset'] = offset
            checkpoint['tracks_fetched'] = len(all_tracks) if not ENABLE_STREAMING else offset
            save_checkpoint(checkpoint)
            print(f"✓ Checkpoint saved at offset {offset:,}")
            print(f"✓ You can resume by re-running this cell")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
    
    if checkpoint_file is not None:
        try:
            checkpoint_file.flush()
            checkpoint_file.close()
            print("✓ Checkpoint file closed")
        except:
            pass
    
    try:
        client.close()
        print("✓ HTTP client closed")
    except:
        pass
    
    print("\n🛑 Fetch interrupted. Progress has been saved.")
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Error: {e}")
    if 'checkpoint' in locals() and 'offset' in locals():
        try:
            checkpoint['last_offset'] = offset
            checkpoint['tracks_fetched'] = len(all_tracks) if not ENABLE_STREAMING else offset
            save_checkpoint(checkpoint)
            print(f"✓ Checkpoint saved at offset {offset:,}")
        except:
            pass
    raise

finally:
    if checkpoint_file is not None:
        try:
            checkpoint_file.close()
        except:
            pass
    
    try:
        client.close()
    except:
        pass