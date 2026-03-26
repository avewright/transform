#!/usr/bin/env python3
"""Autonomous data pipeline monitor and launcher.

Monitors:
1. Source 0 processing (prepare_hf_dataset.py --dry-run)
2. Main Lichess pipeline (process_lichess_parquets.py, sources 12-16)

When both complete:
3. Uploads source 0 shards to HF
4. Verifies all 17 sources present on HF
5. Launches custom position generation
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SOURCE0_LOG = Path("/workspace/source0_processing/process.log")
SOURCE0_SHARD_DIR = Path("/workspace/source0_processing/shards")
PIPELINE_LOG = Path("/workspace/chess_hf_pipeline/pipeline.log")
PIPELINE_EVENTS = Path("/workspace/chess_hf_pipeline/orchestrator_events.jsonl")
TARGET_REPO = "avewright/chess-positions-lichess-sf"


def load_hf_token():
    env_path = Path("/root/transform/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN", "")


def check_source0_done():
    """Check if source 0 processing is complete."""
    if not SOURCE0_LOG.exists():
        return False
    text = SOURCE0_LOG.read_text()
    # Look for completion indicators
    if "Dry run complete" in text or "Processing complete" in text:
        return True
    # Check if the progress manifest indicates completion
    manifest_path = SOURCE0_SHARD_DIR / "progress.json"
    if manifest_path.exists():
        try:
            progress = json.loads(manifest_path.read_text())
            if progress.get("processing_completed") or progress.get("upload_completed"):
                return True
        except Exception:
            pass
    # Check if process is still running by looking for recent progress
    lines = text.strip().split("\n")
    for line in reversed(lines[-5:]):
        if "DRY RUN" in line.upper() or "dry run" in line.lower() or "skipping upload" in line.lower():
            return True
    return False


def check_pipeline_done():
    """Check if the main Lichess pipeline has finished all sources (12-16)."""
    if not PIPELINE_EVENTS.exists():
        return False
    events = []
    for line in PIPELINE_EVENTS.read_text().strip().split("\n"):
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    uploaded = set()
    for e in events:
        if e.get("event") == "source_upload_completed":
            src = e.get("source", "")
            uploaded.add(src)
    # Check if sources 12-16 are all uploaded
    needed = {f"data/train-{i:05d}-of-00017.parquet" for i in range(12, 17)}
    return needed.issubset(uploaded)


def get_pipeline_progress():
    """Get latest progress from main pipeline."""
    if not PIPELINE_LOG.exists():
        return "no log"
    text = PIPELINE_LOG.read_text()
    # Find latest progress line
    for line in reversed(text.strip().split("\n")):
        if "Progress:" in line:
            return line.strip()
        if "Processing source" in line:
            return line.strip()
        if "upload_completed" in line or "Uploaded" in line:
            return line.strip()
    return "unknown"


def get_source0_progress():
    """Get latest progress from source 0."""
    if not SOURCE0_LOG.exists():
        return "no log"
    text = SOURCE0_LOG.read_text()
    for line in reversed(text.strip().split("\n")):
        if "Progress:" in line:
            return line.strip()
    return "unknown"


def upload_source0_shards(token):
    """Upload source 0 train/test shards to HF."""
    from huggingface_hub import HfApi, CommitOperationAdd

    api = HfApi(token=token)

    # Find all parquet shards
    train_dir = SOURCE0_SHARD_DIR / "train"
    test_dir = SOURCE0_SHARD_DIR / "test"

    operations = []
    for d, split in [(train_dir, "train"), (test_dir, "test")]:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            remote_name = f"data/{split}-src00000-{f.stem.split('-')[-1]}.parquet"
            operations.append(CommitOperationAdd(
                path_in_repo=remote_name,
                path_or_fileobj=str(f),
            ))

    if not operations:
        print("  WARNING: No source 0 shards found to upload")
        return False

    print(f"  Uploading {len(operations)} source 0 shards...")
    api.create_commit(
        repo_id=TARGET_REPO,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add source 0 shards ({len(operations)} files)",
    )
    print(f"  Source 0 uploaded: {len(operations)} shards")
    return True


def verify_all_sources(token):
    """Verify all 17 Lichess sources are present on HF."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.dataset_info(TARGET_REPO)
    files = [s.rfilename for s in info.siblings
             if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet")]

    sources = set()
    for f in files:
        m = re.search(r"src(\d+)", f)
        if m:
            sources.add(int(m.group(1)))

    total_files = len(files)
    done = sorted(sources)
    missing = sorted(set(range(17)) - sources)

    print(f"  HF dataset: {total_files} files, sources: {done}")
    if missing:
        print(f"  MISSING sources: {missing}")
        return False
    else:
        print(f"  ALL 17 SOURCES PRESENT!")
        return True


def launch_custom_generation():
    """Launch the custom position generation script."""
    print("\n  Launching custom position generation...")
    cmd = [
        sys.executable, "-u", "generate_and_upload.py",
        "--workers", "48",
        "--batch", "500",
        "--total", "5000000",
        "--depth", "10",
        "--upload-every", "250000",
        "--shard-size", "250000",
        "--prefix", "gen",
    ]
    env = os.environ.copy()
    env["HF_TOKEN"] = load_hf_token()
    env["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

    log_path = Path("outputs/custom_generated/generation.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            env=env, cwd="/root/transform",
        )
    print(f"  Generation PID: {proc.pid}")
    print(f"  Log: {log_path}")
    return proc


def main():
    print(f"{'='*70}")
    print(f" AUTONOMOUS DATA PIPELINE MONITOR")
    print(f"{'='*70}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print()

    token = load_hf_token()
    if not token:
        print("ERROR: No HF_TOKEN found!")
        sys.exit(1)

    # Phase 1: Wait for both pipelines to complete
    print("Phase 1: Monitoring Lichess pipelines...")
    source0_done = False
    pipeline_done = False

    while not (source0_done and pipeline_done):
        if not source0_done:
            source0_done = check_source0_done()
            if source0_done:
                print(f"  [✓] Source 0 processing COMPLETE")
            else:
                print(f"  [ ] Source 0: {get_source0_progress()}")

        if not pipeline_done:
            pipeline_done = check_pipeline_done()
            if pipeline_done:
                print(f"  [✓] Main pipeline COMPLETE (sources 6-16)")
            else:
                print(f"  [ ] Pipeline: {get_pipeline_progress()}")

        if not (source0_done and pipeline_done):
            print(f"  Waiting 120s... ({time.strftime('%H:%M:%S')})")
            time.sleep(120)

    # Phase 2: Upload source 0
    print(f"\nPhase 2: Uploading source 0 shards...")
    if upload_source0_shards(token):
        print("  Source 0 upload complete!")
    else:
        print("  WARNING: Source 0 upload had issues")

    # Phase 3: Verify all sources
    print(f"\nPhase 3: Verifying all 17 sources on HF...")
    all_present = verify_all_sources(token)
    if not all_present:
        print("  WARNING: Not all sources present, but continuing with generation")

    # Phase 4: Launch custom generation
    print(f"\nPhase 4: Starting custom position generation...")
    proc = launch_custom_generation()

    # Phase 5: Monitor generation
    print(f"\nPhase 5: Monitoring generation (checking every 5 min)...")
    gen_log = Path("outputs/custom_generated/generation.log")

    while proc.poll() is None:
        time.sleep(300)
        if gen_log.exists():
            lines = gen_log.read_text().strip().split("\n")
            for line in lines[-5:]:
                if "Round" in line or "Upload" in line or "COMPLETE" in line:
                    print(f"  {line.strip()}")

        # Check manifest for progress
        manifest_path = Path("outputs/custom_generated/manifest.json")
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                print(f"  Progress: {m.get('total_labeled', 0):,} labeled, "
                      f"{m.get('upload_count', 0)} uploads, "
                      f"{m.get('overall_rate', 0)} pos/s")
            except Exception:
                pass

    rc = proc.returncode
    print(f"\n  Generation finished with exit code {rc}")

    # Final verification
    print(f"\nFinal verification:")
    verify_all_sources(token)

    print(f"\n{'='*70}")
    print(f" PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")


if __name__ == "__main__":
    main()
