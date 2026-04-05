"""Quick test of data loader streaming."""
import sys, time
sys.path.insert(0, ".")
from data_loader import StreamingHFChessLoader

print("Creating loader with 1 src file...")
t0 = time.time()
loader = StreamingHFChessLoader(
    repo_id="avewright/chess-positions-lichess-sf",
    batch_size=256,
    encoder_type="fused",
    device="cpu",  # Don't need GPU for this test
    seed=42,
    drop_last=True,
    file_pattern="src",
    max_files=1,
)
print(f"Loader created in {time.time()-t0:.1f}s")

print("Iterating first file...")
t1 = time.time()
count = 0
for batch_input, move_targets, wdl_targets in loader:
    count += 1
    if count == 1:
        print(f"  First batch received in {time.time()-t1:.1f}s")
        print(f"  Batch shape: {move_targets.shape}")
    if count % 100 == 0:
        print(f"  Batch {count}: {count*256:,} positions ({time.time()-t1:.1f}s)")

print(f"Done: {count} batches, {count*256:,} positions in {time.time()-t1:.1f}s")
