"""Quick script to check available HF datasets."""
import sys
sys.path.insert(0, ".")
from data_loader import get_hf_dataset_layout

for repo in ['avewright/chess-positions', 'avewright/chess-positions-sf-200k', 'avewright/chess-positions-lichess-sf']:
    print(f'=== {repo} ===')
    try:
        layout = get_hf_dataset_layout(repo)
        tm = layout["train_main"]
        ts = layout["train_src"]
        print(f'  Train main: {len(tm)} files')
        print(f'  Train src: {len(ts)} files')
        print(f'  Test main: {len(layout["test_main"])} files')
        if tm:
            print(f'  First: {tm[0]}')
        if ts:
            print(f'  First src: {ts[0]}')
    except Exception as e:
        print(f'  Error: {e}')
    print()
