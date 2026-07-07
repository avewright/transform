"""Monitor training log and print eval summaries."""
import sys, time, os

def monitor(logfile, poll_interval=30):
    seen_evals = set()
    while True:
        if not os.path.exists(logfile):
            time.sleep(poll_interval)
            continue
        lines = open(logfile).readlines()
        for i, l in enumerate(lines):
            s = l.strip()
            if 'Eval' in s and i not in seen_evals:
                seen_evals.add(i)
                # Print this eval block
                block = [s]
                for j in range(i+1, min(i+8, len(lines))):
                    ns = lines[j].strip()
                    if any(k in ns for k in ['live:', 'ema:', 'NEW BEST', 'aux:']):
                        block.append(ns)
                    elif 'step=' in ns:
                        break
                for b in block:
                    print(b, flush=True)
                print(flush=True)
            if 'Done.' in s or 'Final:' in s:
                if i not in seen_evals:
                    seen_evals.add(i)
                    print(s, flush=True)
        
        # Check if done
        for l in lines[-5:]:
            if 'Done.' in l:
                return
        
        time.sleep(poll_interval)

if __name__ == '__main__':
    logfile = sys.argv[1] if len(sys.argv) > 1 else '/root/transform/outputs/exp101_long_v2/exp101.log'
    monitor(logfile)
