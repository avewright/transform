#!/usr/bin/env python3
"""Synthetic CUDA training-memory check. No dataset access or saved weights."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import numpy as np
import torch
from torch.nn import functional as F
from chess_value99 import ValueConfig,ValueTransformer,pack_board
from experiments.value99_pretrain import precision_context

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/value99_pretrain.json')
    p.add_argument('--microbatch',type=int)
    p.add_argument('--steps',type=int,default=3)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if not torch.cuda.is_available():raise SystemExit('Run this check on the target CUDA GPU; no data was accessed.')
    if cfg['precision']=='bf16' and not torch.cuda.is_bf16_supported():raise SystemExit('GPU does not support BF16; use precision=fp32.')
    micro=a.microbatch or cfg['microbatch']
    if micro<1 or a.steps<1:raise SystemExit('microbatch and steps must be positive')
    model=ValueTransformer(ValueConfig(**cfg['model'])).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
    x=torch.from_numpy(np.stack([pack_board(chess.Board())]*micro)).cuda()
    target=torch.linspace(.1,.9,micro,device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
        with precision_context(torch.device('cuda'),cfg['precision']):
            loss=F.binary_cross_entropy_with_logits(model(x,return_logit=True),target)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize()
        print(json.dumps(dict(step=step+1,microbatch=micro,loss=loss.item(),seconds=time.monotonic()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            total_vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30)),flush=True)
if __name__=='__main__':main()
