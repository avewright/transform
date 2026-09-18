#!/usr/bin/env python3
"""Score a Value99 checkpoint against official Stockfish 19 WDL.

Does not write gradients or resume the trainer. Shares the GPU if --device cuda.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from chess_value99 import ValueConfig,ValueTransformer
from experiments.value99_pretrain import evaluate
from value99_sfwdl import load_teacher


def load_model(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=ValueTransformer(ValueConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model']);model.eval()
    return model,int(ckpt.get('step',-1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'outputs/value99_data_sfwdl')
    p.add_argument('--device',default='cuda')
    p.add_argument('--microbatch',type=int,default=64)
    p.add_argument('--precision',default='bf16')
    p.add_argument('--append-log',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu')
    precision=a.precision if device.type=='cuda' else 'fp32'
    packed,target,phase,_black=load_teacher(a.data)
    model,step=load_model(a.checkpoint,device)
    data=(torch.from_numpy(np.ascontiguousarray(packed)),
          torch.from_numpy(np.ascontiguousarray(target)),
          torch.from_numpy(np.ascontiguousarray(phase)))
    t=time.time()
    result=evaluate(model,data,device,a.microbatch,precision)
    report=dict(stage='teacher',source='avewright/local-wdl',step=step,
                checkpoint=str(a.checkpoint),elapsed_s=time.time()-t,**result)
    text=json.dumps(report)
    print(text,flush=True)
    if a.append_log:
        a.append_log.parent.mkdir(parents=True,exist_ok=True)
        with a.append_log.open('a') as f:f.write(text+'\n')
    if a.out:a.out.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
