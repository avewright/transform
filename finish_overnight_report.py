import os,time,json
from pathlib import Path
os.environ['MOVE_VOCAB_VERSION']='compact'
os.environ['OMP_NUM_THREADS']='1'
os.environ['STOCKFISH_PATH']='/root/.local/bin/stockfish-19'
from scripts.overnight_sf19 import sf_screen,maybe_upload,write,guard_bad
out=Path('/root/transform/outputs/sf19_ft/overnight_20260908')
deadline=json.loads((out/'started.json').read_text())['deadline']
results=json.loads((out/'elo_results.json').read_text())
validations=json.loads((out/'candidate_validation.json').read_text())
baseline=json.loads((out/'baseline_validation.json').read_text())
candidates={'original':Path('/root/transform/outputs/sf19_ft/init.pt'),'run1_1500':Path('/root/transform/outputs/sf19_ft/run1/step_001500.pt'),'source':out/'source_full.pt'}
for name in ('best_elo','latest','eval_swa'): candidates[name]=out/f'{name}.pt'
for name in ('latest','eval_swa'):
    remaining=deadline-time.time()-300
    if remaining<120: break
    results[name]=sf_screen(candidates[name],'recovered_'+name,out,min(600,remaining/2),repeats=2)
    write(out/'recovered_elo_results.json',results)
qualified=[n for n in ('best_elo','latest','eval_swa') if results.get(n,{}).get('n_games')==64 and results[n]['returncode']==0 and all(not guard_bad(validations[n][k],baseline[k]) for k in ('soft','replay','deep'))]
chosen=max(qualified,key=lambda n:results[n]['score']) if qualified else 'source'
maybe_upload(chosen,candidates,results,validations,baseline,out)
write(out/'finished.json',{'reason':'training_wall_budget','training_steps':99372,'recovered_report':True,'screen_candidate':chosen,'elo_results':results,'promotion_report':str(out/'promotion.json')})
print(json.dumps(json.loads((out/'promotion.json').read_text()),indent=2),flush=True)
