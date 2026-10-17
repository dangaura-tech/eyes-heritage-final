#!/bin/bash
# Label-shuffle test of the full ensemble pipeline (members retrained on permuted labels, seed 0).
cd ~/heritage; source .venv/bin/activate
export HERITAGE_SHUFFLE=1 PYTHONHASHSEED=0 HF_HOME=$HOME/heritage/hf_cache OMP_NUM_THREADS=2
cfg() { python -c "import json,sys
r={}
for l in open('reports/runs.jsonl'):
    d=json.loads(l); r[d['name']]=d
c=dict(r[sys.argv[1]]['cfg']); c['notes']='LABEL SHUFFLE TEST'; print(json.dumps(c))" $1; }
chain=$1
if [ "$chain" = "cpu_gpu" ]; then
  python -u scripts/exp_A.py SHUF_A_lgb_top2_dino "$(cfg A_lgb_top2_dino)" 0 > logs/SHUF_A.log 2>&1; echo EXIT $? >> logs/SHUF_A.log
  python -u scripts/exp_A.py SHUF_Apt_opt "$(cfg Apt_opt)" 0 > logs/SHUF_Apt.log 2>&1; echo EXIT $? >> logs/SHUF_Apt.log
  python -u scripts/exp_B.py SHUF_B_l_B1_norev "$(cfg B_l_B1_norev)" 0 > logs/SHUF_B.log 2>&1; echo EXIT $? >> logs/SHUF_B.log
else
  python -u scripts/exp_C.py SHUF_C_l_B2_lora "$(cfg C_l_B2_lora)" 0 > logs/SHUF_C.log 2>&1; echo EXIT $? >> logs/SHUF_C.log
fi
