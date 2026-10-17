#!/bin/bash
# Evaluate the label-shuffled members with the same ensemble protocols as the real audit.
cd ~/heritage; source .venv/bin/activate; export HERITAGE_SHUFFLE=1
M=SHUF_A_lgb_top2_dino,SHUF_Apt_opt,SHUF_C_l_B2_lora,SHUF_B_l_B1_norev
python -u scripts/ensemble.py $M --out SHUF_ens > logs/SHUF_ens.log 2>&1
python -u scripts/audit_ensemble.py $M --tag SHUF_ens > logs/SHUF_audit.log 2>&1
cp reports/audit_SHUF_ens.json reports/audit_shuffle.json
echo EXIT $?
