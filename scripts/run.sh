#!/bin/bash
# usage: run.sh <logname> <python args...>   -- runs one pipeline step, logs to logs/<logname>.log
cd ~/heritage
source .venv/bin/activate
export PYTHONHASHSEED=0 HF_HOME=$HOME/heritage/hf_cache OMP_NUM_THREADS=2
name=$1; shift
python -u "$@" > logs/$name.log 2>&1
echo "EXIT $?" >> logs/$name.log
