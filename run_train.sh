#!/usr/bin/env bash
# Multi-GPU DDP training for SeLop/LROR on MIDS.
#   GPUS=0,1,2,3 OMP_NUM_THREADS=12 ./run_train.sh   [extra train.py args]
set -e
cd /datasets/work/vLLM/temp/PAAS_SeLop

PY=python3.12
GPUS="${GPUS:-1}"
NPROC=$(echo "$GPUS" | awk -F, '{print NF}')

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# CPU cap ~50%: 4 procs x 12 threads = 48 of 96 cores ("like OMP").
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "GPUS=$GPUS  NPROC=$NPROC  OMP_NUM_THREADS=$OMP_NUM_THREADS"
# Launch via `python3.12 -m torch.distributed.run` (NOT the `torchrun` shim, whose
# shebang points at a different python that lacks transformers) so DDP workers
# run in this env.
"$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
    --master_port="${MASTER_PORT:-29533}" \
    train.py --config config.json "$@"
