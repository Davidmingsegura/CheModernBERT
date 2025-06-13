#!/bin/bash
#SBATCH --job-name=8192-procedure-new
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --output=./output_%j_%N.log  
#SBATCH --error=.error_%j_%N.log  
#SBATCH --no-requeue
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH -A a-a05  

export MASTER_ADDR=$(hostname)
export WANDB_DIR=./wandb_logs
export WANDB_API_KEY=
export MASTER_PORT=25678
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_LEVEL=NVL

set -eo pipefail

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1


export PYTHONPATH="./modernbert-venv/lib/python3.10/site-packages:$PYTHONPATH"

srun -ul --container-writable --environment=./.edf/modernbert-pytorch24.toml bash -c '
export NCCL_DEBUG=INFO

source ./modernbert-venv24/bin/activate

# pip install --no-deps "accelerate==0.26.0"

TORCHRUN_ARGS="
--nnodes=1 \
--nproc_per_node=4 \
--node-rank=$SLURM_PROCID \
--master-addr=$MASTER_ADDR \
--master-port=$MASTER_PORT \
--nnodes=$SLURM_NNODES \
"
echo "Launching torchrun with: $TORCHRUN_ARGS"
NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 python -m torch.distributed.run $TORCHRUN_ARGS ./modernbert-mlm-ft-fineweb-procedures-8192-token.py
'