#!/usr/bin/env bash

#SBATCH --account=mfmpm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name larsbigLR
#SBATCH --mem 500G
#SBATCH --output=/p/scratch/mfmpm/montillet/data/outs/ssl_%j.out

curr_file="$(scontrol show job "$SLURM_JOB_ID" | grep '^[[:space:]]*Command=' | head -n 1 | cut -d '=' -f 2-)"
curr_dir="$(dirname "$curr_file")"

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source "$curr_dir"/activate.sh

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=54123

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0


srun env -u CUDA_VISIBLE_DEVICES python -u -m torchrun_jsc \
       --nproc_per_node=gpu \
       --nnodes="$SLURM_JOB_NUM_NODES" \
       --rdzv_id="$SLURM_JOB_ID" \
       --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
       --rdzv_backend=c10d \
       /p/scratch/mfmpm/montillet/code/moco-v3/main_moco.py \
                -a vit_base -b 1024 \
                --optimizer=lars --lr=1e-3 --weight-decay=.1 \
                --epochs=100 --warmup-epochs=20 \
                --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
                --dist-url 'env://' \
                --tar-dir /p/scratch/mfmpm/tim/data/TCGA/cache \
                --workers 48 --output /p/scratch/mfmpm/montillet/data/ckpt \
                --cache /p/scratch/mfmpm/cache-montillet \
                --num-tiles 1000000 --version free
