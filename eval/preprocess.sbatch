#!/usr/bin/env bash

#SBATCH --account=mfmpm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=2
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name preCPTACF
#SBATCH --mem 500G
#SBATCH --output=/p/scratch/mfmpm/montillet/data/outs/preprocess_%j.out


curr_file="$(scontrol show job "$SLURM_JOB_ID" | grep '^[[:space:]]*Command=' | head -n 1 | cut -d '=' -f 2-)"
curr_dir="$(dirname "$curr_file")"

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

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

module load Stages
module load GCC OpenMPI PyTorch torchvision

source /p/scratch/mfmpm/montillet/code/STAMP/.venv/bin/activate

stamp --config /p/scratch/mfmpm/montillet/code/eval/config.yaml preprocess
#stamp --config /p/scratch/mfmpm/montillet/code/eval/config.yaml train
#stamp --config /p/scratch/mfmpm/montillet/code/eval/config.yaml deploy
