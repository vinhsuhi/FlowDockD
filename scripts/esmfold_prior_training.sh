#!/bin/bash -l
######################### Batch Headers #########################
#SBATCH --nodes=1              # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres gpu:A100:8      # request H100 GPU resource(s)
#SBATCH --ntasks-per-node=8    # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --mem=0                # NOTE: use `--mem=0` to request all memory "available" on the assigned node
#SBATCH --cpus-per-task=4     # NOTE: this needs to match Lightning's `Trainer(num_processes=...)
#SBATCH -t 2-00:00:00          # time limit for the job (up to 2 days: `2-00:00:00`)
#SBATCH -J esmfold_prior_training # job name
#SBATCH --output=R-%x.%j.out   # output log file
#SBATCH --error=R-%x.%j.err    # error log file

random_seconds=$(( (RANDOM % 100) + 1 ))
echo "Sleeping for $random_seconds seconds before starting run"
sleep "$random_seconds"


module load CUDA
module load Miniconda3
source ${EBROOTMINICONDA3}/bin/activate

conda activate se3_genie

echo "Calling flowdock/train.py!"

srun python3 flowdock/train.py \
    experiment='flowdock_fm' \
    environment=slurm \
    model.cfg.prior_type=esmfold \
    model.cfg.task.freeze_score_head=false \
    model.cfg.task.freeze_affinity=false \
    strategy=ddp \
    trainer=ddp \
    trainer.devices=8 \
    trainer.num_nodes=1 \
    ckpt_path="/mnt/beegfs/home/ac141281/FlowDockD/logs/train/runs/2025-02-22_16-23-42/checkpoints/last.ckpt"

echo "Finished calling flowdock/train.py!"

# NOTE: the following commands must be used to resume training from a checkpoint
# ckpt_path="$(realpath 'logs/train/runs/2024-05-17_13-45-06/checkpoints/last.ckpt')" \
# paths.output_dir="$(realpath 'logs/train/runs/2024-05-17_13-45-06')" \

# NOTE: the following commands may be used to speed up training
# model.compile=false \
# +trainer.precision=bf16-mixed
