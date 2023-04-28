mkdir -p hp_grid_logs
dataset_path=office-31
weighting=MGDA
seed=0
gpu_id=0

LR_SET="1e-3 1e-2 1e-1"
# WEIGHT_DECAY_SET="1e-7 1e-6 1e-5 1e-4 1e-3 1e-2"

for lr in $LR_SET; do
    # for weight_decay in $WEIGHT_DECAY_SET; do
        echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --weighting $weighting  --lr $lr  > hp_grid_logs/$weighting-seed-$seed-lr-$lr.out"
        python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --weighting $weighting  --lr $lr  > hp_grid_logs/$weighting-seed-$seed-lr-$lr.out
    # done
done