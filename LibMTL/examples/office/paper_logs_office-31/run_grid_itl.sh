mkdir -p hp_grid_logs
dataset_path=office-31
weighting=ITL
seed=0
gpu_id=1

LR_SET="1e-3 1e-2"
WEIGHT_DECAY_SET="1e-7"
TASK_IXD_SET="0 1 2"

for lr in $LR_SET; do
    for weight_decay in $WEIGHT_DECAY_SET; do
        for task_idx in $TASK_IXD_SET; do
            echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --weighting $weighting  --lr $lr  --weight_decay $weight_decay --task_idx $task_idx > hp_grid_logs/$weighting-seed-$seed-lr-$lr-weight_decay-$weight_decay-task_idx-$task_idx.out"
            python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --weighting $weighting  --lr $lr  --weight_decay $weight_decay --task_idx $task_idx > hp_grid_logs/$weighting-seed-$seed-lr-$lr-weight_decay-$weight_decay-task_idx-$task_idx.out
        done
    done
done