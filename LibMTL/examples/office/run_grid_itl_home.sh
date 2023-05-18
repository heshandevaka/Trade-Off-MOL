folder="hp_grid_logs_home"
mkdir -p $folder
dataset_path=office-home
dataset=office-home
weighting=ITL
seed=0
gpu_id=1

LR_SET="1e-4"
WEIGHT_DECAY_SET="1e-3"
TASK_IXD_SET="0 1 2 3"

for lr in $LR_SET; do
    for weight_decay in $WEIGHT_DECAY_SET; do
        for task_idx in $TASK_IXD_SET; do
            echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --weighting $weighting  --lr $lr  --weight_decay $weight_decay --task_idx $task_idx > $folder/$weighting-seed-$seed-lr-$lr-weight_decay-$weight_decay-task_idx-$task_idx-$dataset.out"
            python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --weighting $weighting  --lr $lr  --weight_decay $weight_decay --task_idx $task_idx > $folder/$weighting-seed-$seed-lr-$lr-weight_decay-$weight_decay-task_idx-$task_idx-$dataset.out
        done
    done
done