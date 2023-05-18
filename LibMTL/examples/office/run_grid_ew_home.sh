folder="hp_grid_logs_home"
mkdir -p $folder
dataset_path=office-home
dataset=office-home
weighting=EW
seed=0
gpu_id=0

LR_SET="1e-6 1e-5 1e-4"
WEIGHT_DECAY_SET="1e-5 1e-4 1e-3"

for lr in $LR_SET; do
    for weight_decay in $WEIGHT_DECAY_SET; do
        echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --weighting $weighting  --lr $lr  --weight_decay $weight_decay > $folder/$weighting-seed-$seed-lr-$lr-weight_decay-$weight_decay.out"
        python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --weighting $weighting  --lr $lr  --weight_decay $weight_decay > $folder/$weighting-seed-$seed-lr-$lr-weight_decay-$weight_decay.out
    done
done