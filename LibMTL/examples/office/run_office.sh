folder="logs"
mkdir -p $folder
dataset_path=office-31
dataset=office-31
weighting=MoDo
gpu_id=0
seed=0
# EW, MGDA, MoDo params
lr="1e-4"
# EW, MGDA params
weight_decay="1e-3"
# MoDo params
gamma="1e-3"
bs=32
# EW, MGDA
# python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --weighting $weighting  --lr $lr  --weight_decay $weight_decay > $folder/$weighting-seed-$seed-lr-$lr-weight_decay-$weight_decay.out
# MoDo
python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --weighting $weighting  --lr $lr  --gamma_modo $gamma --bs $bs  > $folder/$weighting-seed-$seed-lr-$lr-gamma-$gamma-bs-$bs.out
