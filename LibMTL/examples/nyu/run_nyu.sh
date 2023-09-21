folder="logs"
mkdir -p $folder
weighting=EW
gpu_id=0
seed=0
# EW, MGDA, MoDo params
lr="1e-4"
# EW, MGDA params
weight_decay="1e-4"
# MoDo params
gamma="1e-3"
train_bs=2
# EW, MGDA
python -u train_nyu.py --seed $seed --gpu_id $gpu_id --weighting $weighting  --lr $lr  --weight_decay $weight_decay > $folder/$weighting-seed-$seed-lr-$lr-weight_decay-$weight_decay.out
# MoDo
# python -u train_nyu.py --seed $seed --gpu_id $gpu_id --train_bs $train_bs --weighting $weighting  --lr $lr  --gamma_modo $gamma > $folder/$weighting-seed-$seed-lr-$lr-gamma-$gamma-bs-$train_bs.out
