mkdir -p hp_grid_logs
dataset_path=office-31
weighting=MoDo
seed=0
gpu_id=0
bs=32

LR_SET="1e-5 1e-4 1e-3 1e-2"
GAMMA_SET="0.01 0.1 1"
RHO_SET="0.01 0.1 1"

for lr in $LR_SET; do
    for gamma in $GAMMA_SET; do
        for rho in $RHO_SET; do
            echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --weighting $weighting  --lr $lr --gamma_modo $gamma --rho_modo $rho  --bs $bs > hp_grid_logs/$weighting-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho-bs-$bs.out"
            python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --weighting $weighting  --lr $lr --gamma_modo $gamma --rho_modo $rho  --bs $bs > hp_grid_logs/$weighting-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho-bs-$bs.out
        done
    done
done