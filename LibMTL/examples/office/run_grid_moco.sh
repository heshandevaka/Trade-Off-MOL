mkdir -p hp_grid_logs
dataset_path=office-31
weighting=MoCo
seed=0
gpu_id=0

LR_SET="1e-5 1e-4 1e-3 1e-2"
BETA_SET="0.01 0.1 1"
GAMMA_SET="0.01 0.1 1"
RHO_SET="0.01 0.1 1"

for lr in $LR_SET; do
    for beta in $BETA_SET; do
        for gamma in $GAMMA_SET; do
            for rho in $RHO_SET; do
                echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --weighting $weighting  --lr $lr --gamma_modo $gamma --rho_modo $rho  > hp_grid_logs/$weigting-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out &"
                python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --weighting $weighting  --lr $lr --beta_moco $beta --gamma_moco $gamma --rho_moco $rho > hp_grid_logs/$weighting-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out &
            done
        done
    done
done