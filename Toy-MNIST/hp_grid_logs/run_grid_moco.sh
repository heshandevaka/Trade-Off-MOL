folder="hp_grid_logs"
mkdir -p $folder
moo_method=MoCo
seed=0

LR_SET="10"
BETA_SET="0.1"
GAMMA_SET="0.1 1"
RHO_SET="0.0"

for lr in $LR_SET; do
    for beta in $BETA_SET; do
        for gamma in $GAMMA_SET; do
            for rho in $RHO_SET; do
                echo "python -u toy_fast.py --seed $seed --moo_method $moo_method  --lr $lr --beta $beta --gamma $gamma --rho $rho > $folder/$moo_method-seed-$seed-lr-$lr-beta-$beta-gamma-$gamma-rho-$rho.out"
                python -u toy_fast.py --seed $seed --moo_method $moo_method  --lr $lr --beta_moco $beta --gamma_moco $gamma --rho_moco $rho > $folder/$moo_method-seed-$seed-lr-$lr-beta-$beta-gamma-$gamma-rho-$rho.out
            done
        done
    done
done