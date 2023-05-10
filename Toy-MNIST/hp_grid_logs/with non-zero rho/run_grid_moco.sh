mkdir -p hp_grid_logs
moo_method=MoCo
seed=0

LR_SET="0.1 1"
BETA_SET="0.1"
GAMMA_SET="0.01"
RHO_SET="0.0"

for lr in $LR_SET; do
    for beta in $BETA_SET; do
        for gamma in $GAMMA_SET; do
            for rho in $RHO_SET; do
                echo "python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr --beta $beta --gamma $gamma --rho $rho > hp_grid_logs/$moo_method-seed-$seed-lr-$lr-beta-$beta-gamma-$gamma-rho-$rho.out"
                python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr --beta_moco $beta --gamma_moco $gamma --rho_moco $rho > hp_grid_logs/$moo_method-seed-$seed-lr-$lr-beta-$beta-gamma-$gamma-rho-$rho.out
            done
        done
    done
done