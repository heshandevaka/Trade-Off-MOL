mkdir -p hp_grid_logs
moo_method=MoDo
seed=0

LR_SET="0.1 1"
GAMMA_SET="0.01"
RHO_SET="0.1"

for lr in $LR_SET; do
    for gamma in $GAMMA_SET; do
        for rho in $RHO_SET; do
            echo "python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --rho_modo $rho > hp_grid_logs/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out"
            python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --rho_modo $rho > hp_grid_logs/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out
        done
    done
done