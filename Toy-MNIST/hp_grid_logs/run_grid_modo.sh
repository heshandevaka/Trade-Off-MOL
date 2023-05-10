folder="hp_grid_logs"
mkdir -p $folder
moo_method=MoDo
seed=0

LR_SET="1 10"
GAMMA_SET="1 10
"
RHO_SET="0.0"

for lr in $LR_SET; do
    for gamma in $GAMMA_SET; do
        for rho in $RHO_SET; do
            echo "python -u toy_fast.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --rho_modo $rho > $folder/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out"
            python -u toy_fast.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --rho_modo $rho > $folder/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out
        done
    done
done