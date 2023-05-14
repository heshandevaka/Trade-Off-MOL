dir_name="mo_descent_error_logs"
mkdir -p $dir_name
moo_method=MoDo

LR_SET="1."
GAMMA_SET="1."
RHO_SET="0.0"
SEED_SET="1 2 3 4 5 6 7 8 9" # run 0 again

for lr in $LR_SET; do
    for gamma in $GAMMA_SET; do
        for rho in $RHO_SET; do
            for seed in $SEED_SET; do
                echo "python -u toy_descent_error.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --rho_modo $rho > $dir_name/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out"
                python -u toy_descent_error.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --rho_modo $rho > $dir_name/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out
            done
        done
    done
done