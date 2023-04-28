mkdir -p modo_gamma_ablation_logs
moo_method=MoDo

LR_SET="0.1"
GAMMA_SET="0.001 0.01 0.1 1."
RHO_SET="0.1"
SEED_SET="5 6 7 8 9"

for lr in $LR_SET; do
    for gamma in $GAMMA_SET; do
        for rho in $RHO_SET; do
            for seed in $SEED_SET; do
                echo "python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --rho_modo $rho > modo_gamma_ablation_logs/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out"
                python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --rho_modo $rho > modo_gamma_ablation_logs/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-rho-$rho.out
            done
        done
    done
done