mkdir -p perf_logs
moo_method=MoCo

LR_SET="0.1"
BETA_SET="0.1"
GAMMA_SET="0.01"
RHO_SET="0.1"
SEED_SET="0 1 2 3 4 5 6 7 8 9"

for lr in $LR_SET; do
    for beta in $BETA_SET; do
        for gamma in $GAMMA_SET; do
            for rho in $RHO_SET; do
                for seed in $SEED_SET; do
                    echo "python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr --beta $beta --gamma $gamma --rho $rho > perf_logs/$moo_method-seed-$seed-lr-$lr-beta-$beta-gamma-$gamma-rho-$rho.out"
                    python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr --beta_moco $beta --gamma_moco $gamma --rho_moco $rho > perf_logs/$moo_method-seed-$seed-lr-$lr-beta-$beta-gamma-$gamma-rho-$rho.out
                done
            done
        done
    done
done