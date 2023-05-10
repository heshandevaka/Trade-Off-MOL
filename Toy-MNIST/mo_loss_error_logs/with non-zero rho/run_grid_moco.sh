dir_name="mo_loss_error_logs"
mkdir -p $dir_name
moo_method=MoCo

LR_SET="1"
BETA_SET="0.1"
GAMMA_SET="0.1"
RHO_SET="0.0"
SEED_SET="0"

for lr in $LR_SET; do
    for beta in $BETA_SET; do
        for gamma in $GAMMA_SET; do
            for rho in $RHO_SET; do
                for seed in $SEED_SET; do
                    echo "python -u toy_loss_error.py --seed $seed --moo_method $moo_method  --lr $lr --beta $beta --gamma $gamma --rho $rho > $dir_name/$moo_method-seed-$seed-lr-$lr-beta-$beta-gamma-$gamma-rho-$rho.out"
                    python -u toy_loss_error.py --seed $seed --moo_method $moo_method  --lr $lr --beta_moco $beta --gamma_moco $gamma --rho_moco $rho > $dir_name/$moo_method-seed-$seed-lr-$lr-beta-$beta-gamma-$gamma-rho-$rho.out
                done
            done
        done
    done
done