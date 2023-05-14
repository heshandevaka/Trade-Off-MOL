folder="modo_T_ablation_new_logs"
mkdir -p $folder
moo_method=MoDo

LR_SET="0.1"
GAMMA_SET="0.01"
T_SET="10-100000"
T_max=100000
SEED_SET="0 1 2 3 4 5 6 7 8 9"

for lr in $LR_SET; do
    for gamma in $GAMMA_SET; do
        for seed in $SEED_SET; do
            echo "python -u toy_T_ablation.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --num_epoch $T_max > $folder/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-T-$T_SET.out"
            python -u toy_T_ablation.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma --num_epoch $T_max > $folder/$moo_method-seed-$seed-lr-$lr-gamma-$gamma-T-$T_SET.out
        done
    done
done