dir_name="mo_loss_error_logs"
mkdir -p $dir_name
moo_method=EW

LR_SET="0.1"
SEED_SET="0 1 2 3 4 5 6 7 8 9" # 1 2 3 4 5 6 7 8 9

for lr in $LR_SET; do
    for seed in $SEED_SET; do
        echo "python -u toy_loss_error.py --seed $seed --moo_method $moo_method  --lr $lr  > $dir_name/$moo_method-seed-$seed-lr-$lr.out"
        python -u toy_loss_error.py --seed $seed --moo_method $moo_method  --lr $lr  > $dir_name/$moo_method-seed-$seed-lr-$lr.out
    done
done