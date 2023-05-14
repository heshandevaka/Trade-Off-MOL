dir_name="mo_descent_error_logs"
mkdir -p $dir_name
moo_method=MGDA

LR_SET="5.0"
SEED_SET="1 2 3 4 5 6 7 8 9"

for lr in $LR_SET; do
    for seed in $SEED_SET; do
        echo "python -u toy_descent_error.py --seed $seed --moo_method $moo_method  --lr $lr  > $dir_name/$moo_method-seed-$seed-lr-$lr.out"
        python -u toy_descent_error.py --seed $seed --moo_method $moo_method  --lr $lr  > $dir_name/$moo_method-seed-$seed-lr-$lr.out
    done
done