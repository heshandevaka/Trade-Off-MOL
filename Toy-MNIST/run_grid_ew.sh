mkdir -p perf_logs
moo_method=EW

LR_SET="0.1"
SEED_SET="0 1 2 3 4 5 6 7 8 9"

for lr in $LR_SET; do
    for seed in $SEED_SET; do
        echo "python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr  > perf_logs/$moo_method-seed-$seed-lr-$lr.out"
        python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr  > perf_logs/$moo_method-seed-$seed-lr-$lr.out
    done
done