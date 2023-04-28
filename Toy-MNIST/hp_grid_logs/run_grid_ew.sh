mkdir -p hp_grid_logs
moo_method=EW
seed=0

LR_SET="5e-1 1"

for lr in $LR_SET; do
    echo "python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr  > hp_grid_logs/$moo_method-seed-$seed-lr-$lr.out"
    python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr  > hp_grid_logs/$moo_method-seed-$seed-lr-$lr.out
done