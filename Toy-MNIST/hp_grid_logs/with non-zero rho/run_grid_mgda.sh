mkdir -p hp_grid_logs
moo_method=MGDA
seed=0

LR_SET="1e-2 5e-2 1e-1 5e-1 1."

for lr in $LR_SET; do
    echo "python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr  > hp_grid_logs/$moo_method-seed-$seed-lr-$lr.out"
    python -u toy.py --seed $seed --moo_method $moo_method  --lr $lr  > hp_grid_logs/$moo_method-seed-$seed-lr-$lr.out
done