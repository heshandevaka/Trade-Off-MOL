folder="logs"
mkdir -p $folder
moo_method=EW
seed=0
# EW, MGDA, MoDo params
lr="1e-1"
# MoDo params
gamma="1."
# run EW, MGDA
python -u toy_fast.py --seed $seed --moo_method $moo_method  --lr $lr  > $folder/$moo_method-seed-$seed-lr-$lr.out
# run MoDo
# python -u toy_fast.py --seed $seed --moo_method $moo_method  --lr $lr --gamma_modo $gamma > $dir_name/$moo_method-seed-$seed-lr-$lr-gamma-$gamma.out

