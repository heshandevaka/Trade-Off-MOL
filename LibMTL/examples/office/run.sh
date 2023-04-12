# MoCo
# python -u train_office.py --multi_input --dataset_path office-31 --weighting MoCo --rho_moco 0.5 --gpu_id 0 2>&1 >> train_logs/moco_rho_0.5_test.out &
# EW
python -u train_office.py --multi_input --dataset_path office-31 --gpu_id 0 2>&1 >> train_logs/ew_init.out &