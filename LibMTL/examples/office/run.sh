# MoCo
# python -u train_office.py --multi_input --dataset_path office-31 --weighting MoCo --rho_moco 0.5 --gpu_id 0 2>&1 >> test_logs/moco_rho_0.5_grad_calc_remove_test.out &
# MoCo
# python -u train_office.py --multi_input --dataset_path office-31 --weighting MoDo --gpu_id 1 --bs 32 --rho_modo 0.5 2>&1 >> test_logs/modo_bs_32_rho_0.5.out &
# EW
# python -u train_office.py --multi_input --dataset_path office-31 --gpu_id 0 2>&1 >> test_logs/ew_init.out &
# Test balanced data set with EW
python -u train_office.py --multi_input --dataset_path office-31 --balanced --gpu_id 1 2>&1 >> test_logs/ew_balanced_dataset_test.out &