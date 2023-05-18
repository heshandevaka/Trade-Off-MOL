import matplotlib.pyplot as plt
import numpy as np
import  os
import copy

# calc val deltam to choose hp
def calc_val_deltam(log_dir, moo_method, task='office-31'):
    # folder containing data logs for ablation
    folder=f'./{log_dir}/{moo_method}/'
    file_list = os.listdir(folder)
    num_epochs = 21

    print(file_list)

    # manually obtained best results for each tasks
    if task=='office-31':
        base_results = np.array([0.875, 0.9888, 0.9732])
        weights = np.array([1., 1., 1.])
    min_deltam = 100
    best_hp = ''
    best_epoch = ''

    # scrape the log files
    for file in file_list:
        if f'{moo_method}-' in file:
            foo = open(folder+file)
            lines = foo.read().split('\n')
            train_epoch_loss_list = []
            test_epoch_loss_list = []
            for i, line in enumerate(lines):
                # scrape test and training loss at each iteration from log file
                if ("Epoch" in line) and ("Best" not in line):
                    # print(line)
                    val_list_ = line.strip().split(' | VAL: ')[1].split(' | Time: ')[0].split(' | ')
                    # print(val_list_)
                    val_acc_list = np.array([float(val_.split(' ')[1]) for val_ in val_list_])
                    deltam = np.mean((-1)**weights*(val_acc_list-base_results)/base_results)
                    if min_deltam > deltam:
                        min_deltam = deltam
                        best_hp = file
                        best_epoch = line
    print(min_deltam)
    print(best_hp)
    print(best_epoch)

# calc test deltam to report
def calc_test_deltam(log_dir, moo_method):
    # folder containing data logs for ablation
    folder=f'./{log_dir}/{moo_method}/'
    file_list = os.listdir(folder)
    num_epochs = 21

    print(file_list)

    # hp set used for ablation
    # moo_method_list = ['EW', 'MGDA', 'MoCo', 'MoDo']
    base_results = np.array([0.875, 0.9888, 0.9732])
    weights = np.array([1., 1., 1.])
    min_deltam = 100
    best_hp = ''
    best_epoch = ''

    # scrape the log files
    deltm_list = []
    best_results_list = []
    count = 0
    for file in file_list:
        if f'{moo_method}-' in file:
            foo = open(folder+file)
            lines = foo.read().split('\n')
            train_epoch_loss_list = []
            test_epoch_loss_list = []
            for i, line in enumerate(lines):
                # scrape test and training loss at each iteration from log file
                if ("Best Result:" in line):
                    # print(line)
                    best_test_results_ = line.strip().split('[')
                    best_test_results = []
                    for i, xx in enumerate(best_test_results_):
                        if i==0:
                            continue
                        best_test_results.append(float(xx.split(']')[0]))
                    print(best_test_results)
                    best_test_results = np.array(best_test_results)
                    deltam = np.mean((-1)**weights*(best_test_results-base_results)/base_results)
                    deltm_list.append(deltam)
                    best_results_list.append(best_test_results)

    best_results_list = np.array(best_results_list)
    deltm_list = np.array(deltm_list)

    print(f'\nAvg best test results (over {count} seeds): {np.mean(best_results_list, axis=0)}')
    print(f'Std dev best test results (over {count} seeds): {np.std(best_results_list, axis=0)}')
    print(f'Avg deltam (over {count} seeds): {np.mean(deltm_list)}')
    print(f'Std dev deltam (over {count} seeds): {np.std(deltm_list)}\n')

if __name__=='__main__':
    calc_test_deltam('perf_logs', 'MGDA')