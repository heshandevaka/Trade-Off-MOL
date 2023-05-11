import matplotlib.pyplot as plt
import numpy as np
import  os
import copy

# plot loss vs epochs
def calc_deltam(moo_method):
    # folder containing data logs for ablation
    folder=f'./hp_grid_logs/{moo_method}/'
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

if __name__=='__main__':
    calc_deltam('MGDA')