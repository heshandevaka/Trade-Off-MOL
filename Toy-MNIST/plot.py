import matplotlib.pyplot as plt
import numpy as np
import  os

# ablation over different gamma for MoDo
def gamma_ablation(plot_type='gen'):
    # folder containing data logs for ablation
    folder='./modo_gamma_ablation_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # hp set used for ablation
    gamma_list = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1.'] #

    # init lists to collect data from different seeds
    opt_error = {gamma:[] for gamma in gamma_list}
    pop_error = {gamma:[] for gamma in gamma_list}
    gen_error = {gamma:[] for gamma in gamma_list}

    # scrape the log files
    for gamma in gamma_list:
        for file in file_list:
            if f'gamma-{gamma}' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                print(f'\n{file}', gamma)
                for line in lines:
                    # scrape pop. error data from log file
                    if "Population error" in line:
                        pop_error[gamma].append(float(line.strip().split(': ')[1]))
                    # scrape opt. error data from log file
                    if "Optimization error" in line:
                        opt_error[gamma].append(float(line.strip().split(': ')[1]))
                    # scrape gen. error data from log file (absolute value)
                    if "Generalization error" in line:
                        gen_error[gamma].append(abs(float(line.strip().split(': ')[1])))

    # # normalization of values # NOTE: maybe not that usefeul
    # opt_error = {gamma:np.array(opt_error[gamma]) for gamma in gamma_list}
    # pop_error = {gamma:np.array(pop_error[gamma]) for gamma in gamma_list}
    # gen_error = {gamma:np.array(gen_error[gamma]) for gamma in gamma_list}
    # opt_error_norm = {gamma: (opt_error[gamma] - np.min(opt_error[gamma]))/(np.min(opt_error[gamma])) for gamma in gamma_list}
    # pop_error_norm = {gamma: (pop_error[gamma] - np.min(pop_error[gamma]))/(np.min(pop_error[gamma])) for gamma in gamma_list}
    # gen_error_norm = {gamma: (gen_error[gamma] - np.min(gen_error[gamma]))/(np.min(gen_error[gamma])) for gamma in gamma_list}

    # opt_error_mean = np.array([np.mean(opt_error_norm[gamma]) for gamma in gamma_list])
    # opt_error_std = np.array([np.std(opt_error_norm[gamma]) for gamma in gamma_list])
    # gen_error_mean = np.array([np.mean(gen_error_norm[gamma]) for gamma in gamma_list])
    # gen_error_std = np.array([np.std(gen_error_norm[gamma]) for gamma in gamma_list])
    # pop_error_mean = np.array([np.mean(pop_error_norm[gamma]) for gamma in gamma_list])
    # pop_error_std = np.array([np.std(pop_error_norm[gamma]) for gamma in gamma_list])

    # calc mean and std deviation for plotting (without normlization)
    opt_error_mean = np.array([np.mean(opt_error[gamma]) for gamma in gamma_list])
    opt_error_std = np.array([np.std(opt_error[gamma]) for gamma in gamma_list])
    gen_error_mean = np.array([np.mean(gen_error[gamma]) for gamma in gamma_list])
    gen_error_std = np.array([np.std(gen_error[gamma]) for gamma in gamma_list])
    pop_error_mean = np.array([np.mean(pop_error[gamma]) for gamma in gamma_list])
    pop_error_std = np.array([np.std(pop_error[gamma]) for gamma in gamma_list])

    # plot
    fig, ax = plt.subplots()
    gamma_list = [float(gamma) for gamma in gamma_list]
    ax.set_xscale("log")
    if plot_type=='pop_opt':
        ax.errorbar(gamma_list, opt_error_mean, opt_error_std, fmt='o-', capsize=5, color='b', label=r'$R_{opt}$')
        ax.errorbar(gamma_list, pop_error_mean, pop_error_std, fmt='o-', capsize=5, color='r', label=r'$R_{pop}$')
        ax.set_ylabel('Error metric value')
    if plot_type=='gen':
        ax.errorbar(gamma_list, gen_error_mean, gen_error_std, fmt='o-', capsize=5, color='g', label=r'$|R_{gen}|$')
        ax.set_ylabel('Absolute generalization error')
    ax.set_xlabel(r'$\gamma$')
    ax.legend()
    plt.savefig(f'./figures/gamma_{plot_type}_err_comp')

# ablation over different rho for MoDo
def rho_ablation(plot_type='gen'):
    # folder containing data logs for ablation
    folder='./modo_rho_ablation_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # hp set used for ablation
    rho_list = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1.'] #

    # init lists to collect data from different seeds
    opt_error = {rho:[] for rho in rho_list}
    pop_error = {rho:[] for rho in rho_list}
    gen_error = {rho:[] for rho in rho_list}

    # scrape the log files
    for rho in rho_list:
        for file in file_list:
            if f'rho-{rho}' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                print(f'\n{file}', rho)
                for line in lines:
                    # scrape pop. error data from log file
                    if "Population error" in line:
                        pop_error[rho].append(float(line.strip().split(': ')[1]))
                    # scrape opt. error data from log file
                    if "Optimization error" in line:
                        opt_error[rho].append(float(line.strip().split(': ')[1]))
                    # scrape gen. error data from log file (absolute value)
                    if "Generalization error" in line:
                        gen_error[rho].append(abs(float(line.strip().split(': ')[1])))

    # # normalization of values # NOTE: maybe not that usefeul
    # opt_error = {rho:np.array(opt_error[rho]) for rho in rho_list}
    # pop_error = {rho:np.array(pop_error[rho]) for rho in rho_list}
    # gen_error = {rho:np.array(gen_error[rho]) for rho in rho_list}
    # opt_error_norm = {rho: (opt_error[rho] - np.min(opt_error[rho]))/(np.min(opt_error[rho])) for rho in rho_list}
    # pop_error_norm = {rho: (pop_error[rho] - np.min(pop_error[rho]))/(np.min(pop_error[rho])) for rho in rho_list}
    # gen_error_norm = {rho: (gen_error[rho] - np.min(gen_error[rho]))/(np.min(gen_error[rho])) for rho in rho_list}

    # opt_error_mean = np.array([np.mean(opt_error_norm[rho]) for rho in rho_list])
    # opt_error_std = np.array([np.std(opt_error_norm[rho]) for rho in rho_list])
    # gen_error_mean = np.array([np.mean(gen_error_norm[rho]) for rho in rho_list])
    # gen_error_std = np.array([np.std(gen_error_norm[rho]) for rho in rho_list])
    # pop_error_mean = np.array([np.mean(pop_error_norm[rho]) for rho in rho_list])
    # pop_error_std = np.array([np.std(pop_error_norm[rho]) for rho in rho_list])

    # calc mean and std deviation for plotting
    opt_error_mean = np.array([np.mean(opt_error[rho]) for rho in rho_list])
    opt_error_std = np.array([np.std(opt_error[rho]) for rho in rho_list])
    gen_error_mean = np.array([np.mean(gen_error[rho]) for rho in rho_list])
    gen_error_std = np.array([np.std(gen_error[rho]) for rho in rho_list])
    pop_error_mean = np.array([np.mean(pop_error[rho]) for rho in rho_list])
    pop_error_std = np.array([np.std(pop_error[rho]) for rho in rho_list])

    # plot
    fig, ax = plt.subplots()
    rho_list = [float(rho) for rho in rho_list]
    ax.set_xscale("log")
    if plot_type=='pop_opt':
        ax.errorbar(rho_list, opt_error_mean, opt_error_std, fmt='o-', capsize=5, color='b', label=r'$R_{opt}$')
        ax.errorbar(rho_list, pop_error_mean, pop_error_std, fmt='o-', capsize=5, color='r', label=r'$R_{pop}$')
        ax.set_ylabel('Error metric value')
    if plot_type=='gen':
        ax.errorbar(rho_list, gen_error_mean, gen_error_std, fmt='o-', capsize=5, color='g', label=r'$|R_{gen}|$')
        ax.set_ylabel('Absolute generalization error')
    ax.set_xlabel(r'$\rho$')
    ax.legend()
    plt.savefig(f'./figures/rho_{plot_type}_err_comp')

# ablation over different rho for MoDo
def loss_calc():
    # folder containing data logs for ablation
    folder='./perf_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # hp set used for ablation
    moo_method_list = ['EW', 'MGDA', 'MoCo', 'MoDo']

    # init lists to collect data from different seeds
    train_loss = {moo_method:[] for moo_method in moo_method_list}
    test_loss = {moo_method:[] for moo_method in moo_method_list}
    opt_error = {moo_method:[] for moo_method in moo_method_list}
    pop_error = {moo_method:[] for moo_method in moo_method_list}
    gen_error = {moo_method:[] for moo_method in moo_method_list}

    # scrape the log files
    for moo_method in moo_method_list:
        for file in file_list:
            if f'{moo_method}-' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                for line in lines:
                    # scrape test loss data from log file
                    if "Test  | " in line:
                        loss_list = line.strip().split(' | ')[2]
                        loss_list = [float(loss) for loss in loss_list.split(' ')[1:]]
                        test_loss[moo_method].append(loss_list)
                    # scrape pop. error data from log file
                    if "Population error" in line:
                        pop_error[moo_method].append(float(line.strip().split(': ')[1]))
                    # scrape opt. error data from log file
                    if "Optimization error" in line:
                        opt_error[moo_method].append(float(line.strip().split(': ')[1]))
                    # scrape gen. error data from log file (absolute value)
                    if "Generalization error" in line:
                        gen_error[moo_method].append(abs(float(line.strip().split(': ')[1])))

# ablation over different rho for MoDo
def loss_plots():
    # folder containing data logs for ablation
    folder='./loss_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # hp set used for ablation
    moo_method_list = ['EW', 'MGDA', 'MoCo', 'MoDo']

    # init lists to collect data from different seeds
    train_loss = {moo_method:[] for moo_method in moo_method_list}
    test_loss = {moo_method:[] for moo_method in moo_method_list}

    # scrape the log files
    for moo_method in moo_method_list:
        for file in file_list:
            if f'{moo_method}-' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                train_epoch_loss_list = []
                test_epoch_loss_list = []
                for line in lines:
                    # scrape test and training loss at each iteration from log file
                    if ("Epoch" in line) and ("Format" not in line):
                        test_loss_list, train_loss_list = line.strip().split(' | ')[1], line.strip().split(' | ')[2]
                        train_loss_list = [float(loss) for loss in train_loss_list.split(': ')[1].strip().split(' ')]
                        test_loss_list = [float(loss) for loss in test_loss_list.split(': ')[1].strip().split(' ')]
                        train_epoch_loss_list.append(train_loss_list)
                        test_epoch_loss_list.append(test_loss_list)
                train_loss[moo_method].append(train_epoch_loss_list)
                test_loss[moo_method].append(test_epoch_loss_list)

    # print loss statistics
    for moo_method in test_loss:
        print(f"\n{moo_method}")
        print("Number of seeds:", len(test_loss[moo_method]))
        print("Mean of test losses:")
        print(np.mean(np.array(test_loss[moo_method]), axis=0))
        print("Std. dev of test losses:")
        print(np.std(np.array(test_loss[moo_method]), axis=0))
        print(f"Mean of population error: {np.mean(np.array(pop_error[moo_method]))}")
        print(f"St. dev. of population error: {np.std(np.array(pop_error[moo_method]))}")
        print(f"Mean of optimization error: {np.mean(np.array(opt_error[moo_method]))}")
        print(f"St. dev. of optimization error: {np.std(np.array(opt_error[moo_method]))}")
        print(f"Mean of generalization error: {np.mean(np.array(gen_error[moo_method]))}")
        print(f"St. dev. generalization error: {np.std(np.array(gen_error[moo_method]))}")
        print()
    print(pop_error)
    print(opt_error)
    print(gen_error)
    
if __name__=='__main__':    
    # rho ablation
    # rho_ablation(plot_type='gen')

    # gamma ablation
    gamma_ablation(plot_type='pop_opt')

    # perf. metric calc.
    # loss_calc()