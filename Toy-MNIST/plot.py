import matplotlib.pyplot as plt
import numpy as np
import  os

# ablation over different gamma for MoDo
def gamma_ablation():
    # folder containing data logs for ablation
    folder='./modo_gamma_ablation_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # hp set used for ablation
    gamma_list = ['0.001', '0.01', '0.1', '1.'] #

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
    print(gen_error)
    # normalization of values
    opt_error = {gamma:np.array(opt_error[gamma]) for gamma in gamma_list}
    pop_error = {gamma:np.array(pop_error[gamma]) for gamma in gamma_list}
    gen_error = {gamma:np.array(gen_error[gamma]) for gamma in gamma_list}
    opt_error_norm = {gamma: (opt_error[gamma] - np.min(opt_error[gamma]))/(np.min(opt_error[gamma])) for gamma in gamma_list}
    pop_error_norm = {gamma: (pop_error[gamma] - np.min(pop_error[gamma]))/(np.min(pop_error[gamma])) for gamma in gamma_list}
    gen_error_norm = {gamma: (gen_error[gamma] - np.min(gen_error[gamma]))/(np.min(gen_error[gamma])) for gamma in gamma_list}

    opt_error_mean = np.array([np.mean(opt_error_norm[gamma]) for gamma in gamma_list])
    opt_error_std = np.array([np.std(opt_error_norm[gamma]) for gamma in gamma_list])
    gen_error_mean = np.array([np.mean(gen_error_norm[gamma]) for gamma in gamma_list])
    gen_error_std = np.array([np.std(gen_error_norm[gamma]) for gamma in gamma_list])
    pop_error_mean = np.array([np.mean(pop_error_norm[gamma]) for gamma in gamma_list])
    pop_error_std = np.array([np.std(pop_error_norm[gamma]) for gamma in gamma_list])

    # # calc mean and std deviation for plotting (without normlization)
    # opt_error_mean = np.array([np.mean(opt_error[gamma]) for gamma in gamma_list])
    # opt_error_std = np.array([np.std(opt_error[gamma]) for gamma in gamma_list])
    # gen_error_mean = np.array([np.mean(gen_error[gamma]) for gamma in gamma_list])
    # gen_error_std = np.array([np.std(gen_error[gamma]) for gamma in gamma_list])
    # pop_error_mean = np.array([np.mean(pop_error[gamma]) for gamma in gamma_list])
    # pop_error_std = np.array([np.std(pop_error[gamma]) for gamma in gamma_list])

    # plot
    fig, ax = plt.subplots()
    gamma_list = [float(gamma) for gamma in gamma_list]
    ax.semilogx(gamma_list, opt_error_mean, 'o-', label=r'$R_{opt}$')
    plt.fill_between(gamma_list, opt_error_mean - opt_error_std, opt_error_mean + opt_error_std, alpha=0.5)
    ax.semilogx(gamma_list, pop_error_mean, 'o-', label=r'$R_{pop}$')
    plt.fill_between(gamma_list, pop_error_mean - pop_error_std, pop_error_mean + pop_error_std, alpha=0.5)
    # ax.semilogx(gamma_list, gen_error_mean, 'go-', label=r'$R_{gen}$')
    # plt.fill_between(gamma_list, gen_error_mean - gen_error_std, gen_error_mean + gen_error_std, color='g', alpha=0.5)
    ax.set_ylabel('Error metric magnitude (normalized)')
    ax.set_xlabel(r'$\gamma$')
    ax.legend()
    plt.savefig('gamma_pop_opt_norm_err_comp')

# ablation over different rho for MoDo
def rho_ablation():
    # folder containing data logs for ablation
    folder='./modo_rho_ablation_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # hp set used for ablation
    rho_list = ['0.001', '0.01', '0.1', '1.'] #

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

    # normalization of values
    opt_error = {rho:np.array(opt_error[rho]) for rho in rho_list}
    pop_error = {rho:np.array(pop_error[rho]) for rho in rho_list}
    gen_error = {rho:np.array(gen_error[rho]) for rho in rho_list}
    opt_error_norm = {rho: (opt_error[rho] - np.min(opt_error[rho]))/(np.min(opt_error[rho])) for rho in rho_list}
    pop_error_norm = {rho: (pop_error[rho] - np.min(pop_error[rho]))/(np.min(pop_error[rho])) for rho in rho_list}
    gen_error_norm = {rho: (gen_error[rho] - np.min(gen_error[rho]))/(np.min(gen_error[rho])) for rho in rho_list}

    opt_error_mean = np.array([np.mean(opt_error_norm[rho]) for rho in rho_list])
    opt_error_std = np.array([np.std(opt_error_norm[rho]) for rho in rho_list])
    gen_error_mean = np.array([np.mean(gen_error_norm[rho]) for rho in rho_list])
    gen_error_std = np.array([np.std(gen_error_norm[rho]) for rho in rho_list])
    pop_error_mean = np.array([np.mean(pop_error_norm[rho]) for rho in rho_list])
    pop_error_std = np.array([np.std(pop_error_norm[rho]) for rho in rho_list])

    # # calc mean and std deviation for plotting
    # opt_error_mean = np.array([np.mean(opt_error[rho]) for rho in rho_list])
    # opt_error_std = np.array([np.std(opt_error[rho]) for rho in rho_list])
    # gen_error_mean = np.array([np.mean(gen_error[rho]) for rho in rho_list])
    # gen_error_std = np.array([np.std(gen_error[rho]) for rho in rho_list])
    # pop_error_mean = np.array([np.mean(pop_error[rho]) for rho in rho_list])
    # pop_error_std = np.array([np.std(pop_error[rho]) for rho in rho_list])

    # plot
    fig, ax = plt.subplots()
    rho_list = [float(rho) for rho in rho_list]
    ax.semilogx(rho_list, opt_error_mean, 'o-', label=r'$R_{opt}$')
    plt.fill_between(rho_list, opt_error_mean - opt_error_std, opt_error_mean + opt_error_std, alpha=0.5)
    ax.semilogx(rho_list, pop_error_mean, 'o-', label=r'$R_{pop}$')
    plt.fill_between(rho_list, pop_error_mean - pop_error_std, pop_error_mean + pop_error_std, alpha=0.5)
    # ax.semilogx(rho_list, gen_error_mean, 'go-', label=r'$R_{gen}$')
    # plt.fill_between(rho_list, gen_error_mean - gen_error_std, gen_error_mean + gen_error_std, color='g', alpha=0.5)
    ax.set_ylabel('Error metric magnitude (normalized)')
    ax.set_xlabel(r'$\rho$')
    ax.legend()
    plt.savefig('rho_pop_opt_norm_err_comp')


rho_ablation()