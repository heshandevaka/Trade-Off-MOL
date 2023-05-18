import matplotlib.pyplot as plt
import numpy as np
import  os
import copy

import matplotlib.lines as mlines
from scipy.io import savemat

# ------------------------------------------------------------legacy code-----------------------------------------------------------------------------------------

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
    std_scale = 0.5
    # ax.set_yscale("log") # not useful since all errors are in similar order
    if plot_type=='pop_opt':
        ax.errorbar(gamma_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color='b',markeredgecolor='k', markersize=5, linewidth=4, label=r'$R_{opt}$')
        ax.errorbar(gamma_list, pop_error_mean, std_scale*pop_error_std, fmt='o-', capsize=5, 
                    color='r',markeredgecolor='k', markersize=5, linewidth=4, label=r'$R_{pop}$')
        ax.set_ylabel('Error')
    if plot_type=='gen':
        ax.errorbar(gamma_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color='g',markeredgecolor='k', markersize=5, linewidth=4, label=r'$|R_{gen}|$')
        ax.set_ylabel('Absolute generalization error')
    ax.set_xlabel(r'$\gamma$')
    ax.legend()
    plt.savefig(f'./figures/gamma_{plot_type}_err_comp.pdf', bbox_inches='tight')

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
        ax.errorbar(rho_list, opt_error_mean, opt_error_std, fmt='o-', capsize=5, color='b', markeredgecolor='k', markersize=5, label=r'$R_{opt}$')
        ax.errorbar(rho_list, pop_error_mean, pop_error_std, fmt='^-', capsize=5, color='r',markeredgecolor='k', markersize=5, label=r'$R_{pop}$')
        ax.set_ylabel('Error')
    if plot_type=='gen':
        ax.errorbar(rho_list, gen_error_mean, gen_error_std, fmt='v-', capsize=5, color='g',markeredgecolor='k', markersize=5, label=r'$|R_{gen}|$')
        ax.set_ylabel('Absolute generalization error')
        # gen_err_handle = ax.errorbar([], [], [], fmt='v-', capsize=10, color='g',markeredgecolor='k',
        #                   markersize=10, label=r'$|R_{gen}|$')
    ax.legend()
    ax.set_xlabel(r'$\rho$')
    plt.savefig(f'./figures/rho_{plot_type}_err_comp.pdf', bbox_inches='tight')

    # ablation over different hp for MoDo
def hp_ablation_old(plot_type='gen', hp_type='gamma'):
    # folder containing data logs for ablation
    folder=f'./modo_{hp_type}_ablation_new_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # hp set used for ablation
    if hp_type=='gamma':
        hp_list = ['0.0075', '0.01', '0.025', '0.05', '0.075', '0.1', '0.25', '0.5', '0.75', '1.', '1.25']
        hp_label = r'$\gamma$'
    if hp_type=='rho':
        hp_list = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1.']
        hp_label = r'$\rho$'
    if hp_type=='lr':
        hp_list = ['0.0075', '0.01', '0.025', '0.05', '0.075', '0.1', '0.25', '0.5', '0.75', '1.', '1.25']
        hp_label = r'$\alpha$'

    # init lists to collect data from different seeds
    opt_error = {hp:[] for hp in hp_list}
    pop_error = {hp:[] for hp in hp_list}
    gen_error = {hp:[] for hp in hp_list}

    # scrape the log files
    for hp in hp_list:
        for file in file_list:
            if f'{hp_type}-{hp}' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                print(f'\n{file}', hp)
                for line in lines:
                    # scrape pop. error data from log file
                    if "Population error" in line:
                        pop_error[hp].append(float(line.strip().split(': ')[1]))
                    # scrape opt. error data from log file
                    if "Optimization error" in line:
                        opt_error[hp].append(float(line.strip().split(': ')[1]))
                    # scrape gen. error data from log file (absolute value)
                    if "Generalization error" in line:
                        gen_error[hp].append(abs(float(line.strip().split(': ')[1])))

    # calc mean and std deviation for plotting (without normlization)
    opt_error_mean = np.array([np.mean(opt_error[hp]) for hp in hp_list])
    opt_error_std = np.array([np.std(opt_error[hp]) for hp in hp_list])
    gen_error_mean = np.array([np.mean(gen_error[hp]) for hp in hp_list])
    gen_error_std = np.array([np.std(gen_error[hp]) for hp in hp_list])
    pop_error_mean = np.array([np.mean(pop_error[hp]) for hp in hp_list])
    pop_error_std = np.array([np.std(pop_error[hp]) for hp in hp_list])

    # plot
    fig, ax = plt.subplots()
    hp_list = [float(hp) for hp in hp_list]

    ax.set_xscale("log")
    opt_color = np.array([84,172,108])/255 # green
    pop_color = np.array([70,70,70])/255 # grey
    gen_color = np.array([196,78,82])/255 # red
    # ax.set_yscale("log") # not useful since all errors are in similar order
    if plot_type=='pop_opt':
        std_scale = 0.5
        ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)
        ax.set_ylabel('Error', fontsize=18)
        ax.set_xlabel(hp_label, fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(fontsize=18)
    if plot_type=='gen':
        std_scale = 0.5
        ax.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')
        ax.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.set_xlabel(hp_label, fontsize=18)
        # if hp_type=='gamma':
        #     ax.set_ylim([0.0005, 0.004])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(fontsize=18)
        # ax.set_ylabel('Absolute generalization error', fontsize=15)
    if plot_type=='all':
        std_scale = 1.0
        ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')  
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)   
        ax.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')   
        ax.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.set_ylabel('Error', fontsize=18)
        ax.set_xlabel(hp_label+' (log scale)', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if hp_type=='gamma':
            ax.legend(fontsize=18)
    # use left and right sides to plot opt-pop and gen in different scales
    if plot_type=='all-diff-scale':
        std_scale = 0.5
        opt_handle = ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        pop_handle = ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')  
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)  
        ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
        gen_handle = ax2.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')   
        ax2.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='y', which='major', labelsize=15, labelcolor=gen_color)
        ax2.set_ylabel('Generalization', fontsize=18)
        ax2.yaxis.label.set_color(gen_color)
        ax.set_ylabel('Population/Optimization', fontsize=18)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel(hp_label+' (log scale)', fontsize=18)
        if hp_type=='gamma':
            plt.legend(loc='center right', handles=[opt_handle, pop_handle, gen_handle], fontsize=18)
        # ax.set_ylabel('Error', fontsize=18)
    # ax.set_xlabel(hp_label, fontsize=18)
    # if plot_type=='gen' and hp_type=='gamma':
    #     ax.set_ylim([0.0005, 0.004])
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.savefig(f'./figures/{hp_type}_{plot_type}_err_comp_new.pdf', bbox_inches='tight')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------
# from CAGrad
def smooth(x, n=4):
    l = len(x)
    y = []
    for i in range(l):
        ii = max(0, i-n)
        jj = min(i+n, l-1)
        v = np.array(x[ii:jj]).astype(np.float64)
        if i < 1:
            y.append(x[i])
        else:
            y.append(v.mean())
    return np.array(y)

# ablation over different hp for MoDo
def hp_ablation(plot_type='gen', hp_type='gamma'):

    # hp set used for ablation
    if hp_type=='gamma':
        hp_list = ['0.0075', '0.01', '0.025', '0.05', '0.075', '0.1', '0.25', '0.5', '0.75', '1.', '1.25']
        hp_label = r'$\gamma$'
    if hp_type=='lr':
        hp_list = ['0.0075', '0.01', '0.025', '0.05', '0.075', '0.1', '0.25', '0.5', '0.75', '1.', '1.25']
        hp_label = r'$\alpha$'

    
    # folder containing data logs for ablation
    folder=f'./modo_{hp_type}_ablation_new_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # init lists to collect data from different seeds
    opt_error = {hp:[] for hp in hp_list}
    pop_error = {hp:[] for hp in hp_list}
    gen_error = {hp:[] for hp in hp_list}

    # scrape the log files
    for hp in hp_list:
        for file in file_list:
            if f'{hp_type}-{hp}' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                print(f'\n{file}', hp)
                for line in lines:
                    # scrape pop. error data from log file
                    if "Population error" in line:
                        pop_error[hp].append(float(line.strip().split(': ')[1]))
                    # scrape opt. error data from log file
                    if "Optimization error" in line:
                        opt_error[hp].append(float(line.strip().split(': ')[1]))
                    # scrape gen. error data from log file (absolute value)
                    if "Generalization error" in line:
                        gen_error[hp].append(abs(float(line.strip().split(': ')[1])))

    n_smooth=4 # param to smooth out the descrete curve, to show trend
    # calc mean and std deviation for plotting (without normlization)
    opt_error_mean = smooth(np.array([np.mean(opt_error[hp]) for hp in hp_list]), n=n_smooth)
    opt_error_std = smooth(np.array([np.std(opt_error[hp]) for hp in hp_list]), n=n_smooth)
    gen_error_mean = smooth(np.array([np.mean(gen_error[hp]) for hp in hp_list]), n=n_smooth)
    gen_error_std = smooth(np.array([np.std(gen_error[hp]) for hp in hp_list]), n=n_smooth)
    pop_error_mean = smooth(np.array([np.mean(pop_error[hp]) for hp in hp_list]), n=n_smooth)
    pop_error_std = smooth(np.array([np.std(pop_error[hp]) for hp in hp_list]), n=n_smooth)

    print(opt_error_mean)
    print(opt_error_std)


    # folder containing data logs for ablation
    folder=f'./modo_{hp_type}_ablation_descent_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # init lists to collect data from different seeds
    descent_error = {hp:[] for hp in hp_list}

    # scrape the log files
    for hp in hp_list:
        for file in file_list:
            if f'{hp_type}-{hp}' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                print(f'\n{file}', hp)
                # to caclualte iteration Eca from moving avaerage recorded
                count=0
                for line in lines:
                    # scrape final descent error data from log file
                    if ("Epoch" in line) and ("FORMAT" not in line):
                        count += 1
                    if "Epoch:    950" in line:
                        temp = float(line.strip().split(': ')[2]) * count
                    if "Epoch:  1,000" in line:
                        print(line)
                        descent_error[hp].append((float(line.strip().split(': ')[2]) * count - temp)**2)

    descent_error_mean = smooth(np.array([np.mean(descent_error[hp]) for hp in hp_list]), n=n_smooth)
    descent_error_std = smooth(np.array([np.std(descent_error[hp]) for hp in hp_list]), n=n_smooth)

    # save for plotting
    mat_dict = {f"{hp_type}_list":np.array(hp_list), 
                f"opt_error_mean_{hp_type}":opt_error_mean,
                f"opt_error_std_{hp_type}":opt_error_std,
                f"gen_error_mean_{hp_type}":gen_error_mean,
                f"gen_error_std_{hp_type}":gen_error_std,
                f"pop_error_mean_{hp_type}":pop_error_mean,
                f"pop_error_std_{hp_type}":pop_error_std,
                f"descent_error_mean_{hp_type}":descent_error_mean,
                f"descent_error_std_{hp_type}":descent_error_std,
    }
    savemat(f"./matfiles/{hp_type}_mnist_ablation.mat", mat_dict)

    # plot
    fig, ax = plt.subplots()
    hp_list = [float(hp) for hp in hp_list]

    ax.set_xscale("log")
    opt_color = np.array([84,172,108])/255 # green
    pop_color = np.array([70,70,70])/255 # grey
    gen_color = np.array([196,78,82])/255 # red
    descent_color = np.array([204,185,116])/255 # yellow
    # ax.set_yscale("log") # not useful since all errors are in similar order
    if plot_type=='pop_opt':
        std_scale = 0.5
        ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)
        ax.set_ylabel('Error', fontsize=18)
        ax.set_xlabel(hp_label, fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(fontsize=18)
    if plot_type=='gen':
        std_scale = 0.5
        ax.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')
        ax.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.set_xlabel(hp_label, fontsize=18)
        # if hp_type=='gamma':
        #     ax.set_ylim([0.0005, 0.004])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(fontsize=18)
        # ax.set_ylabel('Absolute generalization error', fontsize=15)
    if plot_type=='all':
        std_scale = 1.0
        ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')  
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)   
        ax.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')   
        ax.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.errorbar(hp_list, descent_error_mean, std_scale*descent_error_std, fmt='o-', capsize=5,
                     color=descent_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$\mathcal{E}_{ca}$')   
        ax.fill_between(hp_list, descent_error_mean - std_scale*descent_error_std, descent_error_mean + std_scale*descent_error_std, color=descent_color, alpha=0.5)
        ax.set_ylabel('Error', fontsize=18)
        ax.set_xlabel(hp_label+' (log scale)', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if hp_type=='gamma':
            ax.legend(fontsize=18)
    # use left and right sides to plot opt-pop and gen in different scales
    if plot_type=='all-diff-scale':
        std_scale = 0.5
        opt_handle = ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        pop_handle = ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')  
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)  
        ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
        gen_handle = ax2.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')   
        ax2.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='y', which='major', labelsize=15, labelcolor=gen_color)
        ax2.set_ylabel('Generalization', fontsize=18)
        ax2.yaxis.label.set_color(gen_color)
        ax.set_ylabel('Population/Optimization', fontsize=18)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel(hp_label+' (log scale)', fontsize=18)
        if hp_type=='gamma':
            plt.legend(loc='center right', handles=[opt_handle, pop_handle, gen_handle], fontsize=18)
    # use left and right sides to plot opt-pop-gen and descent in different scales
    if plot_type=='all-2':
        std_scale = 0.5
        opt_handle = ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        pop_handle = ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')  
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)  
        gen_handle = ax.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o--', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')   
        ax.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
        descent_handle = ax2.errorbar(hp_list, descent_error_mean, std_scale*descent_error_std, fmt='o-', capsize=5,
                     color=descent_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$\mathcal{E}_{ca}$')   
        ax2.fill_between(hp_list, descent_error_mean - std_scale*descent_error_std, descent_error_mean + std_scale*descent_error_std, color=descent_color, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='y', which='major', labelsize=15, labelcolor=descent_color)
        ax2.set_ylabel('Descent', fontsize=18)
        ax2.yaxis.label.set_color(descent_color)
        ax.set_ylabel('Pop/Opt/Gen', fontsize=18)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel(hp_label+' (log scale)', fontsize=18)
        if hp_type=='gamma':
            plt.legend(loc='center right', handles=[opt_handle, pop_handle, gen_handle, descent_handle], fontsize=18)
        # ax.set_ylabel('Error', fontsize=18)
    # ax.set_xlabel(hp_label, fontsize=18)
    # if plot_type=='gen' and hp_type=='gamma':
    #     ax.set_ylim([0.0005, 0.004])
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.savefig(f'./figures/{hp_type}_{plot_type}_err_comp_new.pdf', bbox_inches='tight')

# calc final loss and errors for methods
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


# plot loss vs epochs
def loss_plots():
    # folder containing data logs for ablation
    folder='./mo_loss_logs/'
    file_list = os.listdir(folder)
    num_epochs = 21

    print(file_list)

    # hp set used for ablation
    moo_method_list = ['EW', 'MGDA', 'MoCo', 'MoDo']
    loss_list = ['Cross-entropy loss', 'L1 loss', 'Hinge loss']

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
                    if ("Epoch" in line) and ("FORMAT" not in line):
                        # print(line)
                        train_loss_list, test_loss_list = line.strip().split(' | ')[1], line.strip().split(' | ')[2]
                        train_loss_list = [float(loss) for loss in train_loss_list.split(': ')[1].strip().split(' ')]
                        test_loss_list = [float(loss) for loss in test_loss_list.split(': ')[1].strip().split(' ')]
                        train_epoch_loss_list.append(train_loss_list)
                        test_epoch_loss_list.append(test_loss_list)
                train_loss[moo_method].append(train_epoch_loss_list)
                test_loss[moo_method].append(test_epoch_loss_list)
        train_loss[moo_method] = np.array(train_loss[moo_method])
        test_loss[moo_method] = np.array(test_loss[moo_method])

    # print loss statistics
    train_loss_mean = {}
    test_loss_mean = {}
    train_loss_std = {}
    test_loss_std = {}

    # calc means and std devs across seeds
    for moo_method in test_loss:
        print(f"\n{moo_method}")
        print("Number of seeds:", len(test_loss[moo_method]))
        print("Mean of train losses:")
        train_loss_mean_ = np.mean(train_loss[moo_method], axis=0)
        train_loss_mean[moo_method] = copy.deepcopy(train_loss_mean_)
        print(train_loss_mean_)
        print("Std. dev of train losses:")
        train_loss_std_ = np.std(train_loss[moo_method], axis=0)
        train_loss_std[moo_method] = copy.deepcopy(train_loss_std_)
        print(train_loss_std_)
        print("Mean of test losses:")
        test_loss_mean_ = np.mean(test_loss[moo_method], axis=0)
        test_loss_mean[moo_method] = copy.deepcopy(test_loss_mean_)
        print(test_loss_mean_)
        print("Std. dev of test losses:")
        test_loss_std_ = np.std(test_loss[moo_method], axis=0)
        test_loss_std[moo_method] = copy.deepcopy(test_loss_std_)
        print(test_loss_std_)
        print()

    epoch_list = 50*np.arange(num_epochs)
    # plot
    moo_method_color_list = ['#ffff99', '#386cb0', '#fdc086', '#beaed4']
    moo_method_marker_list = ['o', '^', 'v', '*'] #['', '', '', ''] #
    # iterate over losses
    for i, loss in enumerate(loss_list):
        # plot all methods in same plot (not plotting all losses together due to difference in magnitudes accross losses)
        fig, ax = plt.subplots()
        for j, moo_method in enumerate(moo_method_list):
            k = 0# start index (to or not to ommit init loss)
            # ax.scatter(epoch_list[k:][::2], test_loss_mean[moo_method][k:,i][::2], color=moo_method_color_list[j], 
            #         marker=moo_method_marker_list[j], edgecolor='k', s=80)   
            ax.plot(epoch_list[k:], train_loss_mean[moo_method][k:,i], color=moo_method_color_list[j], 
                    linewidth=4)
            ax.fill_between(epoch_list[k:], train_loss_mean[moo_method][k:,i] - train_loss_std[moo_method][k:,i], train_loss_mean[moo_method][k:,i] + train_loss_std[moo_method][k:,i], color=moo_method_color_list[j], alpha=0.5)
            ax.plot(epoch_list[k:][::2], test_loss_mean[moo_method][k:,i][::2], moo_method_color_list[j], 
                    linestyle='--', marker=moo_method_marker_list[j], markeredgecolor='k', markersize=15, linewidth=4)
            ax.fill_between(epoch_list[k:], test_loss_mean[moo_method][k:,i] - test_loss_std[moo_method][k:,i], test_loss_mean[moo_method][k:,i] + test_loss_std[moo_method][k:,i], color=moo_method_color_list[j], alpha=0.5)
     
        # ax.plot(epoch_list[1:], test_loss_mean['EW'][1:,1], label="L1")
        # ax.fill_between(epoch_list[1:], test_loss_mean['EW'][1:,1] - test_loss_std['EW'][1:,1], test_loss_mean['EW'][1:,1] + test_loss_std['EW'][1:,1], alpha=0.5)
        # ax.plot(epoch_list[1:], test_loss_mean['EW'][1:,2], label="Hinge")
        # ax.fill_between(epoch_list[1:], test_loss_mean['EW'][1:,2] - test_loss_std['EW'][1:,2], test_loss_mean['EW'][1:,2] + test_loss_std['EW'][1:,2], alpha=0.5)

        train_handle = mlines.Line2D([], [], color='k',
                          markersize=10, label='Train')
        test_handle = mlines.Line2D([], [], color='k', linestyle='--', marker='s',
                          markersize=10, label='Test')
        ew_handle = mlines.Line2D([], [], color='#ffff99', marker='o', markeredgecolor='k',
                          markersize=10, label='Mean')
        mgda_handle = mlines.Line2D([], [], color='#386cb0', marker='^', markeredgecolor='k',
                          markersize=10, label='MGDA')
        moco_handle = mlines.Line2D([], [], color='#fdc086', marker='v', markeredgecolor='k',
                          markersize=10, label='MoCo')
        modo_handle = mlines.Line2D([], [], color='#beaed4', marker='*', markeredgecolor='k',
                          markersize=10, label='MoDo')
        if i==0:
            ax.set_ylabel(f'Loss')
        if i==2:
            ax.legend(handles=[train_handle, test_handle, ew_handle, mgda_handle, moco_handle, modo_handle])
        ax.set_xlabel(f'Epoch')
        ax.set_xticks(epoch_list[::2])
        ax.set_xticklabels([str(int(i)) for i in epoch_list[::2]])
        plt.savefig(f'./figures/{loss}_method_comp.pdf', bbox_inches='tight') #pdf

# plot loss and errors vs epoch
def loss_error_plots():
    # folder containing data logs for ablation
    folder='./mo_loss_error_new_logs/'
    file_list = os.listdir(folder)
    num_epochs = 21

    print(file_list)

    # keywords used in comparisons
    moo_method_list = ['EW', 'MGDA', 'MoCo', 'MoDo'] #, 'MGDA', 'MoCo'
    loss_list = ['Cross-entropy', 'MSE', 'Huber']
    error_type_list = ['pop', 'opt', 'gen']

    # init lists to collect data from different seeds
    train_loss = {moo_method:[] for moo_method in moo_method_list}
    test_loss = {moo_method:[] for moo_method in moo_method_list}
    error = {moo_method:[] for moo_method in moo_method_list}

    # scrape the log files
    for moo_method in moo_method_list:
        for file in file_list:
            if f'{moo_method}-' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                train_epoch_loss_list = []
                test_epoch_loss_list = []
                error_epoch_list = []
                for line in lines:
                    # scrape test and training loss at each iteration from log file
                    if ("Epoch" in line) and ("FORMAT" not in line):
                        # print(line)
                        train_loss_list, test_loss_list, error_list = line.strip().split(' | ')[1], line.strip().split(' | ')[2], line.strip().split(' | ')[3] 
                        train_loss_list = [float(loss) for loss in train_loss_list.split(': ')[1].strip().split(' ')]
                        test_loss_list = [float(loss) for loss in test_loss_list.split(': ')[1].strip().split(' ')]
                        error_list = [float(error) for error in error_list.split(': ')[1].strip().split(' ')]
                        # only consider the absolute generalization error
                        error_list[-1] = abs(error_list[-1])
                        train_epoch_loss_list.append(train_loss_list)
                        test_epoch_loss_list.append(test_loss_list)
                        error_epoch_list.append(error_list)
                train_loss[moo_method].append(train_epoch_loss_list)
                test_loss[moo_method].append(test_epoch_loss_list)
                error[moo_method].append(error_epoch_list)
        train_loss[moo_method] = np.array(train_loss[moo_method])
        test_loss[moo_method] = np.array(test_loss[moo_method])
        error[moo_method] = np.array(error[moo_method])

    # print loss statistics
    train_loss_mean = {}
    test_loss_mean = {}
    train_loss_std = {}
    test_loss_std = {}
    error_mean = {}
    error_std = {}

    # flag to skip plotting MoCo results
    skip_moco = True

    # calc means and std devs across seeds
    for moo_method in test_loss:
        if skip_moco:
            if moo_method=='MoCo':
                continue
        print(f"\n{moo_method}")
        # silent print results from train losses, since these are not in the table
        print("Number of seeds:", len(test_loss[moo_method]))
        # print("Mean of train losses:")
        train_loss_mean_ = np.mean(train_loss[moo_method], axis=0)
        train_loss_mean[moo_method] = copy.deepcopy(train_loss_mean_)
        # print(train_loss_mean_)
        # print("Std. dev of train losses:")

        # print final test loss and error statistics to be put in the table
        train_loss_std_ = np.std(train_loss[moo_method], axis=0)
        train_loss_std[moo_method] = copy.deepcopy(train_loss_std_)
        # print(train_loss_std_)
        print("Mean of test losses:")
        test_loss_mean_ = np.mean(test_loss[moo_method], axis=0)
        test_loss_mean[moo_method] = copy.deepcopy(test_loss_mean_)
        print(test_loss_mean_[-1]*1000)
        print("Std. dev of test losses:")
        test_loss_std_ = np.std(test_loss[moo_method], axis=0)
        test_loss_std[moo_method] = copy.deepcopy(test_loss_std_)
        print(test_loss_std_[-1]*1000)
        print("Mean of errors:")
        error_mean_ = np.mean(error[moo_method], axis=0)
        error_mean[moo_method] = copy.deepcopy(error_mean_)
        print(error_mean_[-1]*1000)
        print("Std. dev of errors:")
        error_std_ = np.std(error[moo_method], axis=0)
        error_std[moo_method] = copy.deepcopy(error_std_)
        print(error_std_[-1]*1000)
        print()

    epoch_list = 50*np.arange(num_epochs)
    # plot
    moo_method_color_list = ['#a65628', '#984ea3', '#4daf4a', '#377eb8'] # first ['#ffff99', '#386cb0', '#fdc086', '#beaed4']
    moo_method_marker_list = ['o', '^', 'v', '*'] #['', '', '', ''] #
    moo_method_name_list = ['Mean', 'MGDA', 'MoCo', 'MoDo']
    # iterate over losses
    for i, loss in enumerate(loss_list):
        # plot all methods in same plot (not plotting all losses together due to difference in magnitudes accross losses)
        fig, ax = plt.subplots()
        for j, moo_method in enumerate(moo_method_list):
            if skip_moco:
                if moo_method=='MoCo':
                    continue
            k = 0# start index (to or not to ommit init loss)
            ax.plot(epoch_list[k:], train_loss_mean[moo_method][k:,i], color=moo_method_color_list[j], 
                    linewidth=4)
            ax.fill_between(epoch_list[k:], train_loss_mean[moo_method][k:,i] - train_loss_std[moo_method][k:,i], train_loss_mean[moo_method][k:,i] + train_loss_std[moo_method][k:,i], color=moo_method_color_list[j], alpha=0.5)
            ax.plot(epoch_list[k:][::2], test_loss_mean[moo_method][k:,i][::2], moo_method_color_list[j], 
                    linestyle='--', marker=moo_method_marker_list[j], markeredgecolor='k', markersize=15, linewidth=4)
            ax.fill_between(epoch_list[k:], test_loss_mean[moo_method][k:,i] - test_loss_std[moo_method][k:,i], test_loss_mean[moo_method][k:,i] + test_loss_std[moo_method][k:,i], color=moo_method_color_list[j], alpha=0.5)

        train_handle = mlines.Line2D([], [], color='k',
                          markersize=10, label='Train')
        test_handle = mlines.Line2D([], [], color='k', linestyle='--', marker='s',
                          markersize=10, label='Test')
        ew_handle = mlines.Line2D([], [], color='#a65628', marker='o', markeredgecolor='k',
                          markersize=10, label='Mean')
        mgda_handle = mlines.Line2D([], [], color='#984ea3', marker='^', markeredgecolor='k',
                          markersize=10, label='MGDA')
        moco_handle = mlines.Line2D([], [], color='#4daf4a', marker='v', markeredgecolor='k',
                          markersize=10, label='MoCo')
        modo_handle = mlines.Line2D([], [], color='#377eb8', marker='*', markeredgecolor='k',
                          markersize=10, label='MoDo')
        if i==0:
            ax.set_ylabel(f'Loss', fontsize=18)
        if i==2:
            if skip_moco:
                ax.legend(handles=[train_handle, test_handle, ew_handle, mgda_handle, modo_handle], fontsize=18)
            else:
                ax.legend(handles=[train_handle, test_handle, ew_handle, mgda_handle, moco_handle, modo_handle], fontsize=18)
        ax.set_xlabel(f'Epoch', fontsize=18)
        ax.set_xticks(epoch_list[::4])
        ax.set_xticklabels([str(int(i)) for i in epoch_list[::4]])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if skip_moco:
            plt.savefig(f'./figures/{loss}_method_comp_no_moco.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'./figures/{loss}_method_comp.pdf', bbox_inches='tight')

    moo_method_color_list = ['#a65628', '#984ea3', '#4daf4a', '#377eb8']
    moo_method_marker_list = ['o', '^', 'v', '*'] #['', '', '', ''] #

    # iterate over losses
    for i, error_type in enumerate(error_type_list):
        # plot all methods in same plot (not plotting all losses together due to difference in magnitudes accross losses)
        fig, ax = plt.subplots()
        for j, moo_method in enumerate(moo_method_list):
            if skip_moco:
                if moo_method=='MoCo':
                    continue
            k = 0# start index (to or not to ommit init loss)
            ax.errorbar(epoch_list[k:], error_mean[moo_method][k:,i], error_std[moo_method][k:,i], 
                        fmt=moo_method_marker_list[j]+'-', capsize=5, 
                        color=moo_method_color_list[j], markeredgecolor='k', markersize=5, label=moo_method_name_list[j])
            ax.plot(epoch_list[k:], error_mean[moo_method][k:,i], color=moo_method_color_list[j], 
                    markeredgecolor='k', markersize=15, linewidth=4)
            ax.fill_between(epoch_list[k:], error_mean[moo_method][k:,i] - error_std[moo_method][k:,i], error_mean[moo_method][k:,i] + error_std[moo_method][k:,i], 
                            color=moo_method_color_list[j], alpha=0.25)

        if i==0:
            ax.set_ylabel(f'Error', fontsize=18)
        if i==2:
            pass
            # ax.legend() # moved the legend to descent direction plot
        ax.set_xlabel(f'Epoch', fontsize=18)
        ax.set_xticks(epoch_list[::4])
        ax.set_xticklabels([str(int(i)) for i in epoch_list[::4]])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if skip_moco:
            plt.savefig(f'./figures/{error_type}_error_method_comp_no_moco.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'./figures/{error_type}_error_method_comp.pdf', bbox_inches='tight')

# plot descent direction error vs epoch
def descent_error_plots():
    # folder containing data logs for ablation
    folder='./mo_descent_error_new_logs/'
    file_list = os.listdir(folder)
    num_epochs = 21

    print(file_list)

    # keywords used in comparisons
    moo_method_list = ['EW', 'MGDA', 'MoCo', 'MoDo'] #, 'MGDA', 'MoCo'
    loss_list = ['Cross-entropy', 'MSE', 'Huber']

    # init lists to collect data from different seeds
    descent_error = {moo_method:[] for moo_method in moo_method_list}

    # scrape the log files
    for moo_method in moo_method_list:
        for file in file_list:
            if f'{moo_method}-' in file:
                foo = open(folder+file)
                lines = foo.read().split('\n')
                descent_error_epoch_list = []
                for line in lines:
                    # scrape test and training loss at each iteration from log file
                    if ("Epoch" in line) and ("FORMAT" not in line):
                        print(line)
                        descent_error_ = float(line.split(': ')[2].strip())
                        print(descent_error_)
                        descent_error_epoch_list.append(descent_error_)
                descent_error[moo_method].append(descent_error_epoch_list)
        descent_error[moo_method] = np.array(descent_error[moo_method])

    # print loss statistics
    descent_error_mean = {}
    descent_error_std = {}

    # flag to skip plotting MoCo results
    skip_moco = True

    # calc means and std devs across seeds
    for moo_method in descent_error:
        if skip_moco:
            if moo_method=='MoCo':
                continue
        print(f"\n{moo_method}")
        # silent print results from train losses, since these are not in the table
        print("Number of seeds:", len(descent_error[moo_method]))
        print("Mean of errors:")
        descent_error_mean_ = np.mean(descent_error[moo_method], axis=0)
        descent_error_mean[moo_method] = copy.deepcopy(descent_error_mean_)
        print(descent_error_mean_[-1]*1000)
        print("Std. dev of errors:")
        descent_error_std_ = np.std(descent_error[moo_method], axis=0)
        descent_error_std[moo_method] = copy.deepcopy(descent_error_std_)
        print(descent_error_std_[-1]*1000)
        print()

    moo_method_color_list = ['#a65628', '#984ea3', '#4daf4a', '#377eb8']
    moo_method_marker_list = ['o', '^', 'v', '*'] #['', '', '', ''] #
    moo_method_name_list = ['Mean', 'MGDA', 'MoCo', 'MoDo']

    epoch_list = 50*np.arange(num_epochs)

    # plot all methods in same plot (not plotting all losses together due to difference in magnitudes accross losses)
    fig, ax = plt.subplots()
    for j, moo_method in enumerate(moo_method_list):
        if skip_moco:
            if moo_method=='MoCo':
                continue
        k = 0# start index (to or not to ommit init loss)
        ax.errorbar(epoch_list[k:], descent_error_mean[moo_method][k:], descent_error_std[moo_method][k:], 
                    fmt=moo_method_marker_list[j]+'-', capsize=5, 
                    color=moo_method_color_list[j], markeredgecolor='k', markersize=5, label=moo_method_name_list[j])
        ax.plot(epoch_list[k:], descent_error_mean[moo_method][k:], color=moo_method_color_list[j], 
                markeredgecolor='k', markersize=15, linewidth=4)
        ax.fill_between(epoch_list[k:], descent_error_mean[moo_method][k:] - descent_error_std[moo_method][k:], 
                        descent_error_mean[moo_method][k:] + descent_error_std[moo_method][k:], 
                        color=moo_method_color_list[j], alpha=0.25)

    ax.legend(fontsize=18)
    ax.set_xlabel(f'Epoch', fontsize=18)
    ax.set_xticks(epoch_list[::4])
    ax.set_xticklabels([str(int(i)) for i in epoch_list[::4]])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if skip_moco:
        plt.savefig(f'./figures/descent_error_method_comp_no_moco.pdf', bbox_inches='tight')
    else:
        plt.savefig(f'./figures/descent_error_method_comp.pdf', bbox_inches='tight')

def T_ablation(plot_type='all'):
    # folder containing data logs for ablation
    folder='./modo_T_ablation_new_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # init lists to collect data from different seeds
    train_loss = []
    test_loss = []
    error = []

    hp_list = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
    hp_label = r'$T$'
    hp_type = 'T'

    # scrape the log files
    # for moo_method in moo_method_list:
    for file in file_list:
        # if f'{moo_method}-' in file:
        foo = open(folder+file)
        lines = foo.read().split('\n')
        train_epoch_loss_list = []
        test_epoch_loss_list = []
        error_epoch_list = []
        for line in lines:
            # scrape test and training loss at each iteration from log file
            if ("Epoch" in line) and ("FORMAT" not in line) and ("Train" in line):
                # print(line)
                train_loss_list, test_loss_list, error_list = line.strip().split(' | ')[1], line.strip().split(' | ')[2], line.strip().split(' | ')[3] 
                train_loss_list = [float(loss) for loss in train_loss_list.split(': ')[1].strip().split(' ')]
                test_loss_list = [float(loss) for loss in test_loss_list.split(': ')[1].strip().split(' ')]
                error_list = [float(error) for error in error_list.split(': ')[1].strip().split(' ')]
                # only consider the absolute generalization error
                error_list[-1] = abs(error_list[-1])
                train_epoch_loss_list.append(train_loss_list)
                test_epoch_loss_list.append(test_loss_list)
                error_epoch_list.append(error_list)
        train_loss.append(train_epoch_loss_list)
        test_loss.append(test_epoch_loss_list)
        error.append(error_epoch_list)
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    error = np.array(error)

    print(error.shape)


    # calc means and std devs across seeds

    # silent print results from train losses, since these are not in the table
    print("Number of seeds:", len(test_loss))
    # print("Mean of train losses:")
    train_loss_mean = np.mean(train_loss, axis=0)
    # print(train_loss_mean_)
    # print("Std. dev of train losses:")

    # print final test loss and error statistics to be put in the table
    train_loss_std = np.std(train_loss, axis=0)
    # print(train_loss_std_)
    print("Mean of test losses:")
    test_loss_mean = np.mean(test_loss, axis=0)
    print(test_loss_mean[-1]*1000)
    print("Std. dev of test losses:")
    test_loss_std = np.std(test_loss, axis=0)
    print(test_loss_std[-1]*1000)
    print("Mean of errors:")
    error_mean = np.mean(error, axis=0)
    print(error_mean[-1]*1000)
    print("Std. dev of errors:")
    error_std = np.std(error, axis=0)
    print(error_std[-1]*1000)
    print()


    # folder containing data logs for ablation
    folder=f'./modo_T_ablation_descent_logs/'
    file_list = os.listdir(folder)

    print(file_list)

    # init lists to collect data from different seeds
    descent_error = []

    # scrape the log files
    for file in file_list:
        foo = open(folder+file)
        lines = foo.read().split('\n')
        print(f'\n{file}')
        # to caclualte iteration Eca from moving avaerage recorded
        count=0
        descent_error_epoch_list = []
        cum_sum = 0
        for line in lines:
            # scrape final descent error data from log file
            if ("Epoch" in line) and ("Descent Error:" in line) and ("EPOCH" not in line):
                
                count += 1
                descent_error_ = float(line.strip().split(': ')[2]) * count - cum_sum
                descent_error_epoch_list.append(descent_error_**2)
                print(cum_sum)
                cum_sum = float(line.strip().split(': ')[2]) * count
                print(line, descent_error_)
        descent_error.append(descent_error_epoch_list)
    descent_error = np.array(descent_error) 

    descent_error_mean = np.mean(descent_error, axis=0)
    descent_error_std = np.std(descent_error, axis=0)       

    print('descent error', descent_error.shape)
    start_idx=3
    n_smooth=4
    pop_error_mean = smooth(error_mean[:, 0][start_idx:], n=n_smooth)
    opt_error_mean = smooth(error_mean[:, 1][start_idx:], n=n_smooth)
    gen_error_mean = smooth(error_mean[:, 2][start_idx:], n=n_smooth)
    descent_error_mean = smooth(descent_error_mean[start_idx:], n=n_smooth)

    # save arrays for plotting 


    pop_error_std = smooth(error_std[:, 0][start_idx:], n=n_smooth)
    opt_error_std = smooth(error_std[:, 1][start_idx:], n=n_smooth)
    gen_error_std = smooth(error_std[:, 2][start_idx:], n=n_smooth)
    descent_error_std = smooth(descent_error_std[start_idx:], n=n_smooth)

    hp_list = hp_list[start_idx:]

    print(len(hp_list), gen_error_mean.shape)

    # save for plotting
    mat_dict = {"T_list":np.array(hp_list), 
                "opt_error_mean_T":opt_error_mean,
                "opt_error_std_T":opt_error_std,
                "gen_error_mean_T":gen_error_mean,
                "gen_error_std_T":gen_error_std,
                "pop_error_mean_T":pop_error_mean,
                "pop_error_std_T":pop_error_std,
                "descent_error_mean_T":descent_error_mean,
                "descent_error_std_T":descent_error_std,
    }
    savemat(f"./matfiles/T_mnist_ablation.mat", mat_dict)

    # plot
    fig, ax = plt.subplots()
    hp_list = [float(hp) for hp in hp_list]

    ax.set_xscale("log")
    opt_color = np.array([84,172,108])/255 # green
    pop_color = np.array([70,70,70])/255 # grey
    gen_color = np.array([196,78,82])/255 # red
    descent_color = np.array([204,185,116])/255 # yellow
    # ax.set_yscale("log") # not useful since all errors are in similar order
    if plot_type=='pop_opt':
        std_scale = 0.5
        ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)
        ax.set_ylabel('Error', fontsize=18)
        ax.set_xlabel(hp_label, fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(fontsize=18)
    if plot_type=='gen':
        std_scale = 0.5
        ax.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')
        ax.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.set_xlabel(hp_label, fontsize=18)
        # if hp_type=='gamma':
        #     ax.set_ylim([0.0005, 0.004])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(fontsize=18)
        # ax.set_ylabel('Absolute generalization error', fontsize=15)
    if plot_type=='all':
        std_scale = 1.0
        ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')  
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)   
        ax.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')   
        ax.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.set_ylabel('Error', fontsize=18)
        ax.set_xlabel(hp_label+' (log scale)', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if hp_type=='lr':
            ax.legend(fontsize=18)
    # use left and right sides to plot opt-pop and gen in different scales
    if plot_type=='all-diff-scale':
        std_scale = 0.5
        opt_handle = ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        pop_handle = ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')  
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)  
        ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
        gen_handle = ax2.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o-', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')   
        ax2.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='y', which='major', labelsize=15, labelcolor=gen_color)
        ax2.set_ylabel('Generalization', fontsize=18)
        ax2.yaxis.label.set_color(gen_color)
        ax.set_ylabel('Population/Optimization', fontsize=18)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel(hp_label+' (log scale)', fontsize=18)
        if hp_type=='lr':
            plt.legend(handles=[opt_handle, pop_handle, gen_handle], fontsize=18)
    if plot_type=='all-2':
        std_scale = 0.5
        opt_handle = ax.errorbar(hp_list, opt_error_mean, std_scale*opt_error_std, fmt='o-', capsize=5, 
                    color=opt_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{opt}$')
        ax.fill_between(hp_list, opt_error_mean - std_scale*opt_error_std, opt_error_mean + std_scale*opt_error_std, color=opt_color, alpha=0.5)
        pop_handle = ax.errorbar(hp_list, pop_error_mean, std_scale*pop_error_std, fmt='o--', capsize=5, 
                    color=pop_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$R_{pop}$')  
        ax.fill_between(hp_list, pop_error_mean - std_scale*pop_error_std, pop_error_mean + std_scale*pop_error_std, color=pop_color, alpha=0.5)  
        gen_handle = ax.errorbar(hp_list, gen_error_mean, std_scale*gen_error_std, fmt='o--', capsize=5,
                     color=gen_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$|R_{gen}|$')   
        ax.fill_between(hp_list, gen_error_mean - std_scale*gen_error_std, gen_error_mean + std_scale*gen_error_std, color=gen_color, alpha=0.5)
        ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
        descent_handle = ax2.errorbar(hp_list, descent_error_mean, std_scale*descent_error_std, fmt='o-', capsize=5,
                     color=descent_color,markeredgecolor='k', markersize=5, linewidth=4, elinewidth=1, label=r'$\mathcal{E}_{ca}$')   
        ax2.fill_between(hp_list, descent_error_mean - std_scale*descent_error_std, descent_error_mean + std_scale*descent_error_std, color=descent_color, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='y', which='major', labelsize=15, labelcolor=descent_color)
        ax2.set_ylabel('Descent', fontsize=18)
        ax2.yaxis.label.set_color(descent_color)
        ax.set_ylabel('Pop/Opt/Gen', fontsize=18)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel(hp_label+' (log scale)', fontsize=18)
        
        if hp_type=='gamma':
            plt.legend(loc='center right', handles=[opt_handle, pop_handle, gen_handle, descent_handle], fontsize=18)
        # ax.set_ylabel('Error', fontsize=18)
    # ax.set_xlabel(hp_label, fontsize=18)
    # if plot_type=='gen' and hp_type=='gamma':
    #     ax.set_ylim([0.0005, 0.004])
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.savefig(f'./figures/{hp_type}_{plot_type}_err_comp_new.pdf', bbox_inches='tight')
    
if __name__=='__main__':    
    # rho ablation
    # rho_ablation(plot_type='pop_opt')

    # gamma ablation
    # gamma_ablation(plot_type='gen')

    # hp ablation (gamma, rho, lr)
    hp_ablation(plot_type='all-2', hp_type='gamma')

    # T ablation
    # T_ablation(plot_type='all-2')

    # perf. metric calc.
    # loss_calc()

    # plot loss curves
    # loss_plots()

    # plot loss and error (pop, opt, and gen) curves
    # loss_error_plots()

    # plot descent error curves
    # descent_error_plots()