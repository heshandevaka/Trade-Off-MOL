import matplotlib.pyplot as plt
import numpy as np

T0 = 10000
alpha0 = 0.001
gamma0 = 0.001

T_set = 10**np.arange(1, 4, 0.2)
alpha_set = 10**(-np.arange(0, 4, 0.2))
gamma_set = 10**(-np.arange(0, 4, 0.2))

n0 = T_set[-1]**(1.5)

R_pop_alpha = lambda gamma, T: alpha_set**(-0.5) * T**(-0.5) + alpha_set**0.5 + gamma**0.5 + T**0.5 * n0
R_pop_gamma = lambda alpha, T: alpha**(-0.5) * T**(-0.5) + alpha**0.5 + gamma_set**0.5 + T**0.5 * n0
R_pop_T = lambda alpha, gamma: alpha**(-0.5) * T_set**(-0.5) + alpha**0.5 + gamma**0.5 + T_set**0.5 * n0

# alpha
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.plot(alpha_set, R_pop_alpha(gamma0, T0))
ax.set_xlabel(r'$\alpha$', fontsize=18)
ax.set_ylabel(r'$R_{pop}$', fontsize=18)
plt.savefig(f'./figures/non_convex_theory_alpha.pdf', bbox_inches='tight')

# gamma
fig, ax = plt.subplots()
# ax.set_xscale("log")
ax.plot(gamma_set, R_pop_gamma(alpha0, T0))
ax.set_xlabel(r'$\gamma$', fontsize=18)
ax.set_ylabel(r'$R_{pop}$', fontsize=18)
plt.savefig(f'./figures/non_convex_theory_gamma.pdf', bbox_inches='tight')

# T
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.plot(T_set, R_pop_T(alpha0, gamma0))
ax.set_xlabel(r'$T$', fontsize=18)
ax.set_ylabel(r'$R_{pop}$', fontsize=18)
plt.savefig(f'./figures/non_convex_theory_T.pdf', bbox_inches='tight')

# # test
# fig, ax = plt.subplots()
# x = np.arange()