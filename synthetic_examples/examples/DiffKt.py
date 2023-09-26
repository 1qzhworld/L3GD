from scipy.special import lambertw
import numpy as np
import os
import matplotlib.pyplot as plt
T = int(1e6)
Kt_dim = np.zeros(T)
Kt_cst = np.zeros(T)
Kt_cst_lem = np.zeros(T)
alpha = 0.5
for t in range(1, T+1):
    Kt_dim[t-1] = (-(t**(1-alpha))*lambertw(-(1/(np.exp(1)*(t**(1-alpha)))), k=-1))
    Kt_cst[t-1] = np.sqrt(t)*np.log(t)
    Kt_cst_lem[t-1] = np.log(1/(t**(1-alpha))) / np.log(1-1/(t**(1/4)))
plt.semilogy(Kt_dim, 'green', label=r'$K_t^{dim}$')
# plt.semilogy(np.arange(T), 'g--')
# plt.semilogy(Kt_cst, 'r--', label='cst')
plt.semilogy(Kt_cst_lem, 'r', label=r'$K_t^{cst}$')
# plt.legend(loc=4)
# plt.show()

# D = [0.1, 1, 10, 100, 1000]
plt.semilogy((Kt_dim-Kt_cst_lem), 'k--', label='Diff')
plt.legend(loc=4)
fig_name = 'Diff_Kt'
print(os.path.abspath('../../'))
# try:
#     plt.savefig('../../synthesis_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
# except:
#     plt.savefig('./synthesis_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
# plt.savefig('../../synthesis_examples/assets/results_syn_{}.png'.format(fig_name), dpi=300)
plt.xlim([0, T])
plt.savefig('./results_syn_{}.png'.format(fig_name), dpi=300)

plt.show()




