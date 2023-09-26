import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.abspath("."))

from MNIST_examples.utilities.model_utilities import load_results
"""
redirect to: MNIST_ROL06/DiffMag_p01.py
"""




load_case = "part"  # ["all", "part"]

# 'attack000', Kt='thm_cst', attack_prob=0.1, attack_magnitude=11
# 'attack001', Kt='thm_cst', attack_prob=0.1, attack_magnitude=9
# 'attack002', Kt='thm_cst', attack_prob=0.1, attack_magnitude=7
# 'attack003', Kt='thm_cst', attack_prob=0.1, attack_magnitude=5
# 'attack004', Kt='thm_cst', attack_prob=0.1, attack_magnitude=3
# 'attack005', Kt='thm_cst', attack_prob=0, baseline, only OMGD
# 'attack006', Kt='1', attack_prob=0.1, attack_magnitude=11, only ONGD
# 'attack007', Kt='1', attack_prob=0.1, attack_magnitude=9, only ONGD
# 'attack008', Kt='1', attack_prob=0.1, attack_magnitude=7, only ONGD
# 'attack009', Kt='1', attack_prob=0.1, attack_magnitude=5, only ONGD
# 'attack010'  Kt='1', attack_prob=0.1, attack_magnitude=3, only ONGD

attack_names = [
    'attack004',
    'attack010',
    'attack003',
    'attack009',
    'attack002',
    'attack008',
    'attack001',
    'attack007',
    'attack000',
    'attack006',
    'attack005'
]
load_names = []
# print(os.path.abspath('.'))  # where the code runs, instead of where the file

# for running in root
# root = os.path.abspath('.')
# log_path = root + '/MNIST_examples/logs/'
# during debug # log_path = os.path.abspath('./logs')
# root = os.path.abspath('./..')
# log_path = root + '/MNIST_examples/logs/'
# print(log_path)
# print(os.path.abspath('.'))
if os.path.abspath('.').endswith('MNIST_examples'):
    # debug...
    root = os.path.abspath('./..')
    log_path = root + '/MNIST_examples/logs/'
elif os.path.abspath('.').endswith('Byzantine_attack'):
    # run through bash in root.
    root = os.path.abspath('.')
    log_path = root + '/MNIST_examples/logs/'
elif os.path.abspath('.').endswith('L3GD_V2'):
    # run through bash in root.
    root = os.path.abspath('.')
    log_path = root + '/MNIST_examples/logs/'

print(root)
print(log_path)

for attack_case in attack_names:
    for filename_in_logpath in os.listdir(log_path):
        if attack_case in filename_in_logpath and '.pickle' in filename_in_logpath:
            load_names.append(filename_in_logpath)

hyperpara = {
    "alg": ['grad_desc', 'normalized_grad_desc']
}

if load_case == "part":
    regret_list = {log_name: [] for log_name in load_names}
    for pickle_case in load_names:
        _, _, regret_list[pickle_case] = load_results(pickle_case)

fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as per your requirements
for pickle_case in load_names:
    if "attack004" in pickle_case:
        alpha = 0.25
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='m=3')
    elif "attack003" in pickle_case:
        alpha = 0.4
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='m=5')
    elif "attack002" in pickle_case:
        alpha = 0.55
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='m=7')
    elif "attack001" in pickle_case:
        alpha = 0.7
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='m=9')
    elif "attack000" in pickle_case:
        alpha = 0.9
        print(hyperpara["alg"])
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='m=11')
            elif alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in
                               enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'r', alpha=alpha, label='m=11')
    elif "attack006" in pickle_case:
        alpha = 0.9  # ONGD: GaoXiang2018 (OMNGD with Kt=1)
        for alg in hyperpara["alg"]:
            if alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'b', alpha=alpha, label='m=11')
    elif "attack005" in pickle_case:  # the OMGD without attack.
        avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
        ax.plot(avg_reg_gd, 'k--', alpha=1, label='No attack')

# ax2 = ax.twinx()
# ax2.plot(np.NaN, np.NaN, label='OMGD', c='black')
# ax2.plot(np.NaN, np.NaN, label='OMNGD', c='red')
# ax2.plot(np.NaN, np.NaN, label='ONGD', c='blue')
ax.plot(np.NaN, np.NaN, label='OMGD', c='black')
ax.plot(np.NaN, np.NaN, label='OMNGD', c='red')
ax.plot(np.NaN, np.NaN, label='ONGD', c='blue')

leg = ax.legend(loc='upper right')
ax.set_xlabel(r'time (T)')
ax.set_ylabel(r'$\sum_{t=1}^T f_t(x_t)/T$')
plt.ylim([60, 200])
plt.xlim([2, 1000])
fig_name = 'CstDiffMagnitude_p01'
plt.tight_layout()  # Adjusts the layout to prevent labels from being cut off
plt.savefig(root + '/MNIST_examples/assets/results_{}.png'.format(fig_name), dpi=300)
plt.show()
