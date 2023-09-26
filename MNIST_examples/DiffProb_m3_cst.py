import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.abspath("."))

from MNIST_examples.utilities.model_utilities import load_results

load_case = "part"
attack_names = [
    "attack014",
    "attack018",
    "attack013",
    "attack017",
    "attack012",
    "attack016",
    "attack011",
    "attack015",
    "attack005"
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
if os.path.abspath('.').endswith('MNIST_examples'):
    # debug...
    root = os.path.abspath('..')
    log_path = os.path.join(root, 'MNIST_examples', 'logs')
elif os.path.abspath('.').endswith('Byzantine_attack'):
    # run through bash in root.
    root = os.path.abspath('.')
    log_path = os.path.join(root, 'MNIST_examples', 'logs')
elif os.path.abspath('.').endswith('L3GD_V2'):
    # run through bash in root.
    root = os.path.abspath('.')
    log_path = os.path.join(root, 'MNIST_examples', 'logs')  

for attack_case in attack_names:
    for filename_in_logpath in os.listdir(log_path):
        if attack_case in filename_in_logpath and '.pickle' in filename_in_logpath:
            load_names.append(filename_in_logpath)
# print(load_names)

hyperpara = {
    "alg": ['grad_desc', 'normalized_grad_desc']
}

if load_case == "part":
    regret_list = {log_name: [] for log_name in load_names}
    for pickle_case in load_names:
        _, _, regret_list[pickle_case] = load_results(pickle_case)

fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as per your requirements
for pickle_case in load_names:
    if "attack014" in pickle_case:
        alpha = 1  # attack_prob=0.5
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='p=0.5')
            elif alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'r', alpha=alpha, label='p=0.5')
    elif "attack018" in pickle_case:  # disappeared?
        alpha = 1  # attack_prob=0.5
        for alg in hyperpara["alg"]:
            if alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'b', alpha=alpha, label='p=0.5')
    elif "attack013" in pickle_case: # disappeared?
        alpha = 0.7  # attack_prob=0.3
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='p=0.4')
            elif alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'r', alpha=alpha, label='p=0.4')
    elif "attack017" in pickle_case:
        alpha = 0.7  # attack_prob=0.3
        for alg in hyperpara["alg"]:
            if alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'b', alpha=alpha, label='p=0.4')
    elif "attack012" in pickle_case:
        alpha = 0.4  # attack_prob=0.4
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='p=0.3')
            elif alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'r', alpha=alpha, label='p=0.3')
    elif "attack016" in pickle_case:
        alpha = 0.4  # attack_prob=0.4
        for alg in hyperpara["alg"]:
            if alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'b', alpha=alpha, label='p=0.3')
    elif "attack011" in pickle_case:
        alpha = 0.1
        for alg in hyperpara["alg"]:
            if alg == 'grad_desc':
                avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
                ax.plot(avg_reg_gd, 'k', alpha=alpha, label='p=0.2')
            elif alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'r', alpha=alpha, label='p=0.2')
    elif "attack015" in pickle_case:
        alpha = 0.1  # attack_prob=0.5
        for alg in hyperpara["alg"]:
            if alg == 'normalized_grad_desc':
                avg_reg_ngd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['normalized_grad_desc'])]
                ax.plot(avg_reg_ngd, 'b', alpha=alpha, label='p=0.2')
    elif "attack005" in pickle_case:  # the OMGD without attack.
        avg_reg_gd = [val / (t + 1) for t, val in enumerate(regret_list[pickle_case]['grad_desc'])]
        ax.plot(avg_reg_gd, 'k--', alpha=1, label='No attack')

ax.plot(np.NaN, np.NaN, label='OMGD', c='black')
ax.plot(np.NaN, np.NaN, label='OMNGD', c='red')
ax.plot(np.NaN, np.NaN, label='ONGD', c='blue')
leg = ax.legend(loc='upper right')
ax.set_xlabel(r'time (T)')
ax.set_ylabel(r'$\sum_{t=1}^T f_t(x_t)/T$')
plt.ylim([60, 200])
plt.xlim([2, 1000])
fig_name = 'CstDiffProb_m3'
plt.tight_layout()  # Adjusts the layout to prevent labels from being cut off
plt.savefig(root + '/MNIST_examples/assets/results_{}.png'.format(fig_name), dpi=300)
plt.show()
