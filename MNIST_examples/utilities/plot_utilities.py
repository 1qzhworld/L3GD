import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

from MNIST_examples.utilities.model_utilities import load_results


def plot_lists(loss_list, accu_list, reg_list, hyp, log_name):
    """store the plots during training"""
    # epoch_list = list(range(len(loss_list)))
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    for alg in hyp['alg']:
        if alg == 'grad_desc':
            gd_l = axs[0].semilogy(loss_list[alg], 'k--', label='GD')
            gd_a = axs[1].plot(accu_list[alg], 'k--', label='GD')
            gd_r = axs[2].plot(reg_list[alg], 'k--', label='GD')
        elif alg == 'normalized_grad_desc':
            ngd_l = axs[0].semilogy(loss_list[alg], 'r', label='NGD')
            ngd_a = axs[1].plot(accu_list[alg], 'r', label='NGD')
            ngd_r = axs[2].plot(reg_list[alg], 'r', label='NGD')

    axs[0].set_ylabel('Loss Fun')
    axs[0].set_xlabel('Batch')
    axs[0].set_title('Loss')

    axs[1].set_ylabel('Test Accu')
    axs[1].set_xlabel('Batch')
    axs[1].set_title('Accuracy')

    axs[2].set_ylabel('Regret')
    axs[2].set_xlabel('Batch')
    axs[2].set_title('Regret')
    # axs[2].set_ylim([])

    fig.suptitle(log_name)
    if os.path.abspath('.').endswith('Byzantine_attack'):
        root = os.path.abspath('.')
    else:
        pardir = os.path.split(os.path.abspath('.'))[0]
        while 'Byzantine_attack' in os.path.abspath('.'):
            if pardir.endswith('Byzantine_attack'):
                root = pardir
            else:
                pardir = os.path.split(pardir)[0]
    assert root.endswith('Byzantine_attack')
    assets_path = root + '/MNIST_examples/assets/'
    plt.savefig(assets_path+'results_{}.png'.format(log_name), dpi=300)
    # plt.show()


def compare_all_in_logs():
    """
    compare all models in logs/XXX.pickle
    :return:
    """
    hyperpara = {
        "alg": ['grad_desc', 'normalized_grad_desc']
    }

    pickle_files = [path for path in os.listdir('../logs') if path.endswith('.pickle')]
    print(pickle_files)
    total_number = len(pickle_files)
    row_n = int(np.sqrt(total_number))
    col_n = int(np.sqrt(total_number)) + 1

    fig, axs = plt.subplots(row_n, col_n, constrained_layout=True)

    plot_time = datetime.now()
    save_name = 'compare_all_{}'.format(plot_time.strftime('_%Y%m%d_%H%M'))

    for ind, pickle_file in enumerate(pickle_files):
        r = ind // col_n
        c = ind % col_n

        # print('row is {} and col is {}'.format(r, c))
        # print('----')

        log_name = pickle_file.split('.')[0]
        #
        _, _, reg_list = load_results(log_name)
        for alg in hyperpara['alg']:
            if alg == 'grad_desc':
                gd_r = axs[r, c].plot(reg_list[alg], 'k--', label='GD')
            elif alg == 'normalized_grad_desc':
                ngd_r = axs[r, c].plot(reg_list[alg], 'r', label='NGD')

        axs[r, c].set_ylabel('Regret')
        axs[r, c].set_xlabel('Batch')
        axs[r, c].set_title(log_name)
        plt.savefig('../assets/results_{}.png'.format(save_name))
        plt.show()