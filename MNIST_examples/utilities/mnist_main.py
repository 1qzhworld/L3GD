import logging
import numpy as np
from datetime import datetime
import os
import sys
# import pprint

# print('added path: ' + os.path.abspath("."))
sys.path.append(os.path.abspath("."))
# pprint.pprint(sys.path)
# local libraries
from MNIST_examples.utilities.args import parse_args
from MNIST_examples.utilities.mnist_utilities import mnist_load_config, load_mnist, train
from MNIST_examples.utilities.model_utilities import save_model, load_model, initialize, save_results, load_results
from MNIST_examples.utilities.plot_utilities import plot_lists


def main(args, **kwargs):
    if 'root_path' in kwargs.keys():
        root_path = kwargs.pop('root_path')
        os.chdir(root_path)
    start_time = datetime.now()
    log_name = start_time.strftime(args.config + '_%Y%m%d_%H%M')
    logging.basicConfig(filename="./MNIST_examples/logs/{}.log".format(log_name),
                        filemode="w", format="%(message)s", level=logging.INFO)
    fh = logging.FileHandler(filename='./MNIST_examples/logs/{}.log'.format(log_name))

    # hyper-parameters are saved in file named
    cfg_idx = args.config  # postfix of config file
    cfg_name = 'config_mnist_{}.json'.format(cfg_idx)
    hyperpara = mnist_load_config(cfg_name)
    # loading MNIST data
    x_train, y_train, x_test, y_test = load_mnist('./MNIST_examples/dataset/MNISTdata.hdf5', hyperpara)

    # setting the random seed
    np.random.seed(args.rand_seed)

    # initialize the parameters
    num_inputs = x_train.shape[1]  # size of each figure (input dimension)
    num_classes = len(set(y_train))  # number of class (output dimension)
    if hyperpara['load']:
        param = load_model(args.load_name)
    else:
        param = initialize(hyperpara, num_inputs, num_classes)  # param is dict with key 'w', 'b'

    # train the model
    loss_list, accu_list, regret_list = train(param, hyperpara, x_train, y_train, x_test, y_test, load_name=log_name)
    # plot the loss and accuracy
    plot_lists(loss_list, accu_list, regret_list, hyperpara, log_name)

    # save model
    if hyperpara['save']:
        save_model(log_name, param)
        save_results(loss_list, accu_list, regret_list, log_name)

    logging.info('End at time: {}'.format(datetime.now().strftime(args.config + '_%Y%m%d_%H%M.log')))
    logging.getLogger().removeHandler(fh)

    return loss_list, accu_list, regret_list


if __name__ == "__main__":
    args_default = parse_args()
    print('current settings')
    print('Current path.{}'.format(os.path.abspath('.')))
    print(args_default)
    if args_default.config == 'attack_test':

        loss_list, accu_list, regret_list = main(args=args_default, root_path=os.path.abspath("../.."))
    else:
        loss_list, accu_list, regret_list = main(args=args_default)



