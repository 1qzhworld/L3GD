import numpy as np
import copy
import pickle
import os

def initialize(hyp, num_inputs, num_classes):
    """
    initialization of parameters
    :param hyp: the hyper-parameter in the config file
    :param num_inputs: dimension of input (should be (10*784) for MNIST)
    :param num_classes: dimension of output (should be (10*1) for MNIST)
    :return: parameter include the weight and the bias.
    """
    # num_inputs = 28*28 = 784
    # num_classes = 10
    w = np.random.randn(num_classes, num_inputs) / np.sqrt(
        num_classes * num_inputs)  # (10*784) w.T: from input to output
    b = np.random.randn(num_classes, 1) / np.sqrt(num_classes)  # (10*1) bias before output
    param0 = {
        'w': w,  # (10*784) random initialization
        'b': b  # (10*1) interception of the algorithm
    }
    param = {}  # read parameter for each algorithm with param[alg]['w'] and param[alg]['b']
    for alg in hyp['alg']:
        param[alg] = copy.deepcopy(param0)  # initial each algorithm with the same w and b.
    return param


def save_model(save_name, param):
    # because the param is a dict with narray inside, and the narray is not suitable to save with json.
    # # Data to be written
    # # Serializing json
    # json_object = json.dumps(param, indent=4)
    #
    # # Writing to filename with config filename and time (log file) related.
    # save_filename = "./models/{}.json".format(save_name)
    # with open(save_filename, "w") as outfile:
    #     outfile.write(json_object)
    # from: https://stackoverflow.com/questions/30811918/saving-dictionary-of-numpy-arrays
    root = os.path.abspath('./MNIST_examples')
    root += '/models/'
    path2file = root + "{}.npy".format(save_name)
    np.save(path2file, param)


def load_model(load_name):
    path2file = "../models/{}.npy".format(load_name)
    param_tmp = np.load(path2file, allow_pickle=True)
    return param_tmp.item()


def save_results(test_loss_list, test_accuracy_list, regret_list, filename):
    """
    save loss, accuracy and regret list in pickle file.
    :param test_loss_list:
    :param test_accuracy_list:
    :param regret_list:
    :param filename: pickle file name (stored in logs folder)
    :return:
    """
    results = {
        "test_loss_list": test_loss_list,
        "test_accu_list": test_accuracy_list,
        "reg_list": regret_list
    }
    root = os.path.abspath('./MNIST_examples')
    root += '/logs/'
    path2file = root + '{}.pickle'.format(filename)
    with open(path2file, 'wb') as outfile:
        pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(filename):
    if os.path.abspath('.').endswith('MNIST_examples'):
        root = os.path.join(os.path.abspath('.'), 'logs')
    elif os.path.abspath('.').endswith('_Byzantine_attack'):
        root = os.path.join(os.path.abspath('.'), 'MNIST_examples', 'logs')
    elif os.path.abspath('.').endswith('L3GD_V2'):
        root = os.path.join(os.path.abspath('.'), 'MNIST_examples', 'logs')
    print(root)
    
    if '.pickle' in filename:
        with open(os.path.join(root, filename), 'rb') as infile:
            results = pickle.load(infile)
    elif '.log' in filename:
        pass
    else:
        with open(os.path.join(root, filename + '.pickle'), 'rb') as infile:
            results = pickle.load(infile)
    return results["test_loss_list"], results["test_accu_list"], results["reg_list"]
