import logging
import numpy as np
import h5py
import os
import json
from scipy.special import lambertw

# local library
from MNIST_examples.utilities.mnist_opt_algs import grad_desc, normalized_grad_desc
from MNIST_examples.utilities.model_utilities import save_model, save_results


def mnist_load_config(name):
    """
    load config file
    :param name: full path of the config document.
    :return: config dict
    """
    # if 'utilities' in os.getcwd():
    #     root = os.path.abspath('../../MNIST_examples/')
    # else:  # for bash runing in outside.
    root = os.path.abspath('./MNIST_examples')
    root += '/configs/'
    full_path = root + name
    json_cfg_file = open(full_path, 'r')
    cfg = json.load(json_cfg_file)
    json_cfg_file.close()
    return cfg


def load_mnist(filename, hyp):
    """

    :param filename: MNIST_examples is saved in a h5py
    :return:
    """
    TEST_SIZE = hyp['TEST_SIZE']
    mnist_data = h5py.File(filename, 'r')
    x_train = np.float32(mnist_data['x_train'][:])
    y_train = np.int32(np.array(mnist_data['y_train'][:, 0]))
    x_test = np.float32(mnist_data['x_test'][:TEST_SIZE])
    y_test = np.int32(np.array(mnist_data['y_test'][:TEST_SIZE, 0]))
    mnist_data.close()
    return x_train, y_train, x_test, y_test


def softmax(z):
    """implement the softmax functions
    input: numpy ndarray
    output: numpy ndarray # to [0,1]*(10*1)
    """
    exp_list = np.exp(z)
    result = exp_list / sum(exp_list)  # to [0,1]*(10*1)
    return result.reshape((len(z), 1))


def neg_log_loss(pred, label):
    """implement the negative log loss"""
    return -np.log(pred[int(label)])


def mini_batch_gradient(hyp, param, x_batch, y_batch):
    """implement the function to compute the mini batch gradient
    input: param[alg] -- parameters dictionary (w, b)
           x_batch -- a batch of x (size, 784)
           y_batch -- a batch of y (size,)
    output: dw, db, batch_loss
    """
    batch_size = x_batch.shape[0]
    batch_loss = {}
    dw, db = {}, {}
    pred, w_grad, b_grad, w_grad_list, b_grad_list = {}, {}, {}, {}, {}
    for alg in hyp['alg']:
        pred[alg], w_grad_list[alg], b_grad_list[alg], batch_loss[alg] = [], [], [], 0

    for i in range(batch_size):
        x, y = x_batch[i], y_batch[i]
        x = x.reshape((784, 1))  # x: (784,1)
        E = np.zeros((10, 1))  # (10*1) 10 cluster
        E[y][0] = 1  # set ground truth
        for alg in hyp['alg']:
            pred[alg] = softmax(np.matmul(param[alg]['w'], x) + param[alg]['b'])  # (10*1) 10 cluster

            loss = neg_log_loss(pred[alg], y)
            batch_loss[alg] += loss

            w_grad[alg] = E - pred[alg]
            w_grad[alg] = - np.matmul(w_grad[alg], x.reshape((1, 784)))
            w_grad_list[alg].append(w_grad[alg])

            b_grad[alg] = -(E - pred[alg])
            b_grad_list[alg].append(b_grad[alg])

    for alg in hyp['alg']:
        dw[alg] = sum(w_grad_list[alg]) / batch_size
        db[alg] = sum(b_grad_list[alg]) / batch_size
    return dw, db, batch_loss


def evaluate_model(param, x_data, y_data):
    """ implement the evaluation function
    input: param -- parameters dictionary (w, b) for one algorithm
           x_data -- x_train or x_test (size, 784)
           y_data -- y_train or y_test (size,)
    output: loss and accuracy
    """
    # w: (10*784), x: (10000*784), y:(10000,)

    w = param['w'].transpose()
    dist = np.array([np.squeeze(softmax(np.matmul(x_data[i], w))) for i in range(len(y_data))])
    # print(dist.shape)
    result = np.argmax(dist, axis=1)
    accuracy = sum(result == y_data) / float(len(y_data))

    loss_list = [neg_log_loss(dist[i], y_data[i]) for i in range(len(y_data))]
    loss = sum(loss_list)
    return loss, accuracy


# The normal update is the only content requires to change when the update is modified.
def normal_update(hyp, param, dw, db, learning_rate):
    for alg in hyp['alg']:  # hyp[alg] stores the names of algorithms required to run.
        if alg == 'grad_desc':
            param[alg]['w'], param[alg]['b'] = grad_desc(param[alg], dw[alg], db[alg], learning_rate)
            # print(param[alg]['b'])
        elif alg == 'normalized_grad_desc':
            param[alg] = normalized_grad_desc(param[alg], dw[alg], db[alg], learning_rate)
    return param, hyp


def broken_update(hyp, param, dw, db, learning_rate):
    if hyp['attack']['attack_type'] == 'random':
        for alg in hyp['alg']:
            dw[alg] = 2*(np.random.random(dw[alg].shape)-0.5)*hyp['attack']['attack_magnitude']*np.linalg.norm(dw[alg])
            db[alg] = 2*(np.random.random(db[alg].shape)-0.5)*hyp['attack']['attack_magnitude']*np.linalg.norm(db[alg])
    elif hyp['attack']['attack_type'] == 'zero':
        for alg in hyp['alg']:
            dw[alg] = np.zeros_like(dw[alg])
            db[alg] = np.zeros_like(db[alg])
    elif hyp['attack']['attack_type'] == 'flipping':
        for alg in hyp['alg']:  # hyp[alg] stores the names of algorithms required to run.
            dw[alg] = -dw[alg] * hyp['attack']['attack_magnitude']
            db[alg] = -db[alg] * hyp['attack']['attack_magnitude']
    return normal_update(hyp, param, dw, db, learning_rate)


def train(param, hyp, x_train, y_train, x_test, y_test, load_name):
    """
    Byzantine attack with probability hyp['attack_prob']
    :param param:
    :param hyp:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param load_name: log_name, including the attack id and start time.
    :return:f
    """
    num_epoch = hyp['num_epoch']
    batch_size = hyp['batch_size']
    print(os.path.abspath("."))
    train_loss, train_accu, test_loss, test_accu, regret = {}, {}, {}, {}, {}
    test_loss_list, test_accu_list, regret_list = {}, {}, {}
    for alg in hyp['alg']:
        train_loss[alg], train_accu[alg], test_loss[alg], test_accu[alg] = 0, 0, 0, 0
        test_loss_list[alg], test_accu_list[alg], regret_list[alg] = [], [], [0]

    train_num = x_train.shape[0]  # number of image in training set.

    if hyp['num_batch'] == 0:  # number of iterations in each epoch.
        num_batch = int(train_num / batch_size)  # floor integer, default
    else:
        num_batch = hyp['num_batch']

    global_count = 1
    outer_count = 1
    for epoch in range(num_epoch):
        rand_indices = np.random.choice(train_num, train_num, replace=False)  # not put back.

        total_loss = {}
        for alg in hyp['alg']:
            total_loss[alg] = 0

        message = 'the base learning rate: %.10f' % hyp['learning_rate']['base_stp']
        print(message)
        logging.info(load_name)
        
        for batch in range(num_batch):
            outer_count += 1
            # get the function ft for following inner loop.
            index = rand_indices[batch_size * batch:batch_size * (batch + 1)]
            x_batch = x_train[index]
            y_batch = y_train[index]
            index_next = rand_indices[batch_size * (batch + 1):batch_size * (batch + 2)]
            if (batch+1)*batch_size == train_num:
                index_next = rand_indices[0:batch_size]
                x_next_batch = x_train[index_next]
                y_next_batch = y_train[index_next]
            else:
                x_next_batch = x_train[index_next]
                y_next_batch = y_train[index_next]
            # --------------------------------------------------------------------------------------
            # Kt setting
            # --------------------------------------------------------------------------------------
            learning_rate = hyp['learning_rate']['base_stp']
            if hyp['Kt'] == '1':
                Kt = 1
            elif hyp['Kt'] == 'outer_count':
                Kt = outer_count
            elif hyp['Kt'] == 'sqrt_outer_count':
                Kt = int(np.sqrt(outer_count))
            elif hyp['Kt'] == '11_outer_count':
                Kt = int(outer_count**1.1)
            elif hyp['Kt'] == '15_outer_count':
                Kt = int(outer_count**1.5)
            elif hyp['Kt'] == 'alpha':
                alpha = hyp['alpha']
                p = hyp['attack']['attack_prob']
                cosphi = 1/2
                # D = 1  # ignore this term.
                tmp_c = outer_count ** (1 - alpha)
                Kt = int(np.log(tmp_c) / np.log(tmp_c/(tmp_c - 4*((1-p)*cosphi-p)**2)))
            # ####################################################################################
            # ----------------------------- Theoretical iteration --------------------------------
            # ####################################################################################
            elif hyp['Kt'] == 'thm_cst':
                assert not hyp['learning_rate']['diminishing']
                Kt = np.sqrt(outer_count)*np.log(outer_count)
            elif hyp['Kt'] == 'thm_dim':
                assert hyp['learning_rate']['diminishing']
                Kt = np.sqrt(outer_count)
            elif hyp['Kt'] == 'thm_dim_conservative':
                assert hyp['learning_rate']['diminishing']
                Kt = outer_count
            elif hyp['Kt'] == 'thm_dim_lambertw':
                assert hyp['learning_rate']['diminishing']
                Kt_tmp = (-(outer_count**0.5)*lambertw(-(1/(np.exp(1)*(outer_count**0.5))), k=-1)).real
                if Kt_tmp > 1:
                    Kt = int(Kt_tmp)+1
                    print(Kt)
                else:
                    Kt = outer_count
                    print("Kt_tmp <= 1, use Kt=t")

            elif isinstance(hyp['Kt'], int):
                Kt = hyp['Kt']
            else:
                raise Exception("Sorry, Kt is not defined in this way. check check code in broken_train function.")
            # ---------------------------------------------------------
            #   Kt inner loops.
            # ---------------------------------------------------------
            Kt = int(np.max(Kt, 0))
            for k in range(Kt):
                global_count += 1
                # calculate the gradient w.r.t w and b
                dw, db, batch_loss = mini_batch_gradient(hyp, param, x_batch, y_batch)

                for alg in hyp['alg']:
                    total_loss[alg] += batch_loss[alg]
                # learning rate setting
                if hyp['learning_rate']['diminishing']:
                    learning_rate = learning_rate / (k + 1)
                # attack or not
                if np.random.rand() > hyp['attack']['attack_prob']:
                    param, hyp = normal_update(hyp, param, dw, db, learning_rate)
                else:  # attack with probability: hyp['attack']['attack_prob']
                    param, hyp = broken_update(hyp, param, dw, db, learning_rate)
                if global_count % int(1e3) == 0:
                    if hyp['save']:
                        save_model(load_name, param)
                        save_results(test_loss_list, test_accu_list, regret_list, load_name)

            # inner loop end print information every 400 f_t
            if batch+1 % 40 == 0:
                for alg in hyp['alg']:
                    message = 'Epoch %d, Batch %d, Loss %.2f' % (epoch + 1, batch, batch_loss[alg])
                    print(message)

                batch_loss[alg] = 0
            # calculate the regret after every batch is given (ft) -- batch
            for alg in hyp['alg']:
                regret[alg], _ = evaluate_model(param[alg], x_next_batch, y_next_batch)
                regret_list[alg].append(regret_list[alg][-1]+regret[alg])

            # if batch % 20 == 0:
            # for alg in hyp['alg']:
                train_loss[alg], train_accu[alg] = evaluate_model(param[alg], x_train, y_train)
                test_loss[alg], test_accu[alg] = evaluate_model(param[alg], x_test, y_test)

                test_loss_list[alg].append(test_loss[alg])
                test_accu_list[alg].append(test_accu[alg])

                message = 'Alg: %s. Regret %f \n Epoch %d, Batch: %d, Train Loss %.2f, Train Accu %.4f, ' \
                          'Test Loss %.2f, Test Accu %.4f' % \
                          (alg, regret[alg], epoch + 1, batch+1,
                           train_loss[alg], train_accu[alg], test_loss[alg], test_accu[alg])
                print(message)
                logging.info(message)
    return test_loss_list, test_accu_list, regret_list



