# -*- coding: utf-8 -*-


'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import json

import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
from scipy.special import lambertw
import pickle

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except ValueError:
    term_width = 90  # Fallback value if not running in a terminal

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def cifar10_load_config(name):
    """
    load config file
    :param name: full path of the config document.
    :return: config dict
    """
    # root = os.path.abspath('./Cifar10_examples')
    root = os.path.abspath('./')
    root += '/configs/'
    full_path = root + name
    json_cfg_file = open(full_path, 'r')
    cfg = json.load(json_cfg_file)
    json_cfg_file.close()
    return cfg
        
        
def get_inner_Kt(hyp, global_count_t):
    Kt_type = hyp['Kt']['Kt_type']
    outer_count = global_count_t
    if Kt_type == '1':
        Kt = 1
    elif Kt_type == 'outer_count':
        Kt = outer_count
    elif Kt_type == 'sqrt_outer_count':
        Kt = int(np.sqrt(outer_count))
    elif Kt_type == '11_outer_count':
        Kt = int(outer_count**1.1)
    elif Kt_type == '15_outer_count':
        Kt = int(outer_count**1.5)
    elif Kt_type == 'alpha':
        alpha = hyp['alpha']
        p = hyp['attack']['attack_prob']
        cosphi = 1/2
        # D = 1  # ignore this term.
        tmp_c = outer_count ** (1 - alpha)
        Kt = int(np.log(tmp_c) / np.log(tmp_c/(tmp_c - 4*((1-p)*cosphi-p)**2)))
    # ####################################################################################
    # ----------------------------- Theoretical iteration --------------------------------
    # ####################################################################################
    elif Kt_type == 'thm_cst':
        assert not hyp['learning_rate']['diminishing']
        Kt = np.sqrt(outer_count)*np.log(outer_count)
    elif Kt_type == 'thm_dim':
        assert hyp['learning_rate']['diminishing']
        Kt = np.sqrt(outer_count)
    elif Kt_type == 'thm_dim_conservative':
        assert hyp['learning_rate']['diminishing']
        Kt = outer_count
    elif Kt_type == 'thm_dim_lambertw':
        assert hyp['learning_rate']['diminishing']
        Kt_tmp = (-(outer_count**0.5)*lambertw(-(1/(np.exp(1)*(outer_count**0.5))), k=-1)).real
        if Kt_tmp > 1:
            Kt = int(Kt_tmp)+1
            print(Kt)
        else:
            Kt = outer_count
            print("Kt_tmp <= 1, use Kt=t")

    # elif isinstance(Kt_type, int):
    #     Kt = Kt_type
    elif Kt_type == 'constant':
        Kt = hyp['Kt']['Kt_cst']
    else:
        raise Exception("Sorry, Kt is not defined in this way.")    
    return Kt

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
    root = os.path.abspath('.')
    root += '/logs/'
    path2file = root + '{}.pickle'.format(filename)
    with open(path2file, 'wb') as outfile:
        pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(filename):
    if os.path.abspath('.').endswith('Cifar10_examples'):
        root = os.path.abspath('.') + '/logs/'
    elif os.path.abspath('.').endswith('_Byzantine_attack'):
        root = os.path.abspath('./Cifar10_examples/')
        root += 'logs/'
    if '.pickle' in filename:
        with open(root + '{}'.format(filename), 'rb') as infile:
            results = pickle.load(infile)
    elif '.log' in filename:
        pass
    else:
        with open(root + '{}.pickle'.format(filename), 'rb') as infile:
            results = pickle.load(infile)
    return results["test_loss_list"], results["test_accu_list"], results["reg_list"]




def ngd(model, optimizer):
    """
    Normalized gradient descent
    """
    with torch.no_grad():
        for param in model.parameters():
            # Calculate the norm of the gradient
            norm = param.grad.data.norm(2)

            # Normalize the gradient
            param.grad.data.div_(norm)

            # Update parameters using normalized gradients
            param.data -= optimizer.param_groups[0]['lr'] * param.grad.data
    
def attack_grad(hyp, model):
# Iterate over all parameters (i.e. the weights) in your model
    if hyp['attack']['attack_type'] == 'random':
        mag = hyp['attack']['attack_magnitude']
        for param in model.parameters():
            if param.grad is not None:
                # Replace the gradient with random numbers
                param.grad = torch.randn_like(param.grad) * mag
    elif hyp['attack']['attack_type'] == 'zero':
        for param in model.parameters():
            if param.grad is not None:
                # Replace the gradient with zeros
                param.grad.zero_()
    elif hyp['attack']['attack_type'] == 'flipping': 
        mag = hyp['attack']['attack_magnitude']       
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(-mag)  # Multiply the gradient by - mag in-place
                
                
def tune_lr(hyp, optimizer, k):
    learning_rate = hyp['learning_rate']['base_stp']
    if hyp['learning_rate']['diminishing']:
        learning_rate = learning_rate / (k + 1)
    for g in optimizer.param_groups:
        g['lr'] = learning_rate
        

def save_results(train_loss_t, train_accu_t, test_loss_epoch, test_accu_epoch, filename):
    """
    """
    results = {
        "train_loss_t": train_loss_t,
        "train_accu_t": train_accu_t, 
        "test_loss_epoch": test_loss_epoch, 
        "test_accu_epoch": test_accu_epoch
    }
    # root = os.path.abspath('./Cifar10_examples')
    root = os.path.abspath('.')
    root += '/logs/'
    path2file = root + '{}.pickle'.format(filename)
    with open(path2file, 'wb') as outfile:
        pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(filename):
    if os.path.abspath('.').endswith('Cifar10_examples'):
        root = os.path.abspath('.') + '/logs/'
    elif os.path.abspath('.').endswith('_Byzantine_attack'):
        root = os.path.abspath('./Cifar10_examples/')
        root += 'logs/'
    if '.pickle' in filename:
        with open(root + '{}'.format(filename), 'rb') as infile:
            results = pickle.load(infile)
    elif '.log' in filename:
        pass
    else:
        with open(root + '{}.pickle'.format(filename), 'rb') as infile:
            results = pickle.load(infile)
    # TODO: the value returned should be modified.
    # return results["test_loss_list"], results["test_accu_list"], results["reg_list"]
    return results["train_loss_t"], results["train_accu_t"], results["test_loss_epoch"], results["test_accu_epoch"]