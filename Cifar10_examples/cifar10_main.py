# -*- coding: utf-8 -*-

'''
Train CIFAR10 with PyTorch.
Find the original file in https://github.com/kuangliu/pytorch-cifar
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np

import logging
import os
from datetime import datetime
import sys
import argparse
import random

from models import *
from utils import progress_bar, cifar10_load_config, get_inner_Kt, ngd, attack_grad, tune_lr, save_results

# get arguments from command line
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("-c", "--config", type=str,
                        default="attack00016", help="Config of hyper-parameters, prefixed with config_mnist_") 
parser.add_argument('-r', "--rand_seed", type=int,
                    default=1024, help="random seed for np.random.seed")  # unused
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')  
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')  #original
parser.add_argument('--resume', # action='store_true',
                    default=False, help='resume from checkpoint')  # changed to default not resume the model.
# The ckpt_name can be something like
# './checkpoint/{log_name}_{epoch}_netpara.pth'.format(log_name=log_name, epoch=epoch)'
# or: 
parser.add_argument('--ckpt_name', default=None, help='name of checkpoint file end with pth that contain the net.')
# Add the arguments
parser.add_argument('--alg', default='ONGD', choices=['L3GD', 'ONGD', 'OMGD'], help='Algorithm choice')
parser.add_argument('--Kt_cst', default=2, type=int, help='constant for inner Kt')
args = parser.parse_args() 
print('Current settings')
print('Current path.{}'.format(os.path.abspath('.')))
print(args)
rand_seed = args.rand_seed
random.seed(rand_seed)
np.random.seed(rand_seed)

# After getting the arguments, we comes to the main function.


cfg_idx = args.config
cfg_name = 'config_cifar10_{}.json'.format(cfg_idx)
hyperpara = cifar10_load_config(cfg_name)
# get the values from the configuration file.
# algs = hyperpara['alg']
# # Check if `algs` is a list
# assert isinstance(algs, list), "`algs` should be of type list."
# # Check if `algs` has exactly one element
# assert len(algs) == 1, "`algs` should contain exactly one element."
# # algs = ['ONGD'], ['ONGD'], or ['L3GD']
# alg = algs[0]

# Add the following content for logging file.
# root_path = ""
# os.chdir(root_path)
start_time = datetime.now()
alg = args.alg
if args.Kt_cst == None:
    log_name = start_time.strftime(args.config + '_r' + str(rand_seed) + '_%Y%m%d_%H%M') + alg 
else:
    log_name = start_time.strftime(args.config + '_r' + str(rand_seed) + '_Kt' + str(args.Kt_cst) + '_%Y%m%d_%H%M') + alg 
log_name_result = log_name + '_results'
logging.basicConfig(filename="./logs/{}.log".format(log_name),
                        filemode="w", format="%(message)s", level=logging.INFO)
fh = logging.FileHandler(filename="./logs/{}.log".format(log_name))
logging.info(args)



num_epoch = hyperpara["num_epoch"]
batch_size = hyperpara['batch_size']
# Kt_type = hyperpara['Kt']['Kt_type']
# if Kt_type == "constant":
#     Kt_cst = hyperpara['Kt']['constant']
base_lr = hyperpara['learning_rate']['base_stp']
diminishing_lr = hyperpara['learning_rate']['diminishing']

attack_prob = hyperpara['attack']['attack_prob']
# attack_type = hyperpara['attack']['attack_type']
# attack_magnitude = hyperpara['attack']['attack_magnitude']

save = hyperpara['save']
load = hyperpara['load']
############### END the paragram loading ################


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
logging.info(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)  # original
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)  # unchanged, set 100. This would almost doesn't change the speed.

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
logging.info('==> Building model..')
# The training and testing time is recorded when the batch_size=128. I have change this to 256. The training time slightly changed.
# net = VGG('VGG19')  # 22/10 ms/step for training/testing
# net = ResNet18()  # 17/7 ms/step for training/testing
# net = PreActResNet18()  # 17/7 ms/step for training/testing
# net = GoogLeNet()  # 42/13 ms/step for training/testing
# net = DenseNet121()  # 45/15 ms/step for training/testing
# net = ResNeXt29_2x64d()  # 35/10 ms/step for training/testing
net = MobileNet()  # 8/3 ms/step for training/testing  <-------- FASTEST ------
# net = MobileNetV2() # 16/6 ms/step for training/testing
# net = DPN92()  # 107/27 ms/step for training/testing
# net = ShuffleNetG2()  # ERROR
# net = SENet18()  # 20/8 ms/step for training/testing
# net = ShuffleNetV2(1)  # 14/6 ms/step for training/testing
# net = EfficientNetB0()  # 18/7 ms/step for training/testing
# net = RegNetX_200MF()  # 16/6 ms/step for training/testing
# net = SimpleDLA()  # 26/10 ms/step for training/testing
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    load_checkpoint_name = args.ckpt_name
    checkpoint = torch.load('./checkpoint/{}.pth'.format(load_checkpoint_name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()  # original
# criterion = nn.CrossEntropyLoss(reduction='sum')  # the training acc becomes 100% rapidly. weird.
# criterion = nn.CrossEntropyLoss(reduction='mean')  # same as the original (that is the default)

# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), lr=base_lr,
                      momentum=0, weight_decay=5e-4) # without using momentum information.
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



# Training
def train(epoch, alg):
    print('\nEpoch: {}. Alg: {}'.format(epoch, alg))
    net.train()  # This line puts the model into training mode. This is necessary because some layers, like dropout and batch normalization, behave differently during training and evaluation.
    train_loss = 0
    correct = 0
    total = 0
    global_count_t = epoch * (len(trainloader)) + 1  # initialize for each epoch.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.Kt_cst is None:
        # do something if Kt_cst is not provided, compatibility to previous config files.
            Kt = get_inner_Kt(hyp=hyperpara, global_count_t=global_count_t)
        else:
            assert hyperpara['Kt']['Kt_type'] == 'constant'
            Kt = args.Kt_cst
            
        # print(Kt)
        Kt = 1 if Kt<1 else int(Kt)       
        if alg == 'ONGD': 
            Kt = 1  # for online normalized gradient descent, we only need one update.
            random_number_for_attack = random.uniform(0, 1)
            # print(random_number_for_attack)
            # print(global_count_t)
            attack_succ_flag = (random_number_for_attack < attack_prob) # with attack prob, attack.
            
            optimizer.zero_grad()  # resets the gradients in the optimizer before calculating the new gradients for the current batch.
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if attack_succ_flag: attack_grad(hyperpara, net)
            ngd(net, optimizer)
            tune_lr(hyperpara, optimizer, 1)            
        elif alg == 'OMGD':
            for k in range(Kt):
                # print(Kt)
                attack_succ_flag = (random.uniform(0, 1) < attack_prob) # with attack prob, attack.
                optimizer.zero_grad()  # resets the gradients in the optimizer before calculating the new gradients for the current batch.
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if attack_succ_flag: attack_grad(hyperpara, net)
                optimizer.step()
                tune_lr(hyperpara, optimizer, k)
                train_loss
        elif alg == 'L3GD':
            for k in range(Kt):
                attack_succ_flag = (random.uniform(0, 1) < attack_prob) # with attack prob, attack.
                optimizer.zero_grad()  # resets the gradients in the optimizer before calculating the new gradients for the current batch.
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if attack_succ_flag: attack_grad(hyperpara, net)
                ngd(net, optimizer)
                # turn the stepsize (learning rate)
                tune_lr(hyperpara, optimizer, k)
                # for g in optimizer.param_groups:
                #     g['lr'] = 0.01
        
        global_count_t += 1
        # TODO: calculate the regret and accu etc.
        if global_count_t%10==0:
            save_results(train_loss_t, train_accu_t, test_loss_epoch, test_accu_epoch, log_name_result)
        
        # print(global_count_t)
        train_loss += loss.item()
        # print(loss.item()*batch_size)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train ls: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        train_loss_t.append(train_loss)  # record for each t.
        train_accu_t.append(100.*correct/total)

# Testing
def test(epoch, alg):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test ls: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
        # print('Saving checkpoint..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        #     'alg': alg
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/{log_name}_{alg}_best_ckpt.pth'.format(log_name=log_name, alg=alg))
        # best_acc = acc
    
    logging.info('Epoch: {}. Alg: {}: Test Loss: {:.3f}. Test Acc: {:.3f}. End at time: {}'.format(epoch, alg, test_loss, acc, datetime.now().strftime(args.config + '_%Y%m%d_%H%M.log')))
    
    # Save every 50 epoches
    if (epoch+2)%100==0:
        print('Saving at epoch {}'.format(epoch))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'alg': alg
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{log_name}_{epoch}_netpara.pth'.format(log_name=log_name, epoch=epoch))
    test_loss_epoch.append(test_loss) 
    test_accu_epoch.append(acc)
        

train_loss_t, train_accu_t, test_loss_epoch, test_accu_epoch= [], [], [], []

# every algorithm should start with the same model. But if algs contains multiple alg, then the next alg will start with a trained model. Unfair.
# for alg in algs:
nan_counter = 0
nan_threshold = 777
for epoch in range(start_epoch, start_epoch+num_epoch):    
    train(epoch, alg)
    if len(train_loss_t) >= nan_threshold and all(np.isnan(loss) for loss in train_loss_t[-nan_threshold:]):
        print(f"Loss is NaN for {nan_threshold} continuous steps. Stopping training...")
        break
    test(epoch, alg)
    # logging.info('Epoch: {}. End at time: {}'.format(epoch, datetime.now().strftime(args.config + '_%Y%m%d_%H%M.log')))

logging.getLogger().removeHandler(fh)

torch.cuda.empty_cache() 
"""
The torch.cuda.empty_cache() method releases all unused memory in PyTorch's CUDA memory cache to the CUDA free memory pool, and it can then be used by other CUDA applications. However, it does not interfere with the memory that is currently being used by other applications or processes.

So, if you have another process or script that is also using the GPU, calling torch.cuda.empty_cache() in one script will not affect the GPU memory being used by the other script. Each process has its own separate CUDA context and manages its own memory.

However, it's important to remember that overall GPU memory is shared among all processes. If one process uses a significant amount of GPU memory, less memory will be available for other processes. If you're running multiple GPU-intensive processes simultaneously, you'll need to ensure that you're managing memory appropriately in each one to avoid running out of GPU memory.
"""