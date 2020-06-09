import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/citeseer'
opt['hidden_dim'] = 16
opt['input_dropout'] = 0.5
opt['dropout'] = 0
opt['optimizer'] = 'adam'
opt['lr'] = 0.01
opt['decay'] = 5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 2000
opt['epoch'] = 100
opt['iter'] = 1
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['tau'] = 0.0
opt['save'] = 'exp_citeseer'
opt['mixup_alpha'] = 1.0
opt['partition_num'] = 40
opt['task_ratio'] = 0.7


### ict hyperparameters ###
opt['ema_decay'] = 0.999
opt['consistency_type'] = "mse"
opt['consistency_rampup_starts'] = 500
opt['consistency_rampup_ends'] = 1000
opt['mixup_consistency'] =10.0

def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

for k in range(50):
    seed = k + 1
    #print(opt['mixup_alpha'])
    #print(opt['mixup_consistency'])
    opt['seed'] = seed
    run(opt)
