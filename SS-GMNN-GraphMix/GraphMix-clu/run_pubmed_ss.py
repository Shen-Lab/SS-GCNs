import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/pubmed'
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
opt['save'] = 'exp_pubmed'
opt['mixup_alpha'] =1.0
opt['partition_num'] = 0
opt['task_ratio'] = 0


### ict hyperparameters ###
opt['ema_decay'] = 0.999
opt['consistency_type'] = "mse"
opt['consistency_rampup_starts'] = 500
opt['consistency_rampup_ends'] = 1000
opt['mixup_consistency'] = 1.0



def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))


os.system('rm record.txt')
os.system('echo -n -> record.txt')
os.system('rm record_val.txt')
os.system('echo -n -> record_val.txt')

task_ratio_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for t in task_ratio_list:
    os.system('rm record.txt')
    os.system('echo -n -> record.txt')

    # opt['partition_num'] = p
    opt['task_ratio'] = t
    for k in range(10):
        seed = k + 1
        opt['seed'] = seed
        run(opt)
    os.system('python result_cal.py')

    with open('record_val.txt', 'a') as f:
        f.write(str(t) + '\n')


