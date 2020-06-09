import sys
import os
import copy
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trainer import Trainer
from gnn import GNN
from ramps import *
from losses import *
import loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='exp', help = 'name of the folder where the results are saved')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--mixup_alpha', type=float, default=1.0, help='alpha for mixing')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--use_gold', type=int, default=1, help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
parser.add_argument('--draw', type=str, default='max', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')
parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                    choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency_rampup_starts', default=30, type=int, metavar='EPOCHS',
                    help='epoch at which consistency loss ramp-up starts')
parser.add_argument('--consistency_rampup_ends', default=30, type=int, metavar='EPOCHS',
                    help='lepoch at which consistency loss ramp-up ends')
parser.add_argument('--mixup_consistency', default=1.0, type=float,
                    help='max consistency coeff for mixup usup loss')

parser.add_argument('--partition_num', type=int, default=12, help='partition number')
parser.add_argument('--task_ratio', type=float, default=0.3, help='task ratio')

args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)


    
net_file = opt['dataset'] + '/net.txt'
label_file = opt['dataset'] + '/label.txt'
feature_file = opt['dataset'] + '/feature.txt'
train_file = opt['dataset'] + '/train.txt'
dev_file = opt['dataset'] + '/dev.txt'
test_file = opt['dataset'] + '/test.txt'


vocab_node = loader.Vocab(net_file, [0, 1])
vocab_label = loader.Vocab(label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

opt['num_node'] = len(vocab_node)
opt['num_feature'] = len(vocab_feature)
opt['num_class'] = len(vocab_label)


graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])

partition_num = opt['partition_num']
partition_labels = graph.partition(partition_num)

graph.to_symmetric(opt['self_link_weight'])
feature.to_one_hot(binary=True)
adj = graph.get_sparse_adjacency(opt['cuda'])

with open(train_file, 'r') as fi:
    idx_train = [vocab_node.stoi[line.strip()] for line in fi]
with open(dev_file, 'r') as fi:
    idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
with open(test_file, 'r') as fi:
    idx_test = [vocab_node.stoi[line.strip()] for line in fi]
idx_all = list(range(opt['num_node']))

idx_unlabeled = list(set(idx_all)-set(idx_train))
inputs = torch.Tensor(feature.one_hot)
target = torch.LongTensor(label.itol)
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)
idx_unlabeled = torch.LongTensor(idx_unlabeled)
inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
target_q = torch.zeros(opt['num_node'], opt['num_class'])
inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
target_p = torch.zeros(opt['num_node'], opt['num_class'])

if opt['cuda']:
    inputs = inputs.cuda()
    target = target.cuda()
    idx_train = idx_train.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    idx_unlabeled = idx_unlabeled.cuda()
    inputs_q = inputs_q.cuda()
    target_q = target_q.cuda()
    inputs_p = inputs_p.cuda()
    target_p = target_p.cuda()

gnn = GNN(opt, adj)
trainer = Trainer(opt, gnn, partition_labels)

# Build the ema model
gnn_ema = GNN(opt, adj)

for ema_param, param in zip(gnn_ema.parameters(), gnn.parameters()):
            ema_param.data= param.data

for param in gnn_ema.parameters():
            param.detach_()
trainer_ema = Trainer(opt, gnn_ema, partition_labels, ema = False)


def init_data():
    inputs_q.copy_(inputs)
    temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
    temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
    target_q[idx_train] = temp


def update_ema_variables(model, ema_model, alpha, epoch):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (epoch + 1), alpha)
    #print (alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(final_consistency_weight, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - args.consistency_rampup_starts
    return final_consistency_weight *sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts )



def sharpen(prob, temperature):
    temp_reciprocal = 1.0/ temperature
    prob = torch.pow(prob, temp_reciprocal)
    row_sum = prob.sum(dim=1).reshape(-1,1)
    out = prob/row_sum
    return out 




def train(epoches):
    best = 0.0
    init_data()
    results = []
    
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss 
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss 
    
    for epoch in range(epoches):
        rand_index = random.randint(0,1)
        if rand_index == 0:
            trainer.model.train()
            trainer.optimizer.zero_grad()
            k = 10
            temp  = torch.zeros([k, target_q.shape[0], target_q.shape[1]], dtype=target_q.dtype)
            temp = temp.cuda()
            for i in range(k):
                temp[i,:,:] = trainer.predict_noisy(inputs_q)
            target_predict = temp.mean(dim = 0)
            
            target_predict = sharpen(target_predict,0.1)
            target_q[idx_unlabeled] = target_predict[idx_unlabeled]
            temp = torch.randint(0, idx_unlabeled.shape[0], size=(idx_train.shape[0],))
            idx_unlabeled_subset = idx_unlabeled[temp]
            loss , loss_usup, loss0 = trainer.update_soft_aux(inputs_q, target_q, target, idx_train, idx_unlabeled_subset, adj,  opt, mixup_layer =[1])
            mixup_consistency = get_current_consistency_weight(opt['mixup_consistency'], epoch)
            total_loss = loss + mixup_consistency*loss_usup

            total_loss = total_loss * opt['task_ratio'] + loss0 * (1 - opt['task_ratio'])

            total_loss.backward()
            trainer.optimizer.step()

        else:
            trainer.model.train()
            trainer.optimizer.zero_grad()
            loss, loss0 = trainer.update_soft(inputs_q, target_q, idx_train)
            
            total_loss = loss * opt['task_ratio'] + loss0 * (1 - opt['task_ratio'])

            total_loss.backward()
            trainer.optimizer.step()
        _, preds, accuracy_train = trainer.evaluate(inputs_q, target, idx_train)
        _, preds, accuracy_dev = trainer.evaluate(inputs_q, target, idx_dev)
        _, preds, accuracy_test = trainer.evaluate(inputs_q, target, idx_test)
        _, preds, accuracy_test_ema = trainer_ema.evaluate(inputs_q, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
        
        if epoch%400 == 0:
            if rand_index == 0:
                print ('epoch :{:4d},loss:{:.10f},loss_usup:{:.10f}, train_acc:{:.3f}, dev_acc:{:.3f}, test_acc:{:.3f}'.format(epoch, loss.item(),loss_usup.item(), accuracy_train, accuracy_dev, accuracy_test))
            else : 
                 print ('epoch :{:4d},loss:{:.10f}, train_acc:{:.3f}, dev_acc:{:.3f}, test_acc:{:.3f}'.format(epoch, loss.item(), accuracy_train, accuracy_dev, accuracy_test))
        
        if accuracy_dev > best:
            best = accuracy_dev
            state = dict([('model', copy.deepcopy(trainer.model.state_dict())), ('optim', copy.deepcopy(trainer.optimizer.state_dict()))])
    
        update_ema_variables(gnn, gnn_ema, opt['ema_decay'], epoch)
    
        
    return results


base_results = []
base_results += train(opt['pre_epoch'])


def get_accuracy(results):
    best_dev, acc_test = 0.0, 0.0
    for d, t in results:
        if d >= best_dev:
            best_dev, acc_test = d, t
    return best_dev, acc_test

best_dev, acc_test = get_accuracy(base_results)

print('Test acc{:.3f}'.format(acc_test * 100))
with open('record.txt', 'a') as f:
    # f.write(str(best_dev) + ',')
    f.write(str(acc_test) + ',')
