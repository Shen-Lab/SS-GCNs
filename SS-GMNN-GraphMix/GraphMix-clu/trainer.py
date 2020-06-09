import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer

bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
class_criterion = nn.CrossEntropyLoss().cuda()
def mixup_criterion(y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr



class Trainer(object):
    def __init__(self, opt, model, partition_labels, ema= True):

        partition_num = partition_labels.max() + 1
        self.partition_labels = partition_labels.cuda()
        self.task_ratio = opt['task_ratio']
        self.loss_func = nn.CrossEntropyLoss()

        self.opt = opt
        self.ema = ema
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        self.ss_classifier = nn.Linear(opt['hidden_dim'], partition_num, bias=False)

        if opt['cuda']:
            self.criterion.cuda()
            self.ss_classifier.cuda()

        self.parameters.append(self.ss_classifier.weight)

        if  self.ema == True:
            self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        self.model.reset()
        if self.ema == True:
            self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def update(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft(self, inputs, target, idx, idx_u):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

       
        logits= self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
      
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))
 
        logits0 = self.model.forward_partition(inputs)
        logits0 = self.ss_classifier(logits0)
        loss0 = self.loss_func(logits0[idx_u], self.partition_labels[idx_u])
        
        return loss, loss0
    
    
   
    
    
    def update_soft_aux(self, inputs, target,target_discrete, idx, idx_unlabeled, adj, opt, mixup_layer, idx_u):
        """uses the auxiliary loss as well, which does not use the adjacency information"""
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()
            idx_unlabeled = idx_unlabeled.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        
        mixup = True
        if mixup == True:
            # get the supervised mixup loss #
            logits, target_a, target_b, lam = self.model.forward_aux(inputs, target=target, train_idx= idx, mixup_input=False, mixup_hidden = True, mixup_alpha = opt['mixup_alpha'],layer_mix=mixup_layer)

            logits0 = self.model.forward_partition(inputs)
            logits0 = self.ss_classifier(logits0)
            loss0 = self.loss_func(logits0[idx_u], self.partition_labels[idx_u])

            mixed_target = lam*target_a + (1-lam)*target_b
            loss = bce_loss(softmax(logits[idx]), mixed_target)

            # get the unsupervised mixup loss #
            logits, target_a, target_b, lam = self.model.forward_aux(inputs, target=target, train_idx= idx_unlabeled, mixup_input=False, mixup_hidden = True, mixup_alpha = opt['mixup_alpha'],layer_mix= mixup_layer)
            mixed_target = lam*target_a + (1-lam)*target_b
            loss_usup = bce_loss(softmax(logits[idx_unlabeled]), mixed_target)
        else:
            logits = self.model.forward_aux(inputs, target=None, train_idx= idx, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))

            '''
            logits0 = self.model.forward_partition(inputs)
            logits0 = self.ss_classifier(logits0)
            loss0 = self.loss_func(logits0, self.partition_labels)            
            '''

            logits = self.model.forward_aux(inputs, target=None, train_idx= idx_unlabeled, mixup_input= False, mixup_hidden = False, mixup_alpha = 0.0,layer_mix=None)
            logits = torch.log_softmax(logits, dim=-1)
            loss_usup = -torch.mean(torch.sum(target[idx_unlabeled] * logits[idx_unlabeled], dim=-1))
        
        return loss, loss_usup, loss0

    
    
    def evaluate(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()
        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return loss.item(), preds, accuracy.item()

    def predict(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits


    def predict_aux(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model.forward_aux(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    def predict_noisy(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        #self.model.eval()

        logits = self.model(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits


    def predict_noisy_aux(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        #self.model.eval()

        logits = self.model.forward_aux(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits
    
    
    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
