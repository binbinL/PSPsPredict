import numpy as np


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class lr_AIAYN():
    '''
    Learning rate scheduler from the paper:
    Attention is All You Need
    '''
    def __init__(self,optimizer,d_model,warmup_steps=4000,factor=1):
        self.optimizer=optimizer
        self.d_model=d_model
        self.warmup_steps=warmup_steps
        self.step_num=0
        self.factor=factor

    def step(self):
        self.step_num+=1
        lr=self.d_model**-0.5*np.min([self.step_num**-0.5,
                                      self.step_num*self.warmup_steps**-1.5])*self.factor
        update_lr(self.optimizer,lr)
        return lr
        
        
class Cos_Anneal():
    '''
    Learning rate scheduler flat and anneal
    '''
    def __init__(self,optimizer,max_lr,min_lr,T):
        self.optimizer=optimizer
        self.max_lr=max_lr
        self.min_lr=min_lr
        self.step_num=0
        self.T=T

    def step(self):
        pi=3.1415
        self.step_num+=1
        lr=self.min_lr+0.5*(self.max_lr-self.min_lr)*(1+np.cos(self.step_num/self.T*pi))
        if self.optimizer:
            update_lr(self.optimizer,lr)
        return lr        
    
class WarmupLR():
    '''
    Learning rate scheduler WarmupLR
    '''
    def __init__(self, optimizer, num_warm) -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = [group['lr'] for group in self.optimizer.param_groups]
        self.num_step = 0
 
    def __compute(self, lr) -> float:
        return lr * min(self.num_step ** (-0.6), self.num_step * self.num_warm ** (-1.5))
 
    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]
            
class ExponentialLR():
  def __init__(self,optimizer,gamma=0.1,last_epoch=-1): 
    super(ExponentialLR,self).__init__(optimizer,step_size=0,gamma=gamma,last_epoch=last_epoch)

  def get_lr(self):
    if self.last_epoch == 0:
      return [group['lr'] for group in self.optimizer.param_groups]
    return [group['lr'] * self.gamma 
            for group in self.optimizer.param_groups]