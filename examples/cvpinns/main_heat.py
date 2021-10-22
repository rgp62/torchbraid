#@HEADER
# ************************************************************************
# 
#                        Torchbraid v. 0.1
# 
# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC 
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
# Government retains certain rights in this software.
# 
# Torchbraid is licensed under 3-clause BSD terms of use:
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name National Technology & Engineering Solutions of Sandia, 
# LLC nor the names of the contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission.
# 
# Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
# 
# ************************************************************************
#@HEADER

from __future__ import print_function
import sys
import argparse
import torch
import torchbraid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics as stats

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from timeit import default_timer as timer

from mpi4py import MPI

import scipy
import scipy.special
import numpy as np
import matplotlib.pylab as plt

def root_print(rank,s):
  if rank==0:
    print(s)

class Grid:
    def __init__(self,L,nx):
        pts0=1
        pts1=1
        xi0,wi0 = scipy.special.roots_legendre(pts0)
        xi1,wi1 = scipy.special.roots_legendre(pts1)
        wi01 = np.outer(wi1,wi0)
        xi01 = np.transpose(np.meshgrid(xi1,xi0,indexing='ij'),(1,2,0))
        quad = {'0':(xi0,wi0),'1':(xi1,wi1),'01':(xi01,wi01)}
        
        self.L = L
        self.nx = nx
        self.quad = quad


        xi0  = torch.tensor(quad['0'] [0],dtype=torch.float32)
        xi1  = torch.tensor(quad['1'] [0],dtype=torch.float32)
        xi01 = torch.tensor(quad['01'][0],dtype=torch.float32)
        self.wi0  = torch.tensor(quad['0'] [1],dtype=torch.float32)
        self.wi1  = torch.tensor(quad['1'] [1],dtype=torch.float32)
        self.wi01 = torch.tensor(quad['01'][1],dtype=torch.float32)


        x0 = np.linspace(L[0][0],L[0][1],nx[0]+1)
        x1 = np.linspace(L[1][0],L[1][1],nx[1]+1)

        x0c = (x0[0:-1]+x0[1:])/2
        x1c = (x1[0:-1]+x1[1:])/2
        x01c = np.transpose(np.meshgrid(x0c,x1c,indexing='ij'),(1,2,0))


        dx = x01c[1,1] - x01c[0,0]
        self.dx = dx
        
        self.x0  = torch.tensor(np.expand_dims(np.transpose(np.meshgrid(x0,x1c,indexing='ij'),(1,2,0)),2) \
                     + np.stack([np.zeros(len(xi0)),dx[1]/2*xi0],1),dtype=torch.float32)
        self.x1  = torch.tensor(np.expand_dims(np.transpose(np.meshgrid(x0c,x1,indexing='ij'),(1,2,0)),1) \
                     + np.expand_dims(np.stack([dx[0]/2*xi1,np.zeros(len(xi1))],1),1),dtype=torch.float32)
        self.x01 = torch.tensor(np.expand_dims(np.expand_dims(x01c,1),4) + np.expand_dims(xi01*dx/2,1),dtype=torch.float32)

        self.xall = torch.cat([torch.reshape(self.x0,(-1,2)),torch.reshape(self.x1,(-1,2))],0)
        self.x0len = torch.prod(torch.tensor(self.x0.shape[0:-1]))
        self.x1len = torch.prod(torch.tensor(self.x1.shape[0:-1]))
        

class Heat:
    def __init__(self,grid):
        self.grid = grid
        self.F1BC_L = torch.zeros(self.grid.x1[:,:,0:1,...,0].shape + (1,),dtype=torch.float32)
        self.F1BC_R = torch.zeros(self.grid.x1[:,:,0:1,...,0].shape + (1,),dtype=torch.float32)
        x = self.grid.x0[0:1,...,1::].numpy()
        self.F0BC = torch.tensor(np.piecewise(x, [x < .4,(x>=.4) & (x<.5), (x>=.5) & (x<.6), x >= .6], [0., lambda x:10*(x-.4), lambda x: -10*(x-.6),0]),dtype=torch.float32)

    def loss(self,u):

        ux_Jux = u(self.grid.xall)
        pivot = ux_Jux.shape[0]//3
        ux = ux_Jux[0:pivot]
        Jux = torch.reshape(ux_Jux[pivot::],(ux.shape[0],1,2))
        dudx = Jux[:,:,1]
        ux0 = torch.reshape(ux[0:self.grid.x0len],(self.grid.x0.shape[0:-1]+(1,)))
        ux1 = torch.reshape(dudx[self.grid.x0len::],(self.grid.x1.shape[0:-1]+(1,)))
        F0ux = torch.cat([self.F0BC,ux0[1::]],axis=0)
        F1ux = torch.cat([self.F1BC_L,ux1[:,:,1:-1],self.F1BC_R],axis=2)
        
        F0int  = self.grid.dx[1]/2*torch.einsum('abix,i->abx',F0ux,self.grid.wi0)
        F1int  = self.grid.dx[0]/2*torch.einsum('aibx,i->abx',F1ux,self.grid.wi1)
        
        return torch.sum((
                (F0int[1::] - F0int[0:-1])  \
               -(F1int[:,1::] - F1int[:,0:-1])
               )**2)


class OpenLayer(nn.Module):
  def __init__(self,channels):
    super(OpenLayer, self).__init__()
    self.layer = nn.Linear(2,channels,dtype=torch.float32)
    self.channels = channels

  def forward(self, x):
    Ax = self.layer(x)
    hAx = torch.tanh(Ax)
    hpAx = 1-hAx**2
    return torch.cat([hAx,torch.reshape(torch.unsqueeze(hpAx,-1)*self.layer.weight,(-1,self.channels))],0)
# end layer

class CloseLayer(nn.Module):
  def __init__(self,channels):
    super(CloseLayer, self).__init__()
    self.layer = nn.Linear(channels,1,dtype=torch.float32)
    self.channels = channels

  def forward(self, x_Jx):
    pivot = x_Jx.shape[0]//3
    x = x_Jx[0:pivot]
    Ax = self.layer(x)
    Jx = torch.reshape(x_Jx[pivot::],(x.shape[0],self.channels,2))
    AJx = torch.reshape(torch.sum((torch.unsqueeze(Jx,1)*torch.unsqueeze(torch.unsqueeze(self.layer.weight,0),-1)),2),(-1,1))
    return torch.cat([Ax,AJx],0)
# end layer

class StepLayer(nn.Module):
  def __init__(self,channels):
    super(StepLayer, self).__init__()
    self.layer = nn.Linear(channels,channels,dtype=torch.float32)
    self.channels = channels

  def forward(self, x_Jx):
    pivot = x_Jx.shape[0]//3
    x = x_Jx[0:pivot]
    hAx = torch.tanh(self.layer(x))
    hpAx = 1-hAx**2
    Jx = torch.reshape(x_Jx[pivot::],(x.shape[0],self.channels,2))
    hpAJx = torch.reshape(torch.unsqueeze(hpAx,-1)*torch.sum((torch.unsqueeze(Jx,1)*torch.unsqueeze(torch.unsqueeze(self.layer.weight,0),-1)),2),(-1,self.channels))
    return torch.cat([hAx,hpAJx],0)
# end layer

class SerialNet(nn.Module):
  def __init__(self,channels=12,local_steps=8,Tf=1.0):
    super(SerialNet, self).__init__()

    step_layer = lambda: StepLayer(channels)
    
    self.open_nn = OpenLayer(channels)
    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_fwd_levels=1,max_bwd_levels=1,max_iters=1)
    self.parallel_nn.setPrintLevel(0)
    
    self.serial_nn   = self.parallel_nn.buildSequentialOnRoot()
    self.close_nn = CloseLayer(channels)

    self.channels=channels
 
  def forward(self, x):
    x = self.open_nn(x)
    x = self.serial_nn(x)
    x = self.close_nn(x)
    return x
# end SerialNet 

class ParallelNet(nn.Module):
  def __init__(self,channels=12,local_steps=8,Tf=1.0,max_levels=1,max_iters=1,print_level=0):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(channels)
    
    self.open_nn = OpenLayer(channels)
    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_fwd_levels=max_levels,max_bwd_levels=max_levels,max_iters=max_iters)
    self.parallel_nn.setPrintLevel(print_level)
    self.parallel_nn.setCFactor(4)
    self.close_nn = CloseLayer(channels)

    self.channels=channels
 
  def forward(self, x):
    x = self.open_nn(x)
    x = self.parallel_nn(x)
    x = self.close_nn(x)
    return x
# end ParallelNet 


def train(rank, args, model, optimizer, epoch, pde):
    model.train()
    optimizer.zero_grad()

    loss = pde.loss(model)

    loss.backward()
    optimizer.step()
    if not(epoch%10):
        print(loss,loss-pde.loss(model))

def test(rank, model, pde):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        loss = pde.loss(model)

    root_print(rank,'\nloss: {:.4f}\n'.format(loss))

def forward_backward_perf(rank,model,pde):
  model.train()
  model.zero_grad()

  start_fore = timer()
  loss = pde.loss(model)
  end_fore = timer()


  start_back = timer()
  loss.backward()
  end_back = timer()

  root_print(rank,'TIME FORWARD:  %.2e' % (end_fore-start_fore))
  root_print(rank,'TIME BACKWARD: %.2e' % (end_back-start_back))
# end  forward_backward_perf

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # artichtectural settings
    parser.add_argument('--steps', type=int, default=64, metavar='N',
                        help='Number of times steps in the resnet layer (default: 4)')
    parser.add_argument('--channels', type=int, default=4, metavar='N',
                        help='Number of channels in resnet layer (default: 4)')

    # algorithmic settings (gradient descent and batching
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--train-batches', type=int, default=int(50000/50), metavar='N',
                        help='input batch size for training (default: %d)' % (50000/50))
    parser.add_argument('--test-batches', type=int, default=int(10000/50), metavar='N',
                        help='input batch size for training (default: %d)' % (10000/50))
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    # algorithmic settings (parallel or serial)
    parser.add_argument('--force-lp', action='store_true', default=False,
                        help='Use layer parallel even if there is only 1 MPI rank')
    parser.add_argument('--lp-levels', type=int, default=3, metavar='N',
                        help='Layer parallel levels (default: 3)')
    parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                        help='Layer parallel iterations (default: 2)')
    parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                        help='Layer parallel print level default: 0)')

    rank  = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()
    args = parser.parse_args()


    # some logic to default to Serial if on one processor,
    # can be overriden by the user to run layer-parallel
    if args.force_lp:
      force_lp = True
    elif procs>1:
      force_lp = True
    else:
      force_lp = False

    torch.manual_seed(args.seed)

    local_steps = int(args.steps/procs)
    if args.steps % procs!=0:
      root_print(rank,'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
      sys.exit(0)

    kwargs = { }

    if force_lp :
      root_print(rank,'Using ParallelNet')
      model = ParallelNet(channels=args.channels,
                          local_steps=local_steps,
                          max_levels=args.lp_levels,
                          max_iters=args.lp_iters,
                          print_level=args.lp_print)
    else:
      root_print(rank,'Using Serial')
      model = SerialNet(channels=args.channels,local_steps=local_steps)
    
    grid = Grid([[0,1],[0,1]],[args.batch_size,args.batch_size])
    pde = Heat(grid)
    forward_backward_perf(rank,model,pde)

    if force_lp:
      timer_str = model.parallel_nn.getTimersString()
      if rank==0:
        print(timer_str)
    # eend force_lp

    #model = None
    #return 

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    #epoch_times = []
    #test_times = []
    #for epoch in range(1, args.epochs + 1):
    #    start_time = timer()
    #    train(rank,args, model, optimizer, epoch,pde)
    #    end_time = timer()
    #    epoch_times += [end_time-start_time]

        #scheduler.step()

    #root_print(rank,'TIME PER EPOCH: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times),stats.stdev(epoch_times)))
    #root_print(rank,'TIME PER TEST:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))

    #print('shape should be',grid.x01.shape[0:-1]+(1,),'but is',model(grid.x01).shape)
    forward_backward_perf(rank,model,pde)
if __name__ == '__main__':
    main()
