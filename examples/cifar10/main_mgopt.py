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

# some helpful examples
# 
# BATCH_SIZE=50
# STEPS=12
# CHANNELS=8

# IN SERIAL
# python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out
# mpirun -n 4 python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out

from __future__ import print_function
import sys
import argparse
import statistics as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from torchvision import datasets, transforms

from timeit import default_timer as timer

from mpi4py import MPI

def root_print(rank,s):
  if rank==0:
    print(s)

class OpenLayer(nn.Module):
  def __init__(self,channels):
    super(OpenLayer, self).__init__()
    ker_width = 3
    self.conv = nn.Conv2d(3,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv(x))
# end layer

class CloseLayer(nn.Module):
  def __init__(self,channels):
    super(CloseLayer, self).__init__()
    ker_width = 3

    # this is to really eliminate the size of image 
    self.pool = nn.MaxPool2d(3)

    self.fc1 = nn.Linear(channels*10*10, 16)
    self.fc2 = nn.Linear(16, 10)

  def forward(self, x):
    x = self.pool(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output
# end layer

class StepLayer(nn.Module):
  def __init__(self,channels):
    super(StepLayer, self).__init__()
    ker_width = 3
    self.conv1 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.conv2 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x):
    return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
# end layer

class SerialNet(nn.Module):
  def __init__(self,channels=12,local_steps=8,Tf=1.0):
    super(SerialNet, self).__init__()

    step_layer = lambda: StepLayer(channels)
    
    self.open_nn = OpenLayer(channels)
    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_fwd_levels=1,max_bwd_levesl=1,max_iters=1)
    self.parallel_nn.setPrintLevel(0)
    
    self.serial_nn   = self.parallel_nn.buildSequentialOnRoot()
    self.close_nn = CloseLayer(channels)
 
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

    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_fwd_levels=max_levels,max_bwd_levels=max_levels,max_iters=max_iters)
    self.parallel_nn.setPrintLevel(print_level)
    self.parallel_nn.setCFactor(4)
    self.parallel_nn.setSkipDowncycle(True)
    
    self.parallel_nn.setFwdNumRelax(1)         # FCF elsewhere
    self.parallel_nn.setFwdNumRelax(0,level=0) # F-Relaxation on the fine grid for forward solve

    self.parallel_nn.setBwdNumRelax(1)         # FCF elsewhere
    self.parallel_nn.setBwdNumRelax(0,level=0) # F-Relaxation on the fine grid for backward solve

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()
    
    # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels) 
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenLayer,channels)
    self.close_nn = compose(CloseLayer,channels)
 
  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x) 
    # this makes sure this is run on only processor 0

    x = self.compose(self.open_nn,x)
    x = self.parallel_nn(x)
    x = self.compose(self.close_nn,x)

    return x
# end ParallelNet 


##
# Custom loss function with MG/Opt Term
def my_criterion(output, target, network_parameters=None, v=None):
  '''
  Define cross entropy loss with optional new MG/Opt term
  '''
  if True:
    # Manually define basic cross-entropy
    log_prob = -1.0 * F.log_softmax(output, 1)
    loss = log_prob.gather(1, target.unsqueeze(1))
    loss = loss.mean()
  else:
    # Use PyTorch's loss, which should be identical
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
  
  # Compute MGOPT term (be careful to use only PyTorch functions)
  if (network_parameters is not None) and (v is not None): 
    mgopt_term = tensor_list_dot(v, network_parameters)
    return loss - mgopt_term
  else:
    return loss


##
# Carry out one complete training epoch 
def train_epoch(rank, args, model, train_loader, optimizer, epoch, compose):
  model.train()
  
  total_time = 0.0
  for batch_idx, (data, target) in enumerate(train_loader):
    start_time = timer()
    optimizer.zero_grad()
    output = model(data)
    #output.retain_grad()

    #network_parameters = [ params for params in model.parameters() ]
    #loss = compose(my_criterion,output,target, network_parameters, network_parameters)
    loss = compose(my_criterion,output,target)
    loss.backward()
    stop_time = timer()
    optimizer.step()

    total_time += stop_time-start_time
    if batch_idx % args.log_interval == 0:
      root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))

  root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
    100. * (batch_idx+1) / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))


def test(rank, model, test_loader,compose):
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data, target
      output = model(data)
      test_loss += compose(criterion,output,target).item()
       
      output = MPI.COMM_WORLD.bcast(output,root=0)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  root_print(rank,'\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

def compute_levels(num_steps,min_coarse_size,cfactor): 
  from math import log, floor 
  
  # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
  levels =  floor(log(num_steps/min_coarse_size,cfactor))+1 

  if levels<1:
    levels = 1
  return levels
# end compute levels


# compute dot product of two vectors, v and w, where each vector is a list of tensors
def tensor_list_dot(v, w):
  return sum([ torch.dot(vv.flatten(), ww.flatten()) for (vv,ww) in zip(v, w) ])

# compute AXPY two vectors, v and w, where each vector is a list of tensors
# if inplace is True, then w = alpha*v + beta*w
# else, return a new vector equal to alpha*v + beta*w 
def tensor_list_AXPY(alpha, v, beta, w, inplace=False):
  if inplace:
    for (vv, ww) in zip(v, w):
      ww[:] = alpha*vv + beta*ww
  else:
    return [ alpha*vv + beta*ww for (vv,ww) in zip(v, w) ]

# return a deep copy of the tensor list w
def tensor_list_deep_copy(w):
  return [ torch.clone(ww) for ww in w ]

# See
# https://stackoverflow.com/questions/383565/how-to-iterate-over-a-list-repeating-each-element-in-python
def duplicate(iterable,n):
  """A generator that repeats each entry n times"""
  for item in iterable:
    first = True
    for _ in range(n):
      yield item,first
      first = False

def interpolate_network_params(model, cf=2, deep_copy=False, grad=False):
  
  ''' 
  Interpolate the model parameters according to coarsening-factor in time cf.
  Return a list of the interpolated model parameters.

  If deep_copy is True, return a deep copy.
  If grad is True, return the network gradient instead
  '''
  
  interp_params = []

  # loop over all the children, interpolating the layer-parallel weights
  for child in model.children():
    # handle layer parallel modules differently 
    if isinstance(child, torchbraid.LayerParallel):
      
      # loop over each layer-parallel layer -- this is where the "interpolation" occurs
      for (lp_child, lp_f) in duplicate(child.layer_models, cf):
        with torch.no_grad():
          for param in lp_child.parameters():
            if deep_copy:
              if grad: interp_params.append(torch.clone(param.grad))
              else:    interp_params.append(torch.clone(param))
            else: # no deep copy
              if grad: interp_params.append(param.grad)
              else:    interp_params.append(param)
             
    else:
      # Do simple injection for the opening and closing layers
      with torch.no_grad():
        for param in child.parameters():
          if deep_copy:
            if grad: interp_params.append(torch.clone(param.grad))
            else:    interp_params.append(torch.clone(param))
          else: # no deep copy
            if grad: interp_params.append(param.grad)
            else:    interp_params.append(param)
    ##

  return interp_params


def restrict_network_params(model, cf=2, deep_copy=False, grad=False):
  ''' 
  Restrict the model parameters according to coarsening-factor in time cf.
  Return a list of the restricted model parameters.

  If deep_copy is True, return a deep copy.
  If grad is True, return the network gradient instead
  '''
  
  restrict_params = []

  # loop over all the children, restricting 
  for child in model.children():
    
    # handle layer parallel modules differently 
    if isinstance(child, torchbraid.LayerParallel):
      
      # loop over each layer-parallel layer -- this is where the "restriction" occurs
      for lp_child in child.layer_models[0:-1:cf]: 
        with torch.no_grad():
          for param in lp_child.parameters(): 
            if deep_copy:
              if grad: restrict_params.append(torch.clone(param.grad))
              else:    restrict_params.append(torch.clone(param))
            else: # no deep copy
              if grad: restrict_params.append(param.grad)
              else:    restrict_params.append(param)
             
    else:
      # Do simple injection for the opening and closing layers
      with torch.no_grad():
        for param in child.parameters(): 
          if deep_copy:
            if grad: restrict_params.append(torch.clone(param.grad))
            else:    restrict_params.append(torch.clone(param))
          else: # no deep copy
            if grad: restrict_params.append(param.grad)
            else:    restrict_params.append(param)
    ##

  return restrict_params 


def write_network_params_inplace(model, new_params):
  '''
  Write the parameters of model in-place, overwriting with new_params
  '''
  
  with torch.no_grad():
    old_params = list(model.parameters())
    
    assert(len(old_params) == len(new_params)) 
    
    for (op, np) in zip(old_params, new_params):
      op[:] = np[:]

def get_network_params(model, deep_copy=False, grad=False):
  
  if deep_copy:
    if grad: pp = [torch.clone(params.grad) for params in model.parameters() ]
    else:    pp = [torch.clone(params)      for params in model.parameters() ]
  else:
    if grad: pp = [params.grad for params in model.parameters() ]
    else:    pp = [params      for params in model.parameters() ]
  ##

  return pp


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='TORCHBRAID CIFAR10 Example')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--channels', type=int, default=4, metavar='N',
                      help='Number of channels in resnet layer (default: 4)')

  # algorithmic settings (gradient descent and batching
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=2, metavar='N',
                      help='number of epochs to train (default: 2)')
  parser.add_argument('--samp-ratio', type=float, default=1.0, metavar='N',
                      help='number of samples as a ratio of the total number of samples')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (parallel or serial)
  parser.add_argument('--force-lp', action='store_true', default=False,
                      help='Use layer parallel even if there is only 1 MPI rank')
  parser.add_argument('--lp-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels (default: 3)')
  parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                      help='Layer parallel iterations (default: 2)')
  parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                      help='Layer parallel print level (default: 0)')

  # algorithmic settings (nested iteration)
  parser.add_argument('--ni-levels', type=int, default=3, metavar='N',
                      help='Number of nested iteration levels (default: 3)')
  parser.add_argument('--ni-rfactor', type=int, default=2, metavar='N',
                      help='Refinment factor for nested iteration (default: 2)')
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument('--ni-fixed-coarse', dest='ni_fixed_coarse', action='store_true',
                      help='Fix the weights on the coarse levels once trained (default: off)')
  group.add_argument('--ni-no-fixed-coarse', dest='ni_fixed_coarse', action='store_false',
                      help='Fix the weights on the coarse levels once trained (default: off)')
  parser.set_defaults(ni_fixed_coarse=False)

  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  torch.manual_seed(torchbraid.utils.seed_from_rank(args.seed,rank))

  if args.lp_levels==-1:
    min_coarse_size = 7
    args.lp_levels = compute_levels(args.steps,min_coarse_size,4)

  if args.steps % procs!=0:
    root_print(rank,'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
    sys.exit(0)

  ni_levels = args.ni_levels
  ni_rfactor = args.ni_rfactor
  if args.steps % ni_rfactor**(ni_levels-1) != 0:
    root_print(rank,'Steps combined with the coarsest nested iteration level must be an even multiple: %d %d %d' % (args.steps,ni_rfactor,ni_levels-1))
    sys.exit(0)

  if args.steps / ni_rfactor**(ni_levels-1) % procs != 0:
    root_print(rank,'Coarsest nested iteration must fit on the number of processors')
    sys.exit(0)

  transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
  train_set = datasets.CIFAR10('./data', download=False,
                                   transform=transform,train=True)
  test_set  = datasets.CIFAR10('./data', download=False,
                                   transform=transform,train=False)

  # reduce the number of samples for faster execution
  train_set = torch.utils.data.Subset(train_set,range(int(50000*args.samp_ratio)))
  test_set = torch.utils.data.Subset(test_set,range(int(10000*args.samp_ratio)))

  train_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=args.batch_size, 
                                             shuffle=True)
  test_loader  = torch.utils.data.DataLoader(test_set,
                                             batch_size=args.batch_size, 
                                             shuffle=False)

  ni_steps = [int(args.steps/(ni_rfactor**(ni_levels-i-1))) for i in range(ni_levels)]

  root_print(rank,ni_steps)

# 1) Hard-code V-cycle down below the F-cycle
#   - Done with restrict
#   - Done with getting a gradient 
#   - Done with modified loss function
#
#   Write cycle   
#
# 2) Want a more general cycle structure to allow for V- and F-
#    Encapsulate more functionality in functions (???  like compute gradient, which feeds into optimization step?  or leave exposed for now...)
#    Do you want multilevel MG/Opt object?  Probably...
#    - Core data structure would be list of models
#    - Functions would be initialize (call below NI) with a few params
#    - And solve with V or F cycle
#
# 3) Want code to be extended to allow for two mini-batch loops, one outside of MG/Opt, and one inside MG/Opt that is used by each relaxation step.
#    Perhaps, you want train_loader to be post-processed into doubly nested list, where each inside list is the set of batch(es) for relaxation to 
#    iterate over, [  [],   [],   [],  .... ]
#

  ##
  # Initialize with Nested Iteration
  models = []
  cf = 2
  root_print(rank,'Using ParallelNet')
  for steps in ni_steps:

    local_steps = int(steps/procs)
    model = ParallelNet(channels=args.channels,
                        local_steps=local_steps,
                        max_levels=args.lp_levels,
                        max_iters=args.lp_iters,
                        print_level=args.lp_print)

    # pass the weights along to the next iteration
    if len(models) > 0: 
      
      #interpolate_network(model, models[-1], args.ni_fixed_coarse, cf)

      new_params = interpolate_network_params(models[-1], cf=cf, deep_copy=True, grad=False)
      write_network_params_inplace(model, new_params)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    counts = MPI.COMM_WORLD.gather((total_params,trainable_params))

    if rank==0:
      root_print(rank,'-------------------------------------')
      root_print(rank,'-- optimizing %d steps\n' % steps)
      root_print(rank,'-- total params: {}'.format([c[0] for c in counts]))
      root_print(rank,'-- train params: {}'.format([c[1] for c in counts]))
      root_print(rank,'-------------------------------------\n')
    

    # the idea of this object is to handle the parallel communication associated
    # with layer parallel
    compose = model.compose
  
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
  
    epoch_times = []
    test_times = []
    for epoch in range(1, args.epochs + 1):
      start_time = timer()
      train_epoch(rank, args, model, train_loader, optimizer, epoch, compose)
      end_time = timer()
      epoch_times += [end_time-start_time]
  
      start_time = timer()
      test(rank,model, test_loader,compose)
      end_time = timer()
      test_times += [end_time-start_time]
    
    # Store Model
    models.append(model)

  import pdb; pdb.set_trace()

  # is the ordering of parameters the same, when you iterate over children, and
  #    parameters?  That appears to be the case.  If its ever not, your dot
  #    product will break...so maybe not worry about it.
  #
  # do you need to with torch.no_grad()?
  #
  #
  # (do you want to "verify" this and then use the multilevel solver PyAMG
  #    object, and model the solver design off of SA?  For easy relaxation
  #    plug-n-play?
  #
  # make this a recursive function
  # look at linesearch, just do 1, 1/2, 1/4, 1/8/, 1/16, 1/32, halt.  Eval objective, if smaller, stop.  else halve
  # re-arrange functions that are V-cycle specific
  #
  # encapsulate relaxation in function
  #
  # encapsulate the fwd+bwd evaluation of a network (?)  and leave the optimizer.step() to be done afterward ?
  #
  # encapsulate simple line search...
  #
  # ==> clean up code, and combine with list above ... !


  # Now, carry out V-cycles.  Hierarchy is initialized.
  # First, reverse list, so entry 0 is the finest level
  models.reverse()
  #
  ncycles = 1
  nrelax = 1
  nrelax_coarse = 5 # number of optimizations for coarse-grid solve
  cf = 2
  lr = args.lr
  n_line_search = 7
  #
  # likely recursive parameters 
  lvl = 0
  v = None
  #
  for batch_idx, (data, target) in enumerate(train_loader):
    
    for cycle in range(ncycles):

      optimizer_fine = optim.SGD(models[lvl].parameters(), lr=lr, momentum=0.9)
    
      # 1. relax (carry out optimization steps)
      for k in range(nrelax):
        optimizer_fine.zero_grad()
        output = models[lvl](data)
        if lvl == 0:
          loss = compose(my_criterion, output, target)
        else: # add the MG/Opt Term
          x_h = get_network_params(models[lvl], deep_copy=False, grad=False)
          loss = compose(my_criterion, output, target, x_h, v)
        ##
        loss.backward()
        optimizer_fine.step()
  
   
      # 2. compute new gradient g_h
      # First evaluate network, forward and backward
      optimizer_fine.zero_grad()
      output = models[lvl](data)
      if lvl == 0:
        loss = compose(my_criterion, output, target)
      else: # add the MG/Opt Term
        x_h = get_network_params(models[lvl], deep_copy=False, grad=False)
        loss = compose(my_criterion, output, target, x_h, v)
      ##
      loss.backward()
      fine_loss = loss.item()
      # Second, note that the gradient is waiting in models[lvl], to accessed next


      # 3. restrict parameters (x_h) and gradient (g_h) to H
      #    x_H^{zero} = R(x_h)   and    \tilde{g}_H = R(g_h)
      with torch.no_grad():
        gtilde_H = restrict_network_params(models[lvl], cf=cf, deep_copy=True, grad=True)
        x_H      = restrict_network_params(models[lvl], cf=cf, deep_copy=True, grad=False)
        # For x_H to take effect, these parameters must be written to the next coarser network
        write_network_params_inplace(models[lvl+1], x_H)
        # Must store x_H for later error computation
        x_H_zero = tensor_list_deep_copy(x_H)


      # 4. compute gradient on coarse level, using restricted parameters
      #  g_H = grad( f_H(x_H) )
      optimizer_coarse = optim.SGD(models[lvl+1].parameters(), lr=lr, momentum=0.9)
      optimizer_coarse.zero_grad()
      output = models[lvl+1](data)
      loss = compose(my_criterion, output, target)  # we just want gradient of f_H here, with no MG/Opt term
      loss.backward()
      with torch.no_grad():
        g_H = get_network_params(models[lvl+1], deep_copy=True, grad=True)
   
  
      # 5. compute coupling term
      #  v = g_H - \tilde{g}_H
      with torch.no_grad():
        v_H = tensor_list_AXPY(1.0, g_H, -1.0, gtilde_H)


      # 6. solve coarse-grid (eventually do recursive call if not coarsest level)
      #  x_H = min f_H(x_H) - <v_H, x_H>
      for m in range(nrelax_coarse):
        optimizer_coarse.zero_grad()
        output = models[lvl+1](data)
        x_H = get_network_params(models[lvl+1], deep_copy=False, grad=False)
        loss = compose(my_criterion, output, target, x_H, v_H)
        loss.backward()
        optimizer_coarse.step()


      # 7. interpolate correction to fine-level
      #  e_h = P( x_H - x_H^{init})
      with torch.no_grad():
        x_H = get_network_params(models[lvl+1], deep_copy=False, grad=False)
        e_H = tensor_list_AXPY(1.0, x_H, -1.0, x_H_zero)
        #
        # to correctly interpolate e_H --> e_h, we need to put these values in a
        # network, so the interpolation knows the layer-parallel structure and
        # where to refine.
        write_network_params_inplace(models[lvl+1], e_H)
        e_h = interpolate_network_params(models[lvl+1], cf=cf, deep_copy=True, grad=False)
      

      # 8. apply linesearch to update x_h
      #  x_h = x_h + alpha*e_h
      #     
      # Simple line-search: Add e_h to fine parameters.  If loss has
      # diminished, stop.  Else subtract 1/2 of e_h from fine parameters, and
      # continue until loss is reduced.
      
      # Add error update to x_h
      # alpha*e_h + x_h --> x_h
      with torch.no_grad():
        x_h = get_network_params(models[lvl], deep_copy=False, grad=False)
        alpha = 1.0
        tensor_list_AXPY(alpha, e_h, 1.0, x_h, inplace=True)
      
        # Start Line search 
        for m in range(n_line_search):
          print("wwww", m, alpha)
          
          optimizer_fine.zero_grad()
          output = models[lvl](data)
          if lvl == 0:
            loss = compose(my_criterion, output, target)
          else: # add the MG/Opt Term
            x_h = get_network_params(models[lvl], deep_copy=False, grad=False)
            loss = compose(my_criterion, output, target, x_h, v)
          ##
          new_loss = loss.item()
          if new_loss < fine_loss:
            break
          elif m < (n_line_search-1): 
            alpha = alpha/2.0
            tensor_list_AXPY(-alpha, e_h, 1.0, x_h, inplace=True)

      ##
      # no need to return anything, fine network is modified in place

  root_print(rank,'TIME PER EPOCH: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times),stats.stdev(epoch_times)))
  root_print(rank,'TIME PER TEST:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))

if __name__ == '__main__':
  main()




#####################
# Code Graveyard
#####################

# Get access to individual layers, and gradient.  Sometimes, you have to call
#  .retain_grad(), and then you can access .grad after calling backward(), 

if False:
  model.parallel_nn.local_layers[i].conv1.weight.grad
  model.parallel_nn.local_layers[i].bn1.weight.grad

  model.parallel_nn.local_layers[i].conv2.weight.grad
  model.parallel_nn.local_layers[i].bn2.weight.grad

  model.open_nn.conv.bias.grad
  model.open_nn.conv.weight.grad

  model.close_nn.pool  #How to get gradient for pooling? 

  model.close_nn.fc1.weight.grad
  model.close_nn.fc1.bias.grad

  model.close_nn.fc2.weight.grad
  model.close_nn.fc2.bias.grad



# Custom loss function
# https://discuss.pytorch.org/t/custom-loss-functions/29387/2


# Test restriction after running NI on [1, 2, 4] layers
if False:
  # Print weights before restriction
  import pdb; pdb.set_trace()
  print("Model 0 to Model 1 difference")
  diff = models[1].parallel_nn.local_layers[0].conv1.weight.flatten() - models[0].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[1].parallel_nn.local_layers[0].bn1.weight.flatten() - models[0].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[1].parallel_nn.local_layers[0].conv2.weight.flatten() - models[0].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[1].parallel_nn.local_layers[0].bn2.weight.flatten() - models[0].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[1].parallel_nn.local_layers[1].conv1.weight.flatten() - models[0].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[1].parallel_nn.local_layers[1].bn1.weight.flatten() - models[0].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[1].parallel_nn.local_layers[1].conv2.weight.flatten() - models[0].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[1].parallel_nn.local_layers[1].bn2.weight.flatten() - models[0].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  # Now, restrict
  new_params = restrict_network_params(models[1], cf=2, deep_copy=True, grad=False)
  write_network_params_inplace(models[0], new_params)
  #
  print("After restrict: Model 0 to Model 1 difference")
  diff = models[1].parallel_nn.local_layers[0].conv1.weight.flatten() - models[0].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[1].parallel_nn.local_layers[0].bn1.weight.flatten() - models[0].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[1].parallel_nn.local_layers[0].conv2.weight.flatten() - models[0].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[1].parallel_nn.local_layers[0].bn2.weight.flatten() - models[0].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[1].parallel_nn.local_layers[1].conv1.weight.flatten() - models[0].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[1].parallel_nn.local_layers[1].bn1.weight.flatten() - models[0].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[1].parallel_nn.local_layers[1].conv2.weight.flatten() - models[0].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[1].parallel_nn.local_layers[1].bn2.weight.flatten() - models[0].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  print("\n\n")

  print("Model 1 to Model 2 difference")
  diff = models[2].parallel_nn.local_layers[0].conv1.weight.flatten() - models[1].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[0].bn1.weight.flatten() - models[1].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[0].conv2.weight.flatten() - models[1].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[0].bn2.weight.flatten() - models[1].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[1].conv1.weight.flatten() - models[1].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[1].bn1.weight.flatten() - models[1].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[1].conv2.weight.flatten() - models[1].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[1].bn2.weight.flatten() - models[1].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[2].conv1.weight.flatten() - models[1].parallel_nn.local_layers[1].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[2].bn1.weight.flatten() - models[1].parallel_nn.local_layers[1].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[2].conv2.weight.flatten() - models[1].parallel_nn.local_layers[1].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[2].bn2.weight.flatten() - models[1].parallel_nn.local_layers[1].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[3].conv1.weight.flatten() - models[1].parallel_nn.local_layers[1].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[3].bn1.weight.flatten() - models[1].parallel_nn.local_layers[1].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[3].conv2.weight.flatten() - models[1].parallel_nn.local_layers[1].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[3].bn2.weight.flatten() - models[1].parallel_nn.local_layers[1].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  # Now, restrict
  new_params = restrict_network_params(models[2], cf=2, deep_copy=True, grad=False)
  write_network_params_inplace(models[1], new_params)
  #
  print("After restrict: Model 1 to Model 2 difference")
  diff = models[2].parallel_nn.local_layers[0].conv1.weight.flatten() - models[1].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[0].bn1.weight.flatten() - models[1].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[0].conv2.weight.flatten() - models[1].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[0].bn2.weight.flatten() - models[1].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[1].conv1.weight.flatten() - models[1].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[1].bn1.weight.flatten() - models[1].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[1].conv2.weight.flatten() - models[1].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[1].bn2.weight.flatten() - models[1].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[2].conv1.weight.flatten() - models[1].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[2].bn1.weight.flatten() - models[1].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[2].conv2.weight.flatten() - models[1].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[2].bn2.weight.flatten() - models[1].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[3].conv1.weight.flatten() - models[1].parallel_nn.local_layers[0].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[3].bn1.weight.flatten() - models[1].parallel_nn.local_layers[0].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[3].conv2.weight.flatten() - models[1].parallel_nn.local_layers[0].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[3].bn2.weight.flatten() - models[1].parallel_nn.local_layers[0].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  ###
  print("")
  ###
  diff = models[2].parallel_nn.local_layers[0].conv1.weight.flatten() - models[1].parallel_nn.local_layers[1].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[0].bn1.weight.flatten() - models[1].parallel_nn.local_layers[1].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[0].conv2.weight.flatten() - models[1].parallel_nn.local_layers[1].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[0].bn2.weight.flatten() - models[1].parallel_nn.local_layers[1].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[1].conv1.weight.flatten() - models[1].parallel_nn.local_layers[1].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[1].bn1.weight.flatten() - models[1].parallel_nn.local_layers[1].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[1].conv2.weight.flatten() - models[1].parallel_nn.local_layers[1].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[1].bn2.weight.flatten() - models[1].parallel_nn.local_layers[1].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[2].conv1.weight.flatten() - models[1].parallel_nn.local_layers[1].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[2].bn1.weight.flatten() - models[1].parallel_nn.local_layers[1].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[2].conv2.weight.flatten() - models[1].parallel_nn.local_layers[1].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[2].bn2.weight.flatten() - models[1].parallel_nn.local_layers[1].bn2.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[3].conv1.weight.flatten() - models[1].parallel_nn.local_layers[1].conv1.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[3].bn1.weight.flatten() - models[1].parallel_nn.local_layers[1].bn1.weight.flatten()
  print(torch.dot(diff, diff) )
  #
  diff = models[2].parallel_nn.local_layers[3].conv2.weight.flatten() - models[1].parallel_nn.local_layers[1].conv2.weight.flatten()
  print(torch.dot(diff, diff) )
  diff = models[2].parallel_nn.local_layers[3].bn2.weight.flatten() - models[1].parallel_nn.local_layers[1].bn2.weight.flatten()
  print(torch.dot(diff, diff) )


########
def interpolate_network(dest_model, src_model, grad_by_level=True, cf=2):
  ''' 
  Interpolate the src_model to the dest_model by injecting the closing and
  opening layers, and doing piece-wise constant interpolation for the
  layer-parallel part.  The refinement factor "in-time" is cf.
  '''

  # models should have the same number of children
  assert(len(list(dest_model.children()))==len(list(src_model.children())))

  rank  = MPI.COMM_WORLD.Get_rank()

  # loop over all the children, interpolating the weights
  for dest,src in zip(dest_model.children(),src_model.children()):
    
    # do something special for layer parallel modules
    if isinstance(dest,torchbraid.LayerParallel):
      # both are layer parallel
      assert(type(dest) is type(src))
      
      # loop over each layer-parallel layer -- this is where the "interpolation" occurs
      for lp_dest,(lp_src,lp_f) in zip(dest.layer_models, duplicate(src.layer_models, cf)):
        with torch.no_grad():
          for d,s in zip(lp_dest.parameters(),lp_src.parameters()):
            d.copy_(s)
             
            # # if it's the first one, detach from the graph...making it faster (???)
            if lp_f and grad_by_level: 
              d.requires_grad = False
    else:
      # Do simple injection for the opening and closing layers
      with torch.no_grad():
        for d,s in zip(dest.parameters(),src.parameters()):
          d.copy_(s)





