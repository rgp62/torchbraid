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
import numpy as np

import torch
from torchvision import datasets, transforms
from mpi4py import MPI
from mgopt import parse_args, mgopt_solver


import scipy
import scipy.special
import numpy as np
import matplotlib.pylab as plt

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
        

class Advection:
    def __init__(self,grid):
        self.grid = grid
        self.F1BC = torch.zeros(self.grid.x1[:,:,0:1,...,0].shape + (1,),dtype=torch.float32)
        x = self.grid.x0[0:1,...,1::].numpy()
        self.F0BC = torch.tensor(np.piecewise(x, [x < .4,(x>=.4) & (x<.5), (x>=.5) & (x<.6), x >= .6], [0., lambda x:10*(x-.4), lambda x: -10*(x-.6),0]),dtype=torch.float32)

    def loss(self,u):

        ux = u(self.grid.xall)
        ux0 = torch.reshape(ux[0:self.grid.x0len],(self.grid.x0.shape[0:-1]+(1,)))
        ux1 = torch.reshape(ux[self.grid.x0len::],(self.grid.x1.shape[0:-1]+(1,)))
        F0ux = torch.cat([self.F0BC,ux0[1::]],axis=0)
        F1ux = torch.cat([self.F1BC,ux1[:,:,1::]],axis=2)
        
        F0int  = self.grid.dx[1]/2*torch.einsum('abix,i->abx',F0ux,self.grid.wi0)
        F1int  = self.grid.dx[0]/2*torch.einsum('aibx,i->abx',F1ux,self.grid.wi1)
        
        return (
                (F0int[1::] - F0int[0:-1])  \
               +(F1int[:,1::] - F1int[:,0:-1])
               )



def main():
  
  ##
  # Parse command line args (function defined above)
  args = parse_args()
  procs = MPI.COMM_WORLD.Get_size()
  rank  = MPI.COMM_WORLD.Get_rank()
  
  
  ##
  # Compute number of nested iteration steps, going from fine to coarse
  ni_steps = np.array([int(args.steps/(args.ni_rfactor**(args.ni_levels-i-1))) for i in range(args.ni_levels)])
  ni_steps = ni_steps[ ni_steps != 0 ]
  local_ni_steps = np.flip( np.array(ni_steps / procs, dtype=int) )
  print("\nNested iteration steps:  " + str(ni_steps))

  ##
  # Define ParNet parameters for each nested iteration level, starting from fine to coarse
  networks = [] 
  for lsteps in local_ni_steps: 
    networks.append( ('ParallelNet', {'channels'          : args.channels, 
                                      'local_steps'       : lsteps,
                                      'max_iters'         : args.lp_iters,
                                      'print_level'       : args.lp_print,
                                      'Tf'                : args.tf,
                                      'max_fwd_levels'    : args.lp_fwd_levels,
                                      'max_bwd_levels'    : args.lp_bwd_levels,
                                      'max_fwd_iters'     : args.lp_fwd_iters,
                                      'print_level'       : args.lp_print,
                                      'braid_print_level' : args.lp_braid_print,
                                      'fwd_cfactor'       : args.lp_fwd_cfactor,
                                      'bwd_cfactor'       : args.lp_bwd_cfactor,
                                      'fine_fwd_fcf'      : args.lp_fwd_finefcf,
                                      'fine_bwd_fcf'      : args.lp_bwd_finefcf,
                                      'fwd_nrelax'        : args.lp_fwd_nrelax_coarse,
                                      'bwd_nrelax'        : args.lp_bwd_nrelax_coarse,
                                      'skip_downcycle'    : not args.lp_use_downcycle,
                                      'fmg'               : args.lp_use_fmg,
                                      'fwd_relax_only_cg' : args.lp_fwd_relaxonlycg,
                                      'bwd_relax_only_cg' : args.lp_bwd_relaxonlycg,
                                      'CWt'               : args.lp_use_crelax_wt,
                                      'fwd_finalrelax'    : args.lp_fwd_finalrelax
                                      }))
                                 
  ##
  # Specify optimization routine on each level, starting from fine to coarse
  optims = [ ("pytorch_sgd", { 'lr':args.lr, 'momentum':0.9}) for i in range(len(ni_steps)) ]



  grid = Grid([[0,1],[0,1]],[64,64])
  pde = Advection(grid)

  ##
  # Initialize MG/Opt solver with nested iteration 
  epochs = 2
  mgopt_printlevel = 1
  log_interval = args.log_interval
  mgopt = mgopt_solver()
  mgopt.initialize_with_nested_iteration(ni_steps, pde,
          networks, epochs=epochs, log_interval=log_interval,
          mgopt_printlevel=mgopt_printlevel, optims=optims, seed=args.seed) 
   
  print(mgopt)
  mgopt.options_used()
  
  ###
  ## Turn on for fixed-point test.  
  ## Works when running  $$ python3 main_mgopt.py --samp-ratio 0.002 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --mgopt-printlevel 3 --batch-size 1
  #if False:
  #  import torch.nn as nn
  #  criterion = nn.CrossEntropyLoss()
  #  train_set = torch.utils.data.Subset(dataset, [1])
  #  train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=False)
  #  for (data,target) in train_loader:  pass
  #  model = mgopt.levels[0].model
  #  with torch.no_grad():
  #    model.eval()
  #    output = model(data)
  #    loss = model.compose(criterion, output, target)
  #  
  #  print("Doing fixed point test.  Loss on single training example should be zero: " + str(loss.item()))
  #  model.train()

 # Can change MGRIT options from NI to MG/Opt with the following
 #mgopt.levels[0].model.parallel_nn.setFwdNumRelax(0,level=0) 
  
  ##
  # Run the MG/Opt solver
  #   Note: that we use the default restrict and interp options, but these can be modified on a per-level basis
  if( args.mgopt_iter > 0):
    epochs = args.epochs
    line_search = ('no_line_search', {'a' : 1.0})
    log_interval = args.log_interval
    mgopt_printlevel = args.mgopt_printlevel
    mgopt_iter = args.mgopt_iter
    mgopt_levels = args.mgopt_levels
    mgopt_tol=0
    nrelax_pre = args.mgopt_nrelax_pre
    nrelax_post = args.mgopt_nrelax_post
    nrelax_coarse = args.mgopt_nrelax_coarse
    mgopt.mgopt_solve(pde, epochs=epochs,
            log_interval=log_interval, mgopt_tol=mgopt_tol,
            mgopt_iter=mgopt_iter, nrelax_pre=nrelax_pre,
            nrelax_post=nrelax_post, nrelax_coarse=nrelax_coarse,
            mgopt_printlevel=mgopt_printlevel, mgopt_levels=mgopt_levels,
            line_search=line_search)
   
    print(mgopt)
    mgopt.options_used()
  ##
  


if __name__ == '__main__':
  main()



