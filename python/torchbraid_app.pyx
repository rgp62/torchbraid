# cython: profile=True
# cython: linetrace=True

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from mpi4py import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

import pickle # we need this for building byte packs
import sys

ctypedef PyObject _braid_App_struct 
ctypedef _braid_App_struct* braid_App

class BraidVector:
  def __init__(self,tensor,level):
    self.tensor_ = tensor 
    self.level_  = level
    self.time_   = np.nan # are we using this???

  def tensor(self):
    return self.tensor_

  def level(self):
    return self.level_
  
  def clone(self):
    cl = BraidVector(self.tensor().clone(),self.level())
    return cl

  def setTime(self,t):
    self.time_ = t

  def getTime(self):
    return self.time_

ctypedef PyObject _braid_Vector_struct
ctypedef _braid_Vector_struct *braid_Vector

include "./braid.pyx"
include "./torchbraid_callbacks.pyx"

#  a python level module
##########################################################

cdef class MPIData:
  cdef MPI.Comm comm
  cdef int rank
  cdef int size

  def __cinit__(self,comm):
    self.comm = comm
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

  def getComm(self):
    return self.comm

  def getRank(self):
    return self.rank 

  def getSize(self):
    return self.size
# helper class for the MPI communicator

##############################################################

class BraidApp:

  def __init__(self,comm,local_num_steps,Tf,max_levels,max_iters):
    # optional parameters
    self.max_levels  = max_levels
    self.max_iters   = max_iters
    self.print_level = 2
    self.nrelax      = 0
    self.cfactor     = 2
    self.skip_downcycle = 0

    self.mpi_data        = MPIData(comm)
    self.Tf              = Tf
    self.local_num_steps = local_num_steps
    self.num_steps       = local_num_steps*self.mpi_data.getSize()

    self.dt       = Tf/self.num_steps
    self.t0_local = self.mpi_data.getRank()*local_num_steps*self.dt
    self.tf_local = (self.mpi_data.getRank()+1.0)*local_num_steps*self.dt

    self.x_final = None
  
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    self.py_core = None

    # build up the core
    self.py_core = self.initCore()
  # end __init__

  def initCore(self):
    cdef braid_Core core
    cdef PyBraid_Core py_fwd_core
    cdef braid_Core fwd_core
    cdef double tstart
    cdef double tstop
    cdef int ntime
    cdef MPI.Comm comm = self.mpi_data.getComm()
    cdef int rank = self.mpi_data.getRank()
    cdef braid_App app = <braid_App> self
    cdef braid_PtFcnStep  b_step  = <braid_PtFcnStep> my_step
    cdef braid_PtFcnInit  b_init  = <braid_PtFcnInit> my_init
    cdef braid_PtFcnClone b_clone = <braid_PtFcnClone> my_clone
    cdef braid_PtFcnFree  b_free  = <braid_PtFcnFree> my_free
    cdef braid_PtFcnSum   b_sum   = <braid_PtFcnSum> my_sum
    cdef braid_PtFcnSpatialNorm b_norm = <braid_PtFcnSpatialNorm> my_norm
    cdef braid_PtFcnAccess b_access = <braid_PtFcnAccess> my_access
    cdef braid_PtFcnBufSize b_bufsize = <braid_PtFcnBufSize> my_bufsize
    cdef braid_PtFcnBufPack b_bufpack = <braid_PtFcnBufPack> my_bufpack
    cdef braid_PtFcnBufUnpack b_bufunpack = <braid_PtFcnBufUnpack> my_bufunpack

    ntime = self.num_steps
    tstart = 0.0
    tstop = self.Tf

    braid_Init(comm.ob_mpi, comm.ob_mpi, 
               tstart, tstop, ntime, 
               app,
               b_step, b_init, 
               b_clone, b_free, 
               b_sum, b_norm, b_access, 
               b_bufsize, b_bufpack, b_bufunpack, 
               &core)

    # Set Braid options
    braid_SetMaxLevels(core, self.max_levels)
    braid_SetMaxIter(core, self.max_iters)
    braid_SetPrintLevel(core,self.print_level)
    braid_SetNRelax(core,-1,self.nrelax)
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels
    braid_SetSkip(core,self.skip_downcycle)

    # store the c pointer
    py_core = PyBraid_Core()
    py_core.setCore(core)

    return py_core
  # end initCore

  def __del__(self):
    if self.py_core!=None:
      py_core = <PyBraid_Core> self.py_core
      core = py_core.getCore()

      # Destroy Braid Core C-Struct
      # FIXME: braid_Destroy(core) # this should be on
    # end core

  def runBraid(self,x):
    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    self.setInitial(x)
 
    # Run Braid
    braid_Drive(core)

    f = self.getFinal()

    return f

  def getCore(self):
    return self.py_core    
 
  def setPrintLevel(self,print_level):
    self.print_level = print_level

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetPrintLevel(core,self.print_level)

  def setNumRelax(self,relax):
    self.nrelax = relax 

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetNRelax(core,-1,self.nrelax)

  def setCFactor(self,cfactor):
    self.cfactor = cfactor 

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels

  def setSkipDowncycle(self,skip):
    self.skip_downcycle = skip

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetSkip(core,self.skip_downcycle)

  def setStorage(self,storage):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetStorage(core,storage)

  def setRevertedRanks(self,reverted):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetRevertedRanks(core,reverted)

  def getUVector(self,level,index):
    cdef braid_Core core = (<PyBraid_Core> self.py_core).getCore()
    cdef braid_BaseVector bv
    _braid_UGetVectorRef(core, level, index,&bv)

    return <object> bv.userVector

  def getMPIData(self):
    return self.mpi_data

  def getTimeStepIndex(self,t,tf,level):
    return round((t-self.t0_local) / self.dt)

  def getPrimalIndex(self,t,tf,level):
    ts = round(tf / self.dt)
    return  self.num_steps - ts

  def setInitial(self,x0):
    self.x0 = BraidVector(x0,0)

  def buildInit(self,t):
    x = self.x0.clone()
    if t>0:
      t_x = x.tensor()
      t_x[:] = 0.0
    return x

  def access(self,t,u):
    if t==self.Tf:
      self.x_final = u.clone()

  def getFinal(self):
    if self.x_final==None:
      return None

    # assert the level
    assert(self.x_final.level()==0)
    return self.x_final.tensor()

# end BraidApp

##############################################################

class ForewardBraidApp(BraidApp):

  def __init__(self,comm,layer_models,local_num_steps,Tf,max_levels,max_iters):
    BraidApp.__init__(self,comm,local_num_steps,Tf,max_levels,max_iters)

    # note that a simple equals would result in a shallow copy...bad!
    self.layer_models = [l for l in layer_models]

    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # send everything to the left (this helps with the adjoint method)
    if my_rank>0:
      comm.send(self.layer_models[0],dest=my_rank-1,tag=22)
    if my_rank<num_ranks-1:
      neighbor_model = comm.recv(source=my_rank+1,tag=22)
      self.layer_models.append(neighbor_model)

    self.x_final = None

    # build up the core
    self.py_core = self.initCore()
  # end __init__

  def run(self,x):
    return self.runBraid(x)
  # end forward

  def getLayer(self,t,tf,level):
    index = self.getTimeStepIndex(t,tf,level)
    return self.layer_models[index]

  def parameters(self):
    return [list(l.parameters()) for l in self.layer_models]

  def eval(self,x,tstart,tstop,level):
    dt = tstop-tstart

    with torch.no_grad(): 
      t_x = x.tensor()
      layer = self.getLayer(tstart,tstop,x.level())
      t_y = t_x+dt*layer(t_x)
      return BraidVector(t_y,x.level()) 
  # end eval
# end ForwardBraidApp

##############################################################

class BackwardBraidApp(BraidApp):

  def __init__(self,fwd_app):
    # call parent constructor
    BraidApp.__init__(self,fwd_app.getMPIData().getComm(),
                           fwd_app.local_num_steps,
                           fwd_app.Tf,
                           fwd_app.max_levels,
                           fwd_app.max_iters)

    self.fwd_app = fwd_app

    # build up the core
    self.py_core = self.initCore()

    # setup adjoint specific stuff
    self.fwd_app.setStorage(0)

    # reverse ordering for adjoint/backprop
    self.setRevertedRanks(1)
  # end __init__

  def run(self,x):
    f = self.runBraid(x)

    my_params = self.fwd_app.parameters()

    # this code is due to how braid decomposes the backwards problem
    # The ownership of the time steps is shifted to the left (and no longer balanced)
    first = 1
    if self.getMPIData().getRank()==0:
      first = 0

    # preserve the layerwise structure, to ease communication
    self.grads = [ [item.grad.clone() for item in sublist] for sublist in my_params[first:]]
    for m in self.fwd_app.layer_models:
      m.zero_grad()

    return f
  # end forward

  def eval(self,x,tstart,tstop,level):
    dt = tstop-tstart

    finegrid = 0
    primal_index = self.getPrimalIndex(tstart,tstop,level)

    # get the primal vector from the forward app
    px = self.fwd_app.getUVector(finegrid,primal_index)

    t_px = px.tensor().clone()
    t_px.requires_grad = True

    # because we need to get the layer from the forward app, this
    # transformation finds the right layer for the forward app seen
    # from the backward apps perspective
    layer = self.fwd_app.getLayer(self.Tf-tstop,self.Tf-tstart,level)

    # enables gradient calculation 
    with torch.enable_grad():
      # turn off parameter gradients below the fine grid
      #  - for posterity record their old value
      grad_list = []
      if level!=0:
        for p in layer.parameters():
          grad_list += [p.requires_grad]
          p.requires_grad = False
      else:
        # clean up parameter gradients on fine level, 
        # they are only computed once
        layer.zero_grad()

      t_py = t_px+dt*layer(t_px)

      # turn gradients back on
      if level!=0:
        for p,g in zip(layer.parameters(),grad_list):
          p.requires_grad = g
    # end with torch.enable_grad

    t_x = x.tensor()
    t_py.backward(t_x)

    return BraidVector(t_px.grad,level) 
  # end eval
# end BackwardBraidApp
