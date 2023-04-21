from neuron import h
from typing import Optional, Union, List
from modeling_module.synapses import CurrentInjection, Synapse, Listed_Synapse
import numpy as np
import pandas as pd
import math
class cell_model():
  '''expanded cell model class for ECP calculation
  takes hoc cell model and does bookkeeping for analysis functions
  '''
  def __init__(self,model,synapses_list=None,netcons_list=None,gen_3d=True,spike_threshold: Optional[float] = None):
    self.all=model.all
    self.soma=model.soma
    self.apic=model.apic
    self.dend=model.dend
    self.axon=model.axon
    #convert nrn section lists to python lists if applicable
    self.all=self.convert_sectionlist(sectionlist=self.all)
    self.soma=self.convert_sectionlist(sectionlist=self.soma, return_singles=True) #if sectionlist contains only one section returns just the section instead of list of sections
    self.dend=self.convert_sectionlist(sectionlist=self.dend)
    self.apic=self.convert_sectionlist(sectionlist=self.apic)
    self.axon=self.convert_sectionlist(sectionlist=self.axon)
    self.spike_threshold = spike_threshold
    self.synapses_list=synapses_list #list of synapse objects from model reduction
    self.netcons_list=netcons_list # list of netcon objects from model reduction
    self.segments=[] # list for nrn segment objects
    self.injection=[] # list of injection objects
    self.synapse=[] # list of python synapse class objects
    self.sec_id_lookup = {}  # dictionary from section type id to section index
    self.sec_id_in_seg = []  # index of the first segment of each section in the segment list
    self.sec_angs = [] # list of angles that were used to branch the cell
    self.sec_rots = []
    if gen_3d==True:
      self.generate_sec_coords()
    self.__store_segments()
    self.__set_spike_recorder()
    if self.apic[2].n3d()==2: #calc_seg_coords only works for cell with sections that only have two 3d coordinates. #detailed cell has too many 3d coordinates
      self.__calc_seg_coords()
    self.__store_synapses_list() #store and record synapses from the synapses_list used to initialize the cell
    self.grp_ids = []
    self.generate_geometry_file()
    self.calculate_netcons_per_seg()

  def __calc_seg_coords(self):
    """Calculate segment coordinates for ECP calculation"""
    p0 = np.empty((self._nseg, 3))
    p1 = np.empty((self._nseg, 3))
    p05 = np.empty((self._nseg, 3))
    r = np.empty(self._nseg)
    for isec, sec in enumerate(self.all):
        iseg = self.sec_id_in_seg[isec]
        nseg = sec.nseg
        pt0 = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        pt1 = np.array([sec.x3d(1), sec.y3d(1), sec.z3d(1)])
        pts = np.linspace(pt0, pt1, 2 * nseg + 1)
        p0[iseg:iseg + nseg, :] = pts[:-2:2, :]
        p1[iseg:iseg + nseg, :] = pts[2::2, :]
        p05[iseg:iseg + nseg, :] = pts[1:-1:2, :]
        r[iseg:iseg + nseg] = sec.diam / 2
    self.seg_coords = {'dl': p1 - p0, 'pc': p05, 'r': r}

  def __store_segments(self):
    self.segments = []
    self.sec_id_in_seg = []
    nseg = 0
    for sec in self.all:
        self.sec_id_in_seg.append(nseg)
        nseg += sec.nseg
        for seg in sec:
            self.segments.append(seg)
            self.__store_point_processes(seg)
    self._nseg = nseg

  def __store_point_processes(self,seg):
    for pp in seg.point_processes():
        self.injection.append(pp)

  def __set_spike_recorder(self, threshold: Optional = None):
      if threshold is not None:
          self.spike_threshold = threshold
      if self.spike_threshold is None:
          self.spikes = None
      else:
          vec = h.Vector()
          nc = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
          nc.threshold = self.spike_threshold
          nc.record(vec)
          self.spikes = vec

  def add_injection(self, sec_index, **kwargs):
        """Add current injection to a section by its index"""
        self.injection.append(CurrentInjection(self, sec_index, **kwargs))

  def add_synapse(self, stim: h.NetStim, sec_index: int, **kwargs):
        """Add synapse to a section by its index"""
        new_syn=Synapse(self, stim, sec_index, **kwargs)
        self.netcons_list.append(new_syn.nc)
        self.synapse.append(new_syn)

  def generate_sec_coords(self):
      '''
      Note: need to improve branching so that it is random direction in a quadrant of a sphere rather than x-y plane
      takes a cell that has no n3d() coordinates and gives new coordinates
      by choosing an arbitrary direction for the subtree to move
      '''
      section_obj_list= self.all
      # print(section_obj_list)
      axial=False
      parent_sections=[] #list for already seen parent_sections of this type
      for sec in section_obj_list:
        sec_length=sec.L
        if sec is self.soma:
          self.sec_angs.append(0)
          self.sec_rots.append(0)
          # pt0 = [0., -1 * sec.diam, 0.] #does not seem to preserve soma shape , but need to make sure soma children begin at correct 3d coordinate.
          # pt1 = [0., 0., 0.]
          # sec.pt3dclear()
          # sec.pt3dadd(*pt0, sec.diam)
          # sec.pt3dadd(*pt1, sec.diam)
          if sec.nseg != 1:
            print('Changing soma nseg from',sec.nseg,'to 1')
            sec.nseg = 1
        else:
          if sec.parentseg() is not None:
            psec=sec.parentseg().sec
            if (psec in self.apic) and (psec is not self.apic[0]): # branch
              # print('branch')
              nbranch = len(psec.children())
            else:
              nbranch=1
          else:
            psec=None # may need to provide more implementation in the case of no 3d coords and no parent section.
            nbranch=1

          # rot = 2 * math.pi/nbranch
          rot=np.random.uniform(low=0,high=2*np.pi)# rot can be used to uniformly rotate branches if i=parent_sections.count(psec) and rot = 2 * math.pi/nbranch

          i=1 #i can be used to uniformly rotate the sections if rot = 2 * math.pi/nbranch and i=parent_sections.count(psec)
          # if nbranch==1:
          #   i=1
          # else:
          #   i=parent_sections.count(psec)

          parent_sections.append(psec)
          # print("sec: ",sec, "|nbranch: ",nbranch,"|i: ,",i,"|parent_sections:",parent_sections)
          length=sec.L
          diameter=sec.diam
          fullsecname = sec.name()
          # print(fullsecname)
          sec_type = fullsecname.split(".")[1][:4]
          # print(sec_type)
          if sec_type == "apic":
            if sec==self.apic[0]: # trunk
              ang=1.570796327
            else:
              # ang=np.random.uniform(low=0,high=np.pi) #branches
              ang=np.random.normal(loc=np.pi/2,scale=0.5) # could add limits to ang (if ang>val:ang=val)
          elif sec_type=="dend":
            # ang=-np.random.uniform(low=0,high=np.pi)
            ang=-np.random.normal(loc=np.pi/2,scale=0.5) # could add limits to ang (if ang>val:ang=val)
          elif sec_type=="axon":
            ang=-1.570796327
          else:
            print(sec,sec_type,' is not apic, dend or axon')
            ang=0
          if axial == True:
            x = 0
            y = length*((ang>=0)*2-1)
          else:
            x = length * math.cos(ang)
            y = length * math.sin(ang)
          self.sec_angs.append(ang)
          self.sec_rots.append(i*rot)
          #find starting position #need to update to use parent segment coordinates instead of using first section coordinate
          pt0 = [psec.x3d(1), psec.y3d(1), psec.z3d(1)]
          pt1 = [0., 0., 0.]
          pt1[1] = pt0[1] + y
          pt1[0] = pt0[0] + x * math.cos(i * rot)
          pt1[2] = pt0[2] + x * math.sin(i * rot)
          # print(sec,i*rot)
          sec.pt3dclear()
          sec.pt3dadd(*pt0, sec.diam)
          sec.pt3dadd(*pt1, sec.diam)
        if int(sec.L) != int(sec_length):
          print('Error: generation of 3D coordinates resulted in change of section length for',sec,'from',sec_length,'to',sec.L)

  def generate_geometry_file(self):
    '''
    generates geometry file specifying name, pid, ang, radius, length, type
    work in progress
    '''
    df = pd.DataFrame()
    ids=[]
    names=[]
    types=[]
    pids=[]
    axials=[]
    nbranchs=[]
    Ls=[]
    Rs=[]
    angs=self.sec_angs
    rots=self.sec_rots
    for sec in self.all:
      # print(dir(sec))
      name=sec.name()
      # print(name)
      names.append(name)
      ids.append(names.index(name))
      _,sec_type_withinteger=name.split('.')
      sec_type,_=sec_type_withinteger.split('[')
      types.append(sec_type)
      pseg = sec.parentseg()
      if pseg == None:
        pids.append(None)
      else:
        psec=pseg.sec
        px3d=psec.x3d
        pids.append(int(names.index(psec.name())))
      # axials.append('TRUE')
      # nbranchs.append(1)
      Ls.append(sec.L)
      # print(dir(sec))
      Rs.append(sec.diam/2)
    df['id']=ids
    df['name']=names
    df['pid']=pids
    df['type']=types
    df['L']=Ls
    df['R']=Rs
    df['ang']=angs
    df['rot']=rots
    # df['axials']=axials # may need to fix
    # df['nbranch']=nbranchs # may need to fix
    self.geometry=df

  def convert_sectionlist(self,sectionlist,return_singles=False):
    '''
    convert nrn sectionlist objects to python list
    return_singles set to true will return section instead of [section] for lists with only one section
    '''
    new_sectionlist=[]
    if str(type(sectionlist)) == "<class 'hoc.HocObject'>":
      for sec in sectionlist:
        new_sectionlist.append(sec)
    else:
      new_sectionlist=sectionlist
    if return_singles==True:
      if str(type(new_sectionlist))!="<class 'nrn.Section'>":
        if len(new_sectionlist)==1:
          new_sectionlist=new_sectionlist[0]
    return new_sectionlist

  def __store_synapses_list(self):
    '''
    store and record synapses from the list from model reduction algorithm
    '''
    temp_list=[] # generate temp list that has each netcon's synapse obj
    for netcon in self.netcons_list:
      syn=netcon.syn()
      if syn in self.synapses_list:
        syn_seg_id=self.segments.index(netcon.syn().get_segment())
        if syn in self.segments[syn_seg_id].point_processes():
          temp_list.append(syn)
        else:
          temp_list.append(None)
          print("Warning: synapse not in designated segment's point processes")

      else:
        temp_list.append(None)
        print("Warning: potentially deleted synapse:","|NetCon obj:",netcon,"|Synapse obj:",syn,"the NetCon's synapse is not in synapses_list. Check corresponding original cell's NetCon for location, etc.")
    # now use temp list to assign each synapse its netcons
    for synapse in self.synapses_list:
      synapse_netcons=[]
      if synapse in temp_list:
        num_netcons=temp_list.count(synapse)
        START=0
        for i in range(num_netcons):
          netcon_id=temp_list.index(synapse,START) #get all the netcon indices that are pointed toward this synapse # use np.where() instead of index() to return multiple indices.
          START=netcon_id+1
          synapse_netcons.append(self.netcons_list[netcon_id])
        self.synapse.append(Listed_Synapse(synapse,synapse_netcons)) #record synapse and add to the list
      else:
        print('Warning: ', synapse, 'does not have any netcons pointing at it. if synapse is None then deleted synapse may be stored in synapses_list')

def calculate_netcons_per_seg(self):
  self.NetCon_per_seg=[0]*len(self.segments)
  self.inh_NetCon_per_seg=[0]*len(self.segments)
  self.exc_NetCon_per_seg=[0]*len(self.segments)

  #calculate number of synapses for each segment (may want to divide by segment length afterward to get synpatic density)
  for netcon in self.netcons_list:
    syn=netcon.syn()
    if syn in synapses_list:
      syn_seg_id=self.segments.index(netcon.syn().get_segment())
      if syn in self.segments[syn_seg_id].point_processes():
        self.NetCon_per_seg[syn_seg_id]+=1 # get synapses per segment
        # NetCon_per_seg[syn_seg_id].append(netcon) # possible implementation if needing objects per segment
        if syn.e > v_rest:
          self.exc_NetCon_per_seg[syn_seg_id]+=1
          # exc_NetCon_per_seg[syn_seg_id].append(netcon)# possible implementation if needing objects per segment
        else:
          self.inh_NetCon_per_seg[syn_seg_id]+=1
          # inh_NetCon_per_seg[syn_seg_id].append(netcon)# possible implementation if needing objects per segment
      else:
        print("Warning: synapse not in designated segment's point processes")

    else:
      print("Warning: potentially deleted synapse:","|NetCon obj:",netcon,"|Synapse obj:",syn,"the NetCon's synapse is not in synapses_list. Check corresponding original cell's NetCon for location, etc.")
    
