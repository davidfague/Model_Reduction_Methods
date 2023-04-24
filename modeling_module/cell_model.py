from neuron import h
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import math
from neuron import h
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import math
import os
import h5py
import csv

from stylized_module.recorder import Recorder
from modeling_module.synapses import CurrentInjection, Synapse, Listed_Synapse

class cell_model():
  '''expanded cell model class for ECP calculation
  takes hoc cell model and does bookkeeping for analysis functions
  '''
  def __init__(self,model,synapses_list=None,netcons_list=None,gen_3d=True,gen_geom_csv=False,spike_threshold: Optional[float] = None):
    self.all=model.all
    self.soma=model.soma
    self.apic=model.apic
    self.dend=model.dend
    self.axon=model.axon
    #convert nrn section lists to python lists if applicable
    self.all=self.__convert_sectionlist(sectionlist=self.all)
    self.soma=self.__convert_sectionlist(sectionlist=self.soma, return_singles=True) #if sectionlist contains only one section returns just the section instead of list of sections
    self.dend=self.__convert_sectionlist(sectionlist=self.dend)
    self.apic=self.__convert_sectionlist(sectionlist=self.apic)
    self.axon=self.__convert_sectionlist(sectionlist=self.axon)
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
    self.__generate_sec_coords()
    self.__store_segments()
    self.__set_spike_recorder()
    self.__calc_seg_coords()
    self.__store_synapses_list() #store and record synapses from the synapses_list used to initialize the cell
    self.grp_ids = []
    if gen_geom_csv==True:
      self.__generate_geometry_file()
    # self.calculate_netcons_per_seg()
    self.__insert_unused_channels()
    self.__setup_recorders()
  
  def __calc_seg_coords(self):
      """Calculate segment coordinates for ECP calculation"""
      self.seg_coords = {}
      for sec in self.all:
          nseg = sec.nseg
          pt0 = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
          for i in range(sec.n3d()-1):
              arc_length_before = sec.arc3d(i)
              arc_length_after = sec.arc3d(i+1)
              for iseg,seg in enumerate(sec):
                  if (arc_length_before/sec.L) <= seg.x <= (arc_length_after/sec.L):
                      # seg.x is between 3d coordinates i and i+1
                      seg_x_between_coordinates = (seg.x * sec.L - arc_length_before) / (arc_length_after - arc_length_before)
                      # calculate 3d coordinates at seg_x_between_coordinates
                      x_before, y_before, z_before = sec.x3d(i), sec.y3d(i), sec.z3d(i)
                      x_after, y_after, z_after = sec.x3d(i+1), sec.y3d(i+1), sec.z3d(i+1)
                      x_coord = x_before + (x_after - x_before) * seg_x_between_coordinates
                      y_coord = y_before + (y_after - y_before) * seg_x_between_coordinates
                      z_coord = z_before + (z_after - z_before) * seg_x_between_coordinates
                      pt0 = (x_before, y_before, z_before)
                      pt1 = (x_coord, y_coord, z_coord)
                      pt2 = (x_after, y_after, z_after)
                      seg_id = self.segments.index(seg)
                      if seg_id not in self.seg_coords:
                          self.seg_coords[seg_id] = {'p0': np.empty((nseg, 3)), 'p1': np.empty((nseg, 3)), 'p05': np.empty((nseg, 3)), 'r': np.empty(nseg)}
                      self.seg_coords[seg_id]['p0'][iseg, :] = np.array(pt0)
                      self.seg_coords[seg_id]['p1'][iseg, :] = np.array(pt1)
                      self.seg_coords[seg_id]['p05'][iseg, :] = (np.array(pt0) + np.array(pt1)) / 2
                      self.seg_coords[seg_id]['r'][iseg] = seg.diam / 2
      for seg_id in self.seg_coords:
          self.seg_coords[seg_id]['dl'] = self.seg_coords[seg_id]['p1'] - self.seg_coords[seg_id]['p0']
      return self.seg_coords
  
  def __calc_seg_coords__byseg(self):
      """Calculate segment coordinates for ECP calculation"""
      self.seg_coords = {}
      p0 = np.empty((self._nseg, 3))
      p1 = np.empty((self._nseg, 3))
      p05 = np.empty((self._nseg, 3))
      r = np.empty(self._nseg)
      for iseg, seg in enumerate(self.segments):
          sec = seg.sec
          nseg = sec.nseg
          pt0 = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
          for i in range(sec.n3d()-1):
              arc_length_before = sec.arc3d(i)
              arc_length_after = sec.arc3d(i+1)
              if (arc_length_before/sec.L) <= seg.x <= (arc_length_after/sec.L):
                  # seg.x is between 3d coordinates i and i+1
                  seg_x_between_coordinates = (seg.x * sec.L - arc_length_before) / (arc_length_after - arc_length_before)
                  # calculate 3d coordinates at seg_x_between_coordinates
                  x_before, y_before, z_before = sec.x3d(i), sec.y3d(i), sec.z3d(i)
                  x_after, y_after, z_after = sec.x3d(i+1), sec.y3d(i+1), sec.z3d(i+1)
                  x_coord = x_before + (x_after - x_before) * seg_x_between_coordinates
                  y_coord = y_before + (y_after - y_before) * seg_x_between_coordinates
                  z_coord = z_before + (z_after - z_before) * seg_x_between_coordinates
                  pt0 = (x_before, y_before, z_before)
                  pt1 = (x_coord, y_coord, z_coord)
                  pt2 = (x_after, y_after, z_after)
                  self.seg_coords[iseg] = {'dl': sec.L/nseg, 'pc': pt1, 'r': seg.diam/2}
      return self.seg_coords
    

  def __calc_seg_coords_orig(self):
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

  def get_spike_time(self, index: Union[np.ndarray, List[int], int, str] = 0) -> np.ndarray:
      """
      Return soma spike time of the cell by index (indices), ndarray (list of ndarray)
      Parameters
      index: index of the cell to retrieve the spikes from
      """
      if self.spike_threshold is None:
          raise ValueError("Spike recorder was not set up.")
      if type(index) is str and index == 'all':
          index = range(self.ncell)
      if not hasattr(index, '__len__'):
          spk = self.spikes.as_numpy().copy()
      else:
          index = np.asarray(index).ravel()
          spk = np.array([self.spikes.as_numpy().copy() for i in index], dtype=object)
      return spk

  def add_injection(self, sec_index, **kwargs):
        """Add current injection to a section by its index"""
        self.injection.append(CurrentInjection(self, sec_index, **kwargs))

  def add_synapse(self, stim: h.NetStim, sec_index: int, **kwargs):
        """Add synapse to a section by its index"""
        new_syn=Synapse(self, stim, sec_index, **kwargs)
        self.netcons_list.append(new_syn.nc)
        self.synapse.append(new_syn)

  def __generate_sec_coords(self):
      '''
      Note: need to improve branching so that it is random direction in a quadrant of a sphere rather than x-y plane
      takes a cell that has no n3d() coordinates and gives new coordinates
      by choosing an arbitrary direction for the subtree to move
      '''
      section_obj_list = self.all
      axial = False
      parent_sections = [] # list for already seen parent_sections of this type
      for sec in section_obj_list:
          if sec.n3d() == 0: #better to just give all sections only 2 3d coordinates
              print("Generating 3D coordinates for: ", sec)
              sec_length = sec.L
              if sec is self.soma:
                  self.sec_angs.append(0)
                  self.sec_rots.append(0)
                  if sec.nseg != 1:
                      print('Changing soma nseg from', sec.nseg, 'to 1')
                      sec.nseg = 1
                  # Set the soma's 3d coordinates so that it is a cylinder that at least approximates a sphere
                  pt0 = [0., -1 * sec.L / 2., 0.]
                  pt1 = [0., sec.L / 2., 0.]
                  sec.pt3dclear()
                  sec.pt3dadd(*pt0, sec.diam)
                  sec.pt3dadd(*pt1, sec.diam)
              else:
                  if sec.parentseg() is not None:
                      pseg = sec.parentseg()
                      psec = pseg.sec
                      if (psec in self.apic) and (psec is not self.apic[0]): # branch
                          nbranch = len(psec.children())
                      else:
                          nbranch = 1
                  else:
                      print(sec,"is attached to None")
                      psec = None # may need to provide more implementation in the case of no 3d coords and no parent section.
                      nbranch = 1

                  rot = np.random.uniform(low=0, high=2 * np.pi) # rot can be used to uniformly rotate branches if i=parent_sections.count(psec) and rot = 2 * math.pi/nbranch

                  i = 1 # i can be used to uniformly rotate the sections if rot = 2 * math.pi/nbranch and i=parent_sections.count(psec)

                  parent_sections.append(psec)

                  length = sec.L

                  fullsecname = sec.name()
                  sec_type = fullsecname.split(".")[1][:4]

                  if sec_type == "apic":
                      if sec == self.apic[0]: # trunk
                          ang = 1.570796327
                      else:
                          # ang = np.random.uniform(low=0, high=np.pi) # branches
                          ang = np.random.normal(loc=np.pi/2, scale=0.5) # could add limits to ang (if ang>val:ang=val)
                  elif sec_type == "dend":
                      # ang = -np.random.uniform(low=0, high=np.pi)
                      ang = -np.random.normal(loc=np.pi/2, scale=0.5) # could add limits to ang (if ang>val:ang=val)
                  elif sec_type == "axon":
                      ang = -1.570796327
                  else:
                      print(sec, sec_type, ' is not apic, dend or axon')
                      ang = 0

                  if axial == True:
                      x = 0
                      y = length*((ang>=0)*2-1)
                  else:
                      x = length * math.cos(ang)
                      y = length * math.sin(ang)
                  self.sec_angs.append(ang)
                  self.sec_rots.append(i*rot)
                  # find starting position using parent segment coordinates
                  for i in range(psec.n3d()-1):
                      arc_length_before = psec.arc3d(i)
                      arc_length_after = psec.arc3d(i+1)
                      if (arc_length_before/psec.L) <= pseg.x <= (arc_length_after/psec.L):
                          # pseg.x is between 3d coordinates i and i+1
                          psec_x_between_coordinates = (pseg.x * psec.L - arc_length_before) / (arc_length_after - arc_length_before)
                          # calculate 3d coordinates at psec_x_between_coordinates
                          x_before, y_before, z_before = psec.x3d(i), psec.y3d(i), psec.z3d(i)
                          x_after, y_after, z_after = psec.x3d(i+1), psec.y3d(i+1), psec.z3d(i+1)
                          x_coord = x_before + (x_after - x_before) * psec_x_between_coordinates
                          y_coord = y_before + (y_after - y_before) * psec_x_between_coordinates
                          z_coord = z_before + (z_after - z_before) * psec_x_between_coordinates
                          pt0 = (x_coord, y_coord, z_coord)
                          break
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

  def __generate_sec_coords__old(self):
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
          if sec.n3d() != 0:
            print("Generating 3D coordinates for: ",sec)
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

  def __generate_geometry_file(self):
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
    try:df['ang']=angs
    except:pass
    df['rot']=rots
    # df['axials']=axials # may need to fix
    # df['nbranch']=nbranchs # may need to fix
    self.geometry=df

  def __convert_sectionlist(self,sectionlist,return_singles=False):
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

  def __insert_unused_channels(self):
      channels = [('NaTa_t', 'gNaTa_t_NaTa_t', 'gNaTa_tbar'),
                  ('Ca_LVAst', 'ica_Ca_LVAst', 'gCa_LVAstbar'),
                  ('Ca_HVA', 'ica_Ca_HVA', 'gCa_HVAbar'),
                  ('Ih', 'ihcn_Ih', 'gIhbar')]
      for channel, attr, conductance in channels:
          for sec in self.all:
              if not hasattr(sec(0.5), attr):
                  sec.insert(channel)
                  for seg in sec:
                      setattr(getattr(seg, channel), conductance, 0)
                  # print(channel, sec) # empty sections

  def __setup_recorders(self):
      self.gNaTa_T = Recorder(obj_list=self.segments, var_name='gNaTa_t_NaTa_t')
      self.ina = Recorder(obj_list=self.segments, var_name='ina_NaTa_t')
      self.ical = Recorder(obj_list=self.segments, var_name='ica_Ca_LVAst')
      self.icah = Recorder(obj_list=self.segments, var_name='ica_Ca_HVA')
      self.ih = Recorder(obj_list=self.segments, var_name='ihcn_Ih')
      self.Vm = Recorder(obj_list=self.segments)

  def __create_output_folder(self):
      nbranches = len(self.apic)-1
      nc_count = len(self.netcons_list)
      syn_count = len(self.synapses_list)
      seg_count = len(self.segments)
      

      self.output_folder_name = (
          str(h.tstop)+
          "outputcontrol_" +
          str(nbranches) + "nbranch_" +
          str(nc_count) + "NCs_" +
          str(syn_count) + "nsyn_" +
          str(seg_count) + "nseg"
      )

      if not os.path.exists(self.output_folder_name):
          print('Outputting data to ', self.output_folder_name)
          os.makedirs(self.output_folder_name)

      return self.output_folder_name

  def get_recorder_data(self):
      '''
      Method for calculating net synaptic currents and getting data after simulation
      '''
      numTstep = int(h.tstop/h.dt)
      i_NMDA_bySeg = [[0] * (numTstep+1)] * len(self.segments)
      i_AMPA_bySeg = [[0] * (numTstep+1)] * len(self.segments)
      # i_bySeg = [[0] * (numTstep+1)] * len(self.segments)

      for synapse in self.synapses_list:
          try:
              i_NMDA = np.array(synapse.rec_vec.vec_list[1])
              i_AMPA = np.array(synapse.rec_vec.vec_list[0])
              seg = synapse.get_segment_id()

              try:
                  i_NMDA_bySeg[seg] = i_NMDA_bySeg[seg] + i_NMDA
                  i_AMPA_bySeg[seg] = i_AMPA_bySeg[seg] + i_AMPA
              except:
                  pass
          except:
              continue

      i_NMDA_df = pd.DataFrame(i_NMDA_bySeg) * 1000
      i_AMPA_df = pd.DataFrame(i_AMPA_bySeg) * 1000
      

      self.data_dict = {}
      self.data_dict['spikes']=self.get_spike_time()
      self.data_dict['ih_data'] = self.ih.as_numpy()
      self.data_dict['gNaTa_T_data'] = self.gNaTa_T.as_numpy()
      self.data_dict['ina_data'] = self.ina.as_numpy()
      self.data_dict['icah_data'] = self.icah.as_numpy()
      self.data_dict['ical_data'] = self.ical.as_numpy()
      self.data_dict['Vm'] = self.Vm.as_numpy()
      self.data_dict['i_NMDA'] = i_NMDA_df
      self.data_dict['i_AMPA'] = i_AMPA_df
      # self.data_dict['i'] = i_bySeg
      self.__create_output_files(self.__create_output_folder())

      return self.data_dict

  def __create_output_files(self,output_folder_name):
      for name, data in self.data_dict.items():
        try:
          self.__report_data(f"{output_folder_name}/{name}_report.h5", data.T)
        except:
          self.__report_data(f"{output_folder_name}/{name}_report.h5", data)

  def __report_data(self,reportname, dataname):
      try:
          os.remove(reportname)
      except FileNotFoundError:
          pass

      with h5py.File(reportname, 'w') as f:
          f.create_dataset("report/biophysical/data", data=dataname)

  def plot_seg_heatmap(self, seg_df, color_column, subtype=None):
      '''
      Plots a heatmap of a segment dataframe, using a specified column for color
      color_column  :   attribute that is per segment
      Can update segments dataframe so that is instead a segment class with the option of saving the dataframe when initializeing the cell?
      '''
      #may be able to replace seg_df with seg_info or self or something
      if isinstance(getattr(seg_df, color_column), list):
          color_data = np.concatenate(getattr(seg_df, color_column))
      else:
          color_data = getattr(seg_df, color_column)
          
      if isinstance(color_data, dict):
          if sub_type is not None:
              color_data = [v.get(sub_type) for v in color_data.values()]
          else:
              raise ValueError("If color_column is a dictionary, sub_type parameter must be specified")
              

      label = color_column.capitalize()
      savename = color_column.lower()

      plt.figure(figsize=(4,10))
      ax = plt.scatter(seg_df["X Coord"], seg_df["Y Coord"],c = color_data, cmap='jet',)
      plt.vlines(110,400,500)
      plt.text(0,450,'100 um')
      plt.hlines(400,110,210)
      plt.text(110,350,'100 um')
      plt.xticks([])
      plt.yticks([])
      cbar = plt.colorbar()
      cbar.ax.set_ylabel(label, rotation=270)

      plt.box(False)
      plt.show()
      plt.savefig("/"+self.output_folder_name+savename+'.svg')

  # def plot_seg_heatmap(self, seg_df, color_column):
  #     '''
  #     Plots a heatmap of a segment dataframe, using a specified column for color
  #     color_column  :   attribute that is per segment
  #     Can update segments dataframe so that is instead a segment class with the option of saving the dataframe when initializeing the cell?
  #     '''
  #     if isinstance(getattr(self, color_column), list):
  #         color_data = np.concatenate(getattr(self, color_column))
  #     else:
  #         color_data = getattr(self, color_column)

  #     label = color_column.capitalize()
  #     savename = color_column.lower()

  #     plt.figure(figsize=(4,10))
  #     ax = plt.scatter(seg_df["X Coord"], seg_df["Y Coord"],c = color_data, cmap='jet',)
  #     plt.vlines(110,400,500)
  #     plt.text(0,450,'100 um')
  #     plt.hlines(400,110,210)
  #     plt.text(110,350,'100 um')
  #     plt.xticks([])
  #     plt.yticks([])
  #     cbar = plt.colorbar()
  #     cbar.ax.set_ylabel(label, rotation=270)

  #     plt.box(False)
  #     plt.savefig("/"+self.output_folder_name+savename+'.svg')

  def plot_temporal_spatial_heatmap(seg_df, color_column):
      '''
      NEED TO UPDATE
      Plots a temporal-spatial heatmap of a segment dataframe, using a specified column for color
      '''
      if isinstance(seg_df[color_column].iloc[0], list):
          color_data = np.concatenate(seg_df[color_column])
      else:
          color_data = seg_df[color_column]

      label = color_column.capitalize()
      savename = color_column.lower()

      plt.figure(figsize=(10, 4))
      ax = plt.scatter(seg_df["Time (ms)"], seg_df["Distance from Soma (um)"], c=color_data, cmap='jet', alpha=0.5)
      plt.vlines(110, 400, 500)
      plt.text(0, 450, '100 um')
      plt.hlines(400, 110, 210)
      plt.text(110, 350, '100 um')
      plt.xticks([])
      plt.yticks([])
      cbar = plt.colorbar()
      cbar.ax.set_ylabel(label, rotation=270)

      plt.box(False)
      plt.xlabel('Time (ms)')
      plt.ylabel('Distance from Soma (um)')
      plt.title(label)
      plt.show()

  def write_seg_info_to_csv(self):
      seg_info=self.__get_segment_info__()
      with open(self.output_folder_name+'/seg_info.csv', mode='w') as file:
          writer = csv.DictWriter(file, fieldnames=seg_info[0].keys())
          writer.writeheader()
          for row in seg_info:
              writer.writerow(row)

  def __get_segment_info__(self):      
      seg_info = []
      k = 0
      j = 0
      for sec in self.all:
          sec_type = sec.name().split('.')[1][:4]
          for i, seg in enumerate(sec):
              seg_info.append({ #update to have consistent naming scheme (will then need to debug plotting functions too, but should be easy)
                  'seg': seg,
                  'seg_id': j,
                  'X Coord': self.seg_coords['pc'][i][0],
                  'Y Coord': self.seg_coords['pc'][i][1],
                  'Z Coord': self.seg_coords['pc'][i][2],
                  'seg diam': seg.diam,
                  'bmtk_id': k,
                  'x': seg.x,
                  'Type': sec_type,
                  'Sec ID': int(sec.name().split('[')[2].split(']')[0]),
                  'sec diam': sec.diam,
                  'nseg': seg.sec.nseg,
                  'Ra': seg.sec.Ra,
                  'seg_L': sec.L/sec.nseg,
              })
              j += 1
          k += 1
      return self.__get_parent_segment_ids(seg_info)

  def __get_parent_segment_ids(self, seg_info):
      for seg in seg_info:
          seg['parent_seg_id'] = None
      pseg_ids = []
      for i, seg in enumerate(seg_info):
          idx = int(np.floor(seg['x'] * seg['nseg']))
          if idx != 0:
              pseg_id = i-1
          else:
              pseg = seg['seg'].sec.parentseg()
              if pseg is None:
                  pseg_id = None
              else:
                  psec = pseg.sec
                  nseg = psec.nseg
                  pidx = int(np.floor(pseg.x * nseg))
                  if pseg.x == 1.:
                      pidx -= 1
                  try:
                      pseg_id = next(idx for idx, info in enumerate(seg_info) if info['seg'] == psec((pidx + .5) / nseg))
                  except StopIteration:
                      pseg_id = "Segment not in segments"
              seg_info[i]['parent_seg_id'] = pseg_id
          # pseg_ids.append(pseg_id)
      return self.__get_segment_elec_dist(seg_info)

  def __get_segment_elec_dist(self, seg_info):
      for seg in seg_info:
          seg['seg_elec_info'] = {}
      freqs = {'delta': 1, 'theta': 4, 'alpha': 8, 'beta': 12, 'gamma': 30}

      soma_passive_imp = h.Impedance()
      soma_active_imp = h.Impedance()
      nexus_passive_imp = h.Impedance()
      nexus_active_imp = h.Impedance()
      try:
          soma_passive_imp.loc(self.hobj.soma[0](0.5))
          soma_active_imp.loc(self.hobj.soma[0](0.5))
      except:
          try:
              soma_passive_imp.loc(self.soma[0](0.5))
              soma_active_imp.loc(self.soma[0](0.5))
          except:
              try:
                  soma_passive_imp.loc(self.soma(0.5))
                  soma_active_imp.loc(self.soma(0.5))
              except:
                  raise AttributeError("Could not locate soma for impedance calculation")
      try:
          nexus_passive_imp.loc(self.hobj.apic[0](0.99))
          nexus_active_imp.loc(self.hobj.apic[0](0.99))
      except:
          try:
              nexus_passive_imp.loc(self.apic[0](0.99))
              nexus_active_imp.loc(self.apic[0](0.99))
          except:
              try:
                  nexus_passive_imp.loc(self.apic(0.99))
                  nexus_active_imp.loc(self.apic(0.99))
              except:
                  raise AttributeError("Could not locate the nexus for impedance calculation")

      for freq_name, freq_hz in freqs.items():
          soma_passive_imp.compute(freq_hz + 1 / 9e9, 0) #passive from soma
          soma_active_imp.compute(freq_hz + 1 / 9e9, 1) #active from soma
          nexus_passive_imp.compute(freq_hz + 1 / 9e9, 0) #passive from nexus
          nexus_active_imp.compute(freq_hz + 1 / 9e9, 1) #active from nexus
          for i, seg in enumerate(seg_info):
              elec_dist_info = {
                  'active_soma': soma_active_imp.ratio(seg['x']),
                  'active_nexus': nexus_active_imp.ratio(seg['x']),
                  'passive_soma': soma_passive_imp.ratio(seg['x']),
                  'passive_nexus': nexus_passive_imp.ratio(seg['x'])
              }
              seg_info[i]['seg_elec_info'][freq_name] = elec_dist_info
      return self.__calculate_netcons_per_seg(seg_info)

  def __calculate_netcons_per_seg(self, seg_info):
      NetCon_per_seg = [0] * len(seg_info)
      inh_NetCon_per_seg = [0] * len(seg_info)
      exc_NetCon_per_seg = [0] * len(seg_info)

      v_rest = -60 #used to determine exc/inh may adjust or automate
      
      # calculate number of synapses for each segment (may want to divide by segment length afterward to get synpatic density)
      for netcon in self.netcons_list:
          syn = netcon.syn()
          if syn in self.synapses_list:
              syn_seg_id = seg_info.index(next((s for s in seg_info if s['seg'] == syn.get_segment()), None))
              seg_dict = seg_info[syn_seg_id]
              if syn in seg_dict['seg'].point_processes():
                  NetCon_per_seg[syn_seg_id] += 1 # get synapses per segment
                  if syn.e > v_rest:
                      exc_NetCon_per_seg[syn_seg_id] += 1
                  else:
                      inh_NetCon_per_seg[syn_seg_id] += 1
              else:
                  print("Warning: synapse not in designated segment's point processes")
          else:
              print("Warning: potentially deleted synapse:","|NetCon obj:",netcon,"|Synapse obj:",syn,"the NetCon's synapse is not in synapses_list. Check corresponding original cell's NetCon for location, etc.")
      
      for i, seg in enumerate(seg_info):
          seg['netcons_per_seg'] = {
              'exc': exc_NetCon_per_seg[i],
              'inh': inh_NetCon_per_seg[i],
              'total': NetCon_per_seg[i]
          }
          seg['netcon_density_per_seg'] = {
              'exc': exc_NetCon_per_seg[i]/seg['seg_L'],
              'inh': inh_NetCon_per_seg[i]/seg['seg_L'],
              'total': NetCon_per_seg[i]/seg['seg_L']
          }
      
      return seg_info
