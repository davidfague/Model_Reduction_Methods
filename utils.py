import math
import pandas as pd
import matplotlib.pyplot as plt
from neuron import h,gui
import numpy as np
import h5py
import os

def generate_stylized_geometry(cell,savename):
  df = pd.DataFrame()
  ids=[]
  names=[]
  types=[]
  pids=[]
  axials=[]
  nbranchs=[]
  Ls=[]
  Rs=[]
  angs=[]

  id=0
  for sec in cell.all:
    # print(dir(sec))
    name=sec.name()
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
      pids.append(names.index(psec.name()))
    axials.append('TRUE')
    nbranchs.append(1)
    Ls.append(sec.L)
    # print(dir(sec))
    Rs.append(sec.diam/2) # may need to compute single radius for each section
    if sec_type == "apic":
      ang=np.random.normal(scale=0.1,loc=1.570796327)
    elif sec_type=="dend":
      ang=np.random.uniform(low=0,high=-np.pi)
    else:
      ang=0
    angs.append(ang)

  # print(len(ids))
  # print(len(names))
  # print(len(types))
  # print(len(pids))
  # print(len(axials))
  # print(len(nbranchs))
  # print(len(Ls))
  # print(len(Rs))
  # print(angs)
  df["id"] = ids
  df["name"] = names
  df["pid"] = pids
  df["axial"] = axials
  df["type"] = types
  df["nbranch"] = nbranchs
  df["L"] = Ls
  df["R"] = Rs
  df["ang"] = angs
  df.to_csv(savename)

def returnSegmentValues(section):
    ##
    
    #Section naming
    #Set impedance measurement location and frequency
    name=section.name()
    # Get section 3d coordinates and put in numpy array
    n3d = section.n3d()
    x3d = np.empty(n3d)
    y3d = np.empty(n3d)
    z3d = np.empty(n3d)
    L = np.empty(n3d)
    for i in range(n3d):
        x3d[i]=section.x3d(i)
        y3d[i]=section.y3d(i)
        z3d[i]=section.z3d(i)

    # Compute length of each 3d segment
    for i in range(n3d):
        if i==0:
            L[i]=0
        else:
            L[i]=np.sqrt((x3d[i]-x3d[i-1])**2 + (y3d[i]-y3d[i-1])**2 + (z3d[i]-z3d[i-1])**2)

    # Get cumulative length of 3d segments
    cumLength = np.cumsum(L)
    N = section.nseg
    # Now upsample coordinates to segment locations
    xCoord = np.empty(N)
    yCoord = np.empty(N)
    zCoord = np.empty(N)


    if N > 1:
      dx = section.L / (N-1)
    else:
      dx = section.L

    #print(str(section))
    if "axon" in name:
      xCoord=[np.NaN]
      yCoord=[np.NaN]
      zCoord=[np.NaN]
    else:
      for n in range(N):

        if n==(N-1):
            xCoord[n]=x3d[-1]
            yCoord[n]=y3d[-1]
            zCoord[n]=z3d[-1]
        else:
            cIdxStart = np.where(n*dx >= cumLength)[0][-1] # which idx of 3d segments are we starting at
            cDistFrom3dStart = n*dx - cumLength[cIdxStart] # how far along that segment is this upsampled coordinate
            cFraction3dLength = cDistFrom3dStart / L[cIdxStart+1] # what's the fractional distance along this 3d segment
            # compute x and y positions
            xCoord[n] = x3d[cIdxStart] + cFraction3dLength*(x3d[cIdxStart+1] - x3d[cIdxStart])
            yCoord[n] = y3d[cIdxStart] + cFraction3dLength*(y3d[cIdxStart+1] - y3d[cIdxStart])
            zCoord[n] = z3d[cIdxStart] + cFraction3dLength*(z3d[cIdxStart+1] - z3d[cIdxStart])
              
    return xCoord, yCoord, zCoord

def make_seg_df(cell,savename="Segments.csv"):
    frequency=0
    # seg_locs = cell.morphology.seg_coords['p05']
    # px = seg_locs[0]
    # py = seg_locs[1]
    # pz = seg_locs[2]
    df = pd.DataFrame()
    k = 0
    j = 0
    lens = []
    seg_lens = []
    diams = []
    segdiams = []
    bmtk_ids = []
    seg_ids = []
    sec_ids = []
    full_names = []
    xs = []
    parts = []
    distances = []
    try:h.distance(sec=cell.hobj.soma[0])
    except:h.distance(sec=cell.soma[0])
    nsegs=[]
    RAs=[]
    Parentx=[]
    passive_elec_distances = []
    active_elec_distances = []
    passive = h.Impedance()
    try:passive.loc(cell.hobj.soma[0](0.5))
    except:passive.loc(cell.soma[0](0.5))
    passive.compute(frequency + 1 / 9e9, 0)
    active = h.Impedance()
    try:active.loc(cell.hobj.soma[0](0.5))
    except:active.loc(cell.soma[0](0.5))
    active.compute(frequency + 1 / 9e9, 1)
#############################################################
    
    psegids=[]
    segments=[]
    sections=[]
    all_seg_x_coords=[]
    all_seg_y_coords=[]
    all_seg_z_coords=[]
    section_names_by_seg=[]
#    segments=cell.hobj.segments
#    for i in range(len(cell.hobj.segments)):
#      pseg = segments[i].sec.parentseg()
#      psec=pseg.sec
#      nseg = psec.nseg
#      idx = int(np.floor(pseg.x * nseg))
#      if pseg.x == 1.:
#        idx -= 1
#      psegid=segments.index(psec((i + .5) / nseg))
#      psegids.append(psegid)
################################################################
    try: section_obj_list=cell.hobj.all
    except: section_obj_list=cell.all
    for sec in section_obj_list:
        xCoords,yCoords,zCoords=returnSegmentValues(sec) # get 3d coordinates for this section #may not be accurate
        for i,seg in enumerate(sec):
            try:
              all_seg_x_coords.append(cell.seg_coords['pc'][j][0])
              all_seg_y_coords.append(cell.seg_coords['pc'][j][1])
              all_seg_z_coords.append(cell.seg_coords['pc'][j][2])
             except:
              print('Exception made. Calculating Seg Coords')
              all_seg_x_coords.append(zCoords[i])
              all_seg_y_coords.append(yCoords[i])
              all_seg_z_coords.append(zCoords[i])
            lens.append(seg.sec.L)
            seg_lens.append(seg.sec.L/seg.sec.nseg)
            diams.append(seg.sec.diam)
            segdiams.append(seg.diam)
            distances.append(h.distance(seg))
            bmtk_ids.append(k)
            seg_ids.append(j)
            xs.append(seg.x)
            fullsecname = sec.name()
            sec_ids.append(int(fullsecname.split("[")[2].split("]")[0]))
            sec_type = fullsecname.split(".")[1][:4]
            nsegs.append(seg.sec.nseg)
            RAs.append(seg.sec.Ra)
            parts.append(sec_type)
            full_names.append(str(seg))
            passive_elec_distances.append(passive.ratio(seg))
            active_elec_distances.append(active.ratio(seg))
            section_names_by_seg.append(fullsecname)
            #if seg.sec.parentseg() is not None:
              #get parentseg coordinates or something to identify parentseg by
            j += 1
            segments.append(seg)
        k+=1
        sections.append(sec)
    #print(segments)
    for i in range(len(segments)): #calculate parentseg id using seg index on section
      idx = int(np.floor(segments[i].x * segments[i].sec.nseg)) #get seg index on section
      #case where segment is not first segment on section:
      if idx != 0: #if the segment is not the first on the section then the parent segment is the previous segment index
        psegid=i-1 #set parent segment id to the previous segment index
        psegids.append(psegid)
      #case where segment is first segment on section:
      else:
        pseg = segments[i].sec.parentseg()
        if pseg == None:
          psegids.append(None)
        else:
          psec=pseg.sec
          nseg = psec.nseg
          pidx = int(np.floor(pseg.x * nseg)) #get parent seg index on section
          if pseg.x == 1.:
            pidx -= 1
          psegid=segments.index(psec((pidx + .5) / nseg)) #find the segment id of the parent seg by comparing with segment list after calculating the parent seg's section(x)
          psegids.append(psegid)


    df["segmentID"] = seg_ids
    df["BMTK ID"] = bmtk_ids
    df["Seg_L"] = seg_lens
    df["Seg_diam"] = segdiams
    df["X"] = xs
    df["Type"] = parts
    df["Sec ID"] = sec_ids
    df["Distance"] = distances
    df["Section_L"] = lens
    df["Section_diam"] = diams
    df["Section_nseg"] = nsegs
    df["Section_Ra"] = RAs
    df["Coord X"] = all_seg_x_coords
    df["Coord Y"] = all_seg_y_coords
    df["Coord Z"] = all_seg_z_coords
    df["ParentSegID"] = psegids
    df["Passive ElecDist"] = passive_elec_distances
    df["Active ElecDist"] = active_elec_distances
    df["Section Name"] = section_names_by_seg


    df.to_csv(savename, index=False)

def make_reduced_seg_df(cell,savename="Segments.csv"):
    frequency=0
    # seg_locs = cell.morphology.seg_coords['p05']
    # px = seg_locs[0]
    # py = seg_locs[1]
    # pz = seg_locs[2]
    df = pd.DataFrame()
    k = 0
    j = 0
    lens = []
    seg_lens = []
    diams = []
    segdiams = []
    bmtk_ids = []
    seg_ids = []
    sec_ids = []
    full_names = []
    xs = []
    parts = []
    distances = []
    try:h.distance(sec=cell.soma)
    except:h.distance(sec=cell.soma[0])
    nsegs=[]
    RAs=[]
    Parentx=[]
    passive_elec_distances = []
    active_elec_distances = []
    passive = h.Impedance()
    try:passive.loc(cell.hobj.soma[0](0.5))
    except: 
      try: passive.loc(cell.soma[0](0.5))
      except: passive.loc(cell.soma(0.5))
    passive.compute(frequency + 1 / 9e9, 0)
    active = h.Impedance()
    try:active.loc(cell.hobj.soma[0](0.5))
    except: 
      try: active.loc(cell.soma[0](0.5))
      except: active.loc(cell.soma(0.5))
    active.compute(frequency + 1 / 9e9, 1)
#############################################################
    
    psegids=[]
    segments=[]
    sections=[]
    all_seg_x_coords=[]
    all_seg_y_coords=[]
    all_seg_z_coords=[]
    section_names_by_seg=[]
#    segments=cell.hobj.segments
#    for i in range(len(cell.hobj.segments)):
#      pseg = segments[i].sec.parentseg()
#      psec=pseg.sec
#      nseg = psec.nseg
#      idx = int(np.floor(pseg.x * nseg))
#      if pseg.x == 1.:
#        idx -= 1
#      psegid=segments.index(psec((i + .5) / nseg))
#      psegids.append(psegid)
################################################################
    try: section_obj_list=cell.all
    except: section_obj_list=cell.hoc_model.all
    for sec in section_obj_list:
        xCoords,yCoords,zCoords=returnSegmentValues(sec) # get 3d coordinates for this section
        for i,seg in enumerate(sec):
            all_seg_x_coords.append(xCoords[i])
            all_seg_y_coords.append(yCoords[i])
            all_seg_z_coords.append(zCoords[i])
            lens.append(seg.sec.L)
            seg_lens.append(seg.sec.L/seg.sec.nseg)
            diams.append(seg.sec.diam)
            segdiams.append(seg.diam)
            distances.append(h.distance(seg))
            bmtk_ids.append(k)
            seg_ids.append(j)
            xs.append(seg.x)
            fullsecname = sec.name()
            sec_ids.append(int(fullsecname.split("[")[2].split("]")[0]))
            sec_type = fullsecname.split(".")[1][:4]
            nsegs.append(seg.sec.nseg)
            RAs.append(seg.sec.Ra)
            parts.append(sec_type) # section types
            full_names.append(str(seg))
            passive_elec_distances.append(passive.ratio(seg))
            active_elec_distances.append(active.ratio(seg))
            section_names_by_seg.append(fullsecname)
            #if seg.sec.parentseg() is not None:
              #get parentseg coordinates or something to identify parentseg by
            j += 1
            segments.append(seg)
        k+=1
        sections.append(sec)
    #print(segments)
    for i in range(len(segments)): #calculate parentseg id using seg index on section
      idx = int(np.floor(segments[i].x * segments[i].sec.nseg)) #get seg index on section
      #case where segment is not first segment on section:
      if idx != 0: #if the segment is not the first on the section then the parent segment is the previous segment index
        psegid=i-1 #set parent segment id to the previous segment index
        psegids.append(psegid)
      #case where segment is first segment on section:
      else:
        pseg = segments[i].sec.parentseg()
        if pseg == None:
          psegids.append(None)
        else:
          psec=pseg.sec
          nseg = psec.nseg
          pidx = int(np.floor(pseg.x * nseg)) #get parent seg index on section
          if pseg.x == 1.:
            pidx -= 1
          try:psegid=segments.index(psec((pidx + .5) / nseg)) #find the segment id of the parent seg by comparing with segment list after calculating the parent seg's section(x)
          except: psegid="Segment not in segments"
          psegids.append(psegid)


    df["segmentID"] = seg_ids
    df["BMTK ID"] = bmtk_ids
    df["Seg_L"] = seg_lens
    df["Seg_diam"] = segdiams
    df["X"] = xs
    df["Type"] = parts
    df["Sec ID"] = sec_ids
    df["Distance"] = distances
    df["Section_L"] = lens
    df["Section_diam"] = diams
    df["Section_nseg"] = nsegs
    df["Section_Ra"] = RAs
    df["Coord X"] = all_seg_x_coords
    df["Coord Y"] = all_seg_y_coords
    df["Coord Z"] = all_seg_z_coords
    df["ParentSegID"] = psegids
    df["Passive ElecDist"] = passive_elec_distances
    df["Active ElecDist"] = active_elec_distances
    df["Section Name"] = section_names_by_seg


    df.to_csv(savename, index=False)

def generate_reduced_stylized_geometry(reduced_cell=None,complex_geometry_file='geom_complex.csv',savename='geom_reduced.csv'):
  '''
  builds reduced_geom.csv 
  using dendritic sections from neuron_reduce()'s reduced_cell object 
  and soma from complex cell
  '''

  complex_df = pd.read_csv(complex_geometry_file) #need to begin reduced_df with soma from complex_cell
  df = pd.DataFrame()
  ids=[complex_df['id'][0]]
  names=[complex_df['name'][0]]
  types=[complex_df['type'][0]]
  pids=[complex_df['pid'][0]]
  axials=[complex_df['axial'][0]]
  nbranchs=[complex_df['nbranch'][0]]
  Ls=[complex_df['L'][0]]
  Rs=[complex_df['R'][0]]
  angs=[complex_df['ang'][0]]

  for sec in reduced_cell.hoc_model.all:
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
      pids.append(names.index(psec.name()))
    axials.append('TRUE')
    nbranchs.append(1)
    Ls.append(sec.L)
    # print(dir(sec))
    Rs.append(sec.diam/2)
    if sec_type == "apic":
      ang=np.random.normal(scale=0.1,loc=1.570796327)
    elif sec_type=="dend":
      ang=np.random.uniform(low=0,high=-np.pi)
    else:
      ang=0
    angs.append(ang)


    # if sec.n3d() != 0: # all reduced cell 3d coordinates are missin; need a fix for angles
    #   print(sec.n3d())
    #   first_x3d=sec.x3d(0)
    #   first_y3d=sec.y3d(0)
    #   first_z3d=sec.z3d(0)
    #   last_x3d=sec.x3d(sec.n3d()-1)
    #   last_y3d=sec.y3d(sec.n3d()-1)
    #   last_z3d=sec.z3d(sec.n3d()-1)

    #   pfirst_x3d=psec.x3d(0)
    #   pfirst_y3d=psec.y3d(0)
    #   pfirst_z3d=psec.z3d(0)
    #   plast_x3d=psec.x3d(psec.n3d()-1)
    #   plast_y3d=psec.y3d(psec.n3d()-1)
    #   plast_z3d=psec.z3d(psec.n3d()-1)
    #   print(first_x3d-plast_x3d)
    #   print(first_y3d-plast_y3d)
    #   print(first_z3d-plast_z3d)
    # else:
    #   angs.append(None)

  # print(len(ids))
  # print(len(names))
  # print(len(types))
  # print(len(pids))
  # print(len(axials))
  # print(len(nbranchs))
  # print(len(Ls))
  # print(len(Rs))
  # print(angs)
  df["id"] = ids
  df["name"] = names
  df["L"] = pids
  df["axial"] = axials
  df["type"] = types
  df["nbranch"] = nbranchs
  df["L"] = Ls
  df["R"] = Rs
  df["ang"] = angs
  df.to_csv(savename)

def generate_reduced_cell_seg_coords(cell):
  '''
  WILL NEED TO BE EDITED FOR EXPANDED REDUCED CELL
  takes a cell that has no n3d() coordinates and gives new coordinates
  by choosing an arbitrary direction for the subtree to move
  '''
  parent_sections=[] #list for already seen parent_sections
  try: section_obj_list=cell.all
  except: section_obj_list=cell.hoc_model.all
  print(section_obj_list)
  axial=False
  for sec in section_obj_list:
    if sec.n3d() !=0 :
      if sec.parentseg() is not None:
        psec=sec.parentseg().sec
        parent_sections.append(psec)
        if psec==cell.soma:
          nbranch=1
        else:
          nbranch = len(psec.children())
        rot = 2 * math.pi/nbranch #one branch
        i=parent_sections.count(psec)
        length=sec.L
        diameter=sec.diam
        fullsecname = sec.name()
        sec_type = fullsecname.split(".")[1][:4]
        if sec_type == "apic":
          ang=np.random.normal(scale=0.1,loc=1.570796327)
        elif sec_type=="dend":
          ang=-np.random.uniform(low=0,high=np.pi)
        else:
          ang=0
        if axial == True:
          x = 0
          y = length*((ang>=0)*2-1)
        else:
          x = length * math.cos(ang)
          y = length * math.sin(ang)
        #find starting position
        pt0 = [psec.x3d(1), psec.y3d(1), psec.z3d(1)]
        pt1 = [0., 0., 0.]
        pt1[1] = pt0[1] + y
        pt1[0] = pt0[0] + x * math.cos(i * rot)
        pt1[2] = pt0[2] + x * math.sin(i * rot)
        sec.pt3dadd(*pt0, sec.diam)
        sec.pt3dadd(*pt1, sec.diam)

def plot_morphology(exc_syns,savename):
  plt.figure(figsize=(4,10))
  for i in exc_syns[exc_syns.Type=='apic']['Sec ID'].unique():
      plt.plot(exc_syns[(exc_syns.Type=='apic')&(exc_syns['Sec ID']==i)]['Coord X'],
              exc_syns[(exc_syns.Type=='apic')&(exc_syns['Sec ID']==i)]['Coord Y'],
              color='k',
              linewidth = 2*exc_syns[(exc_syns.Type=='apic')&(exc_syns['Sec ID']==i)]['Section_diam'].unique())
      
  for i in exc_syns[exc_syns.Type=='dend']['Sec ID'].unique():
      plt.plot(exc_syns[(exc_syns.Type=='dend')&(exc_syns['Sec ID']==i)]['Coord X'],
          exc_syns[(exc_syns.Type=='dend')&(exc_syns['Sec ID']==i)]['Coord Y'],
              color='k',
              linewidth = 2*exc_syns[(exc_syns.Type=='dend')&(exc_syns['Sec ID']==i)]['Section_diam'].unique())
      

  plt.scatter(exc_syns[(exc_syns.Type=='soma')&(exc_syns['Sec ID']==0)]['Coord X'],
          exc_syns[(exc_syns.Type=='soma')&(exc_syns['Sec ID']==0)]['Coord Y'],color='k',s=100)
  plt.savefig(savename)

  
def check_connectivity(cell):
    '''
    prints the sections within the given cell object's section lists and each section's children sections
    '''
    print('SOMA')
    for sec in [cell.soma]:
        if sec.children() != []:
            print(sec,'children',sec.children())
        else:
            print(sec)
    print('APICAL')
    for sec in cell.apic:
        if sec.children() != []:
            print(sec,'children',sec.children())
        else:
            print(sec)
    print('BASAL')
    for sec in cell.dend:
        if sec.children() != []:
            print(sec,'children',sec.children())
        else:
            print(sec)
    print('AXONAL')
    for sec in cell.axon:
        if sec.children() != []:
            print(sec,'children',sec.children())
        else:
            print(sec)
            
def create_seg_var_report(reportname,dataname):
  try:
    os.remove(reportname)
    #print('replacing ',reportname)
  except:
    pass
    #print('creating ',reportname)
    

  f = h5py.File(reportname,'w')
  v = f.create_dataset("report/biophysical/data", data = dataname)
  f.close()
