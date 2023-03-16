from abc import ABC, abstractmethod
from neuron import h
import math
import numpy as np
import pandas as pd
from typing import Optional, Union, List
from enum import Enum

from cell_inference.utils.currents.currentinjection import CurrentInjection
from cell_inference.utils.currents.synapse import Synapse
from cell_inference.utils.currents.recorder import Recorder

h.load_file('stdrun.hoc')

class CellTypes(Enum):
    PASSIVE = 1
    ACTIVE = 2
    ACTIVE_FULL = 3
    REDUCED_ORDER = 4


class StylizedCell(ABC):
    def __init__(self, geometry: pd.DataFrame = None,
                 dl: float = 30., vrest: float = -70.0, nbranch: int = 4,
                 record_soma_v: bool = True, spike_threshold: Optional[float] = None,
                 attr_kwargs: dict = {}):
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        dL: maximum segment length
        vrest: reversal potential of leak channel for all segments
        nbranch: number of branches of each non-axial section
        record_soma_v: whether or not to record soma membrane voltage
        spike_threshold: membrane voltage threshold for recording spikes, if not specified, do not record
        attr_kwargs: dictionary of class attribute - value pairs
        """
        self._h = h
        self._dL = dl
        self._vrest = vrest
        self._nbranch = max(nbranch, 2)
        self._record_soma_v = record_soma_v
        self.spike_threshold = spike_threshold
        self._nsec = 0
        self._nseg = 0
        self.soma = None
        self.all = []  # list of all sections
        self.segments = []  # list of all segments
        self.sec_id_lookup = {}  # dictionary from section type id to section index
        self.sec_id_in_seg = []  # index of the first segment of each section in the segment list
        self.injection = []  # current injection objects
        self.synapse = []  # synapse objects
        self.spikes = None
        self.geometry = None
        self.biophysical_division()
        for key, value in attr_kwargs.items():
            setattr(self, key, value)
        self.__set_geometry(geometry)
        self.__setup_all()

    #  PRIVATE METHODS
    def __set_geometry(self, geometry: Optional[pd.DataFrame] = None):
        if geometry is None:
            raise ValueError("geometry not specified.")
        else:
            if not isinstance(geometry, pd.DataFrame):
                raise TypeError("geometry must be a pandas dataframe")
            if geometry.iloc[0]['type'] != 1:
                raise ValueError("first row of geometry must be soma")
            self.geometry = geometry.copy()

    def __setup_all(self):
        self.__create_morphology()
        self.__calc_seg_coords()
        self.set_channels()
        self.v_rec = self.__record_soma_v() if self._record_soma_v else None
        self.__set_spike_recorder()

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

    def __create_morphology(self):
        """Create cell morphology"""
        self._nsec = 0
        rot = 2 * math.pi / self._nbranch
        for sec_id, sec in self.geometry.iterrows():
            start_idx = self._nsec
            if sec_id == 0:
                r0 = sec['R']
                pt0 = [0., -2 * r0, 0.]
                pt1 = [0., 0., 0.]
                self.soma = self.__create_section(name=sec['name'], diam=2 * r0)
                self.__set_location(self.soma, pt0, pt1, 1)
            else:
                length = sec['L']
                radius = sec['R']
                ang = sec['ang']
                nseg = math.ceil(length / self._dL)
                pid = self.sec_id_lookup[sec['pid']]
                if sec['axial']:
                    nbranch = 1
                    x = 0
                    y = length*((ang>=0)*2-1)
                else:
                    nbranch = self._nbranch
                    x = length * math.cos(ang)
                    y = length * math.sin(ang)
                    if len(pid) == 1:
                        pid = pid*nbranch
                for i in range(nbranch):
                    psec = self.all[pid[i]]
                    pt0 = [psec.x3d(1), psec.y3d(1), psec.z3d(1)]
                    pt1[1] = pt0[1] + y
                    pt1[0] = pt0[0] + x * math.cos(i * rot)
                    pt1[2] = pt0[2] + x * math.sin(i * rot)
                    section = self.__create_section(name=sec['name'], diam=2 * radius)
                    section.connect(psec(1), 0)
                    self.__set_location(section, pt0, pt1, nseg)
            self.sec_id_lookup[sec_id] = list(range(start_idx, self._nsec))
        self.__set_location(self.soma, [0., -r0, 0.], [0., r0, 0.], 1)
        self.__store_segments()

    def __create_section(self, name: str = 'null_sec', diam: float = 500.0) -> h.Section:
        sec = h.Section(name=name, cell=self)
        sec.diam = diam
        self.all.append(sec)
        self._nsec += 1
        return sec

    def __set_location(self, sec: h.Section, pt0: List[float], pt1: List[float], nseg: int):
        sec.pt3dclear()
        sec.pt3dadd(*pt0, sec.diam)
        sec.pt3dadd(*pt1, sec.diam)
        sec.nseg = nseg

    def __store_segments(self):
        self.segments = []
        self.sec_id_in_seg = []
        nseg = 0
        for sec in self.all:
            self.sec_id_in_seg.append(nseg)
            nseg += sec.nseg
            for seg in sec:
                self.segments.append(seg)
        self._nseg = nseg

    def __record_soma_v(self) -> Recorder:
        return Recorder(self.soma(.5), 'v')

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

    #  PUBLIC METHODS
    @abstractmethod
    def set_channels(self):
        """Abstract method for setting biophysical properties, inserting channels"""
        pass

    def biophysical_division(self):
        """Define biophysical division in morphology"""
        pass

    def get_sec_by_id(self, index):
        """Get section(s) objects by index(indices) in the section list"""
        if hasattr(index, '__len__'):
            sec = [self.all[i] for i in index]
        else:
            sec = self.all[index]
        return sec

    def get_seg_by_id(self, index):
        """Get segment(s) objects by index(indices) in the segment list"""
        if hasattr(index, '__len__'):
            seg = [self.segments[i] for i in index]
        else:
            seg = self.segments[index]
        return seg

    def set_all_passive(self, gl: float = 0.0003):
        """A use case of 'set_channels', set all sections passive membrane"""
        for sec in self.all:
            sec.cm = 1.0
            sec.insert('pas')
            sec.g_pas = gl
            sec.e_pas = self._vrest

    def add_injection(self, sec_index, **kwargs):
        """Add current injection to a section by its index"""
        self.injection.append(CurrentInjection(self, sec_index, **kwargs))

    def add_synapse(self, stim: h.NetStim, sec_index: int, **kwargs):
        """Add synapse to a section by its index"""
        self.synapse.append(Synapse(self, stim, sec_index, **kwargs))

    def v(self) -> Optional[Union[str, np.ndarray]]:
        """Return recorded soma membrane voltage in numpy array"""
        if self.v_rec is None:
            raise NotImplementedError("Soma membrane voltage has not been recorded")
        else:
            return self.v_rec.as_numpy()
