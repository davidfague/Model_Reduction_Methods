from abc import ABC, abstractmethod
from typing import List, Optional, Any, TYPE_CHECKING
import numpy as np

class PointCurrent(ABC):
    """A module for current point process"""

    def __init__(self, segment):
        """
        segment: target segment
        """
        self.segment = segment
        self.pp_obj = None  # point process object
        self.rec_vec = None  # vector for recording

    @abstractmethod
    def setup(self, record: bool = None) -> None:
        pass

    def setup_recorder(self):
        size = [round(h.tstop / h.dt) + 1] if hasattr(h, 'tstop') else []
        self.rec_vec = h.Vector(*size).record(self.pp_obj._ref_i)

    def get_section(self) -> h.Section:
        return self.pp_obj.get_segment().sec

    def get_segment(self):
        return self.pp_obj.get_segment()
      
class CurrentInjection(PointCurrent):
    """A module for current injection
    was current: Optional[np.ndarray, List[int]] = None,
    """

    def __init__(self, segment,
                 pulse: bool = True, current: Optional[np.ndarray] = None,
                 dt: Optional[np.ndarray] = None, record: bool = False, **pulse_param: Any) -> None:
        """
        segment: target segment
        pulse: If True, use pulse injection with keyword arguments in 'pulse_param'
               If False, use waveform resources in vector 'current' as injection
        Dt: current vector time step size
        record: If True, enable recording current injection history
        """
        super().__init__(segment)
        self.pp_obj = h.IClamp(self.segment)
        self.inj_vec = None
        if pulse:
            self.setup_pulse(**pulse_param)
        else:
            if current is None:
                current = [0]
            self.setup_current(current, dt)
        self.setup(record)

    def setup(self, record: bool = False) -> None:
        if record:
            self.setup_recorder()

    def setup_pulse(self, **pulse_param: Any) -> None:
        """Set IClamp attributes. Argument keyword: attribute name, arugment value: attribute value
        was current: Optional[np.ndarray, List[int]]
        """
        for param, value in pulse_param.items():
            setattr(self.pp_obj, param, value)

    def setup_current(self, current: Optional[np.ndarray], dt: Optional[np.ndarray]) -> None:
        """Set current injection with the waveform in vector 'current'"""
        ccl = self.pp_obj
        ccl.dur = 0
        ccl.dur = h.tstop if hasattr(h, 'tstop') else 1e30
        if dt is None:
            dt = h.dt
        self.inj_vec = h.Vector()
        self.inj_vec.from_python(current)
        self.inj_vec.append(0)
        self.inj_vec.play(ccl._ref_amp, dt)

class Synapse(PointCurrent):
    '''
    class for adding synapses
    '''
    def __init__(self, segment,
                  syn_mod: str = 'Exp2Syn', gmax: float = 0.01,
                  record: bool = False):
        super().__init__(segment)
        self.stim = stim
        self.gmax = gmax
        self.__synapse_type(syn_mod)
        self.setup(record)
        self.ncs=[]

    # PRIVATE METHODS
    def __synapse_type(self, syn_mod):
        if syn_mod == 'AlphaSynapse1':
            # Reversal potential (mV); Synapse time constant (ms)
            self.syn_params = {'e': 0., 'tau': 2.0}
            # Variable name of maximum conductance (uS)
            self.gmax_var = 'gmax'
        elif syn_mod == 'Exp2Syn':
            self.syn_params = {'e': 0., 'tau1': 1.0, 'tau2': 3.0}
            self.gmax_var = '_nc_weight'
        elif syn_mod == 'pyr2pyr':
            self.gmax_var = 'initW'
        elif syn_mod == 'int2pyr':
            self.gmax_var = 'initW'
        elif 'AMPA_NMDA' in syn_mod:
            self.gmax_var = 'initW'
        elif 'GABA_AB' in syn_mod:
            self.gmax_var = 'initW'
        else:
            raise ValueError("Synpase type not defined.")
        self.syn_type = syn_mod
        self.pp_obj = getattr(h, syn_type)(self.segment)

    def __setup_synapse(self):
        self.syn = self.pp_obj
        for key, value in self.syn_params.items():
            setattr(self.syn, key, value)
        self.set_gmax()
        
    def setup_recorder(self):
      size = [round(h.tstop/h.dt)+1] if hasattr(h,'tstop') else []
      try:
          self.rec_vec = h.Vector(*size).record(self.pp_obj._ref_igaba)
          self.current_type = "igaba"
      except:
          try:
            self.rec_vec = MultiSynCurrent()
            vec_inmda = h.Vector(*size).record(self.pp_obj._ref_inmda)
            vec_iampa = h.Vector(*size).record(self.pp_obj._ref_iampa)
            self.rec_vec.add_vec(vec_inmda)
            self.rec_vec.add_vec(vec_iampa)
            self.current_type = "iampa_inmda"
          except:
            self.rec_vec = h.Vector(*size).record(self.pp_obj._ref_i)
            self.current_type = "i"

    # PUBLIC METHODS
    def setup(self, record: bool = False):
        self.__setup_synapse()
        if record:
            self.setup_recorder()

    def set_gmax(self, gmax: float = None):
        if gmax is not None:
            self.gmax = gmax
        if self.gmax_var == '_nc_weight':
            self.nc.weight[0] = self.gmax
        else:
            setattr(self.syn, self.gmax_var, self.gmax)
           
class MultiSynCurrent(object):
    '''
    Class for storing inmda and iampa
    '''
    def __init__(self):
        self.vec_list = []

    def add_vec(self,vec):
        self.vec_list.append(vec)

    def as_numpy(self):
        return np.sum(np.array([vec.as_numpy() for vec in self.vec_list]), axis=0)
