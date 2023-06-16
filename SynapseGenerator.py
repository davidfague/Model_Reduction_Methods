import random
from synapses.py import Synapse

class SynapseGenerator:
    def __init__(self):
        self.synapses = [] # list of lists of synapses that were generated using this class
        
    def add_synapses(self, segments, gmax, syn_mod = str,
                     probs=None,
                     density=None,
                     number_of_synapses=int,
                     record: bool = False):
      '''
      Creates a list of synapses by specifying density or number of synapses
      --------------------------------------------------------------------------------------------------------
      segments: list of neuron segments to choose from
      gmax: float or distribution to sample from
      syn_mod:  str of synapse modfile ex. 'AMPA_NMDA', 'pyr2pyr', 'GABA_AB', 'int2pyr'
      probs: list of probabilities of choosing each segment -- if not provided, segment_length/total_segment_length will be used
      density: density of excitatory synapses in synapses/micron 
      number_of_synapses: int number of synapses to distribute to the cell
      record: bool whether or not to record synapse currents
      '''
      synapses=[]
      
      # Error checking
      if (density is not None) and (number_of_synapses is not None):
        raise ValueError('Cannot specify both density and number_of_synapses')

      # Calculate probabilities if not given
      if not probs:
          total_length = sum([seg.sec.L/seg.sec.nseg for seg in segments])
          probs = [(seg.sec.L/seg.sec.nseg)/total_length for seg in segments]

      # Calculate number of synapses if given densities
      if density:
          total_length = sum([seg.sec.L/seg.sec.nseg for seg in segments])
          number_of_synapses = int(total_length * density)

      # Add synapses
      if callable(gmax): # gmax is distribution
          for _ in range(number_of_synapses):
              segment = random.choices(segments, probs)[0]
              synapses.append(Synapse(segment,
                      syn_mod = syn_mod, gmax = gmax(size=1),
                      record = record)
      else: # gmax is float
          for _ in range(number_of_synapses):
              segment = random.choices(segments, probs)[0]
              synapses.append(Synapse(segment,
                      syn_mod = syn_mod, gmax = gmax,
                      record = record)
                             
      self.synapses.append(synapses)

      return synapses
      
  def add_synapses_to_cell(self, cell, segments, probs=None, exc_density=None, inh_density=None, total_synapses=None, ratio=None):
      '''
      method for adding synapses to cell after cell python object has already been initialized
      see add_synapses for more information
      '''
      synapses = self.add_synapses(segments, probs, exc_density, inh_density, total_synapses, ratio)
      for syn in synapses:
        cell.synapses.append(syn)
