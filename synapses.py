import random
from synapses.py import Synapse

class SynapseGenerator:
    def __init__(self):
        self.synapses = [] # list of lists of synapses that were generated using this class
        
    def add_synapses(self, segments, probs=None, 
                     exc_density=None, inh_density=None,
                     total_synapses=int, ratio=None):
      '''
      Creates a list of synapses by specifying excitatory and inhibitory densities, or number of synapses and ratio
      --------------------------------------------------------------------------------------------------------
      segments: list of neuron segments to choose from
      probs: list of probabilities of choosing each segment -- if not provided, segment_length/total_segment_length will be used
      exc_density: density of excitatory synapses in synapses/micron
      inh_density: density of inhibitory synapses in synapses/micron
      total_synapses: integar number of synapses to distribute to the cell -- Optional
      ratio: proporiton of synapses that are excitatory (0 to 1)
      '''
      synapses=[]
      
      # Error checking
      if (exc_density is not None and inh_density is not None) and (total_synapses is not None or ratio is not None):
        raise ValueError('Cannot specify both densities and total_synapses/ratio')

      # Calculate probabilities if not given
      if not probs:
          total_length = sum([seg.sec.L/seg.sec.nseg for seg in segments])
          probs = [(seg.sec.L/seg.sec.nseg)/total_length for seg in segments]

      # Calculate total synapses if given densities
      if exc_density and inh_density:
          total_length = sum([seg.sec.L/seg.sec.nseg for seg in segments])
          total_synapses = total_length * (exc_density + inh_density)
          ratio = exc_density / inh_density

      # Calculate number of excitatory and inhibitory synapses
      num_exc_synapses = round(total_synapses * ratio / (1 + ratio))
      num_inh_synapses = total_synapses - num_exc_synapses

      # Add synapses
      for _ in range(num_exc_synapses):
          segment = random.choices(segments, probs)[0]
          synapses.append(Synapse(segment, 'excitatory'))

      for _ in range(num_inh_synapses):
          segment = random.choices(segments, probs)[0]
          synapses.append(Synapse(segment, 'inhibitory'))
          
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
