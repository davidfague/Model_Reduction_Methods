from neuron_reduce import subtree_reductor
from cable_expander import cable_expander_func

class Reductor():
  def __init__(cell, method=str, synapses_list=None, netcons_list=None, reduction_frequency=0, sections_to_expand=None, furcations_x=None, nbranches=None, return_seg_to_seg: bool = False):
    '''
    cell: hoc model cell object (TO TEST: providing python cell_model object instead)
    method: str for method to use ex. 'expand cable', 'neuron_reduce'
    synapses_list: list of synapse objects
    netcons_list: list of netcon objects
    reduction_frequency: frequency used in calculated transfer impedance
    sections_to_expand: list of sections for 'cable expand' method
    furcations_x: list of x (0 to 1) locs corresponding to sections_to_expand to bifurcate section at
    nbranches: list of integar number of branches corresponding to sections_to_expand to choose the number of new branches at furcation_x
    return_seg_to_seg: bool for returning a dictionary mapping original segments to reduced.
    '''
    if method == 'expand cable': # sanity checks
      if sections_to_expand:
        if furcations_x:
          if nbranches:
            reduced_cell, synapses_list, netcons_list, txt = cable_expander(cell, sections_to_expand, furcations_x, nbranches,
                                                                              synapses_list, netcons_list, reduction_frequency=reduction_frequency,return_seg_to_seg=True)
          else:
            raise ValueError('Must specify nbranches list for cable_expander()')
        else:
          raise ValueError('Must specify furcations_x list for cable_expander()')
      else:
        raise ValueError('Must specify sections_to_expand for cable_expander()')
    elif method == 'neuron_reduce':
      reduced_cell, synapses_list, netcons_list, txt = subtree_reductor(cell, synapses_list, netcons_list, reduction_frequency=reduction_frequency,return_seg_to_seg=True)
    else:
      raise ValueError(f"Method '{method}' not implemented.")

    if return_seg_to_seg:
      return reduced_cell, synapses_list, netcons_list, txt
    else:
      return reduced_cell, synapses_list, netcons_list
