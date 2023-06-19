import numpy as np

class SpikeGenerator():
  
	def __init__(self):
		self.netcons = []
		self.spike_trains = []
	
	#TODO: add docstring, check typing
	def generate_inputs(self, synapses, t, mean_firing_rate, method: str, same_presynaptic_cell: bool = False, 
		     			same_presynaptic_region: bool = False, 
					  	rhythmicity: bool = False, 
						rhythmic_mod = None, rhythmic_f = None,
					  	spike_trains_to_delay = None, time_shift = None) -> tuple:
		'''
		Generate spike trains.

		Parameters:
		----------
		synapses: list of synapse objects
		t: time vector
		method: str to choose how to vary the profile over time ex. '1/f noise', 'delay', 'rhythmic'
		mean_firing_rate: float or distribution of mean firing rate of spike train
		spike_trains_to_delay: list of time stamps where spikes occured for delay modulation
		'''
	
		spike_trains = []
		netcons_list = []

		if same_presynaptic_cell: # same fr profile # same spike train # same mean fr
			mean_fr = get_mean_fr(mean_firing_rate)
			fr_profile = self.firing_rate_profile(method, t, rhythmicity, spike_trains_to_delay)
			spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
			for synapse in synapses:
				netcons_list.append(self.set_spike_train(synapse, spikes))
				spike_trains.append(spikes)
		  
		elif same_presynaptic_region: # same fr profile # unique spike train # unique mean fr
			fr_profile = self.firing_rate_profile(method, t, rhythmicity, spike_trains_to_delay)
			for synapse in synapses:
				mean_fr = get_mean_fr(mean_firing_rate)
				spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
				netcons_list.append(self.set_spike_train(synapse, spikes))
				spike_trains.append(spikes)
			
		else: # unique fr profile # unique spike train # unqiue mean fr
			for synapse in synapses:
				fr_profile = self.firing_rate_profile(method, t, rhythmicity, spike_trains_to_delay)
				mean_fr = get_mean_fr(mean_firing_rate)
				spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
				netcons_list.append(self.set_spike_train(synapse, spikes))
				spike_trains.append(spikes)
			
		self.netcons.append(netcons_list)
		self.spike_trains.append(spike_trains)
		
		return netcons_list, spike_trains
		
  
	def firing_rate_profile(self, t, mean_firing_rate, method: str, 
						  	rhythmicity = False, rhythmic_f = None, rhythmic_mod = None,
						  	spike_trains_to_delay = None, time_shift = None):
		'''
		method for creating a firing rate profile / modulatory trace
		'''
		fr_profile = np.zeros((t.shape[0]))
		
		if method == '1/f noise':
			fr_profile = noise_modulation(fr_profile)
		elif method == 'delay':
			if spike_trains_to_delay:
				if time_shift:
					fr_profile = delay_modulation(fr_profile, spike_trains_to_delay, time_shift, t)
				else:
					raise ValueError('Must specify time_shift for delay modulation')
			else:
				raise ValueError('Must specify spike_trains_to_delay for delay modulation')
		else:
			raise ValueError(f"Method '{method}' not implemented.")
		
		if rhythmicity:
			if rhythmic_f:
				if rhythmic_mod:
					fr_profile = rhythmic_modulation(fr_profile, rhythmic_f, rhythmic_mod, t)
				else:
					raise ValueError('Must specify rhythmic_mod for rhythmic modulation')
		else:
			raise ValueError('Must specify rhythmic_f for rhythmic modulation')
		
		fr_profile[fr_profile < 0] = 0 # Can't have negative firing rates.
		
		return fr_profile
	
	#TODO: fix minmax call
	def noise_modulation(fr_profile, 
						B = [0.049922035, -0.095993537, 0.050612699, -0.004408786],
						A = [1, -2.494956002,   2.017265875,  -0.522189400]):
		wn = np.random.normal(loc=1,scale=0.5,size=(len(fr_profile)+2000))
		fr_profile[:] = minmax(ss.lfilter(B, A, wn)[2000:])+0.5 # Create '1/f' Noise
		return fr_profile
	
	#TODO: fix minmax call, vars
	def delay_modulation(times_where_spikes, time_shift, t):
		'''
		'''
		hist = np.histogram(times_where_spikes,bins=t)
		fr_profile = hist[0]/(0.001*(len(times_where_spikes)+1))
		fr_profile = list(fr_prof)
		wrap = fr_profile[-time_shift:]
		fr_profile[time_shift:] = fr_profile[0:-time_shift]
		fr_profile[0:time_shift] = wrap
		fr_profile = minmax(fr_profile)+0.5
		return fr_profile

	#TODO:
	def rhythmic_modulation(fr_profile, rhythmic_f, rhythmic_mod, t):
		'''
		method for modulating spike_trains by rhythmicity
		'''
		A = fr_profile / ((1/rhythmic_mod)-1)
		fr_profile[0,:] = A*np.sin((2 * np.pi * rhythmic_f * t)+P) + fr_profile
		return fr_profile
	
	#TODO: fix call
	def generate_spikes_from_profile(fr_profile):
		''' sample spikes '''
		sample_values = st.poisson(fr_profile/1000).rvs() #Poisson number of points
		spike_times = np.where(sample_values>0)[0]
		return spike_times
	
	#TODO: fix call
	def set_spike_train(synapse, spikes):
		stim = set_vecstim(spikes)
		nc = set_netcon(stim,synapse)
		return nc
	
	#TODO: fix call
	def set_vecstim(spikes):
		vec = h.Vector(stim_spikes)
		stim=h.VecStim()
		stim.play(vec)
	
	#TODO: fix self
	def set_netcon(synapse, stim):
		nc = h.NetCon(stim, synapse.pp_obj, 1, 0, 1)
		netcons_list.append(nc)
		synapse.ncs.append(nc)
		return nc
	
	#TODO: fix self
	def get_mean_fr(mean_firing_rate):
		''' gets mean_fr if it is a distribution or float'''
		if callable(mean_firing_rate): # mean_firing_rate is a distribution
			mean_fr = mean_firing_rate(size=1) # sample from distribution
		else: # mean firing_rate is a float
			mean_fr = mean_firing_rate
		return mean_fr

	#TODO: Walt's Check, better implementation for complete removal of inputs: netcon removed from synapse.ncs, netcons lists
	def remove_inputs(synapses=None,netcons=None):
		''' Makes netcons inactive (do not deliver their spike trains) but netcons remove present '''
		if synapses:
			for synapse in synapses:
				for netcon in synapse.ncs:
					netcon.active(False)
		if netcons:
			for netcon in netcons:
				netcon.active(False)
