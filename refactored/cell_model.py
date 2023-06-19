import numpy as np
import warnings

# Typing
from typing import TypeVar
NeuronHocTemplate = TypeVar("NeuronHocTemplate")
NeuronAnyType = TypeVar("NeuronAnyType")
NeuronSection = TypeVar("NeuronSection")
NeuronSegment = TypeVar("NeuronSegment")
NeuronAnyNumber = TypeVar("NeuronAnyNumber")

class CellModel:

    def __init__(self, hoc_model: NeuronHocTemplate, synapses: list, netcons: list) -> None:

        # Parse the hoc model
        self.all, self.soma, self.apic, self.dend, self.axon = None
        for model_part in ["all", "soma", "apic", "dend", "axon"]:
            setattr(self, model_part, self.convert_section_list(getattr(hoc_model, model_part)))
        
        self.synapses = synapses
        self.netcons = netcons

        # Angles and rotations that were used to branch the cell
        # Store to use for geometry file generation
        self.sec_angs = [] 
        self.sec_rots = []

        self.generate_sec_coords()
        self.seg_coords = self.calc_seg_coords()

    # PRAGMA MARK: Section Generation

    # TODO: CHECK
    def generate_sec_coords(self, verbose = True) -> None:

        for sec in self.all:
            # Do only for sections without already having 3D coordinates
            if sec.n3d() != 0: continue

            if verbose: print(f"Generating 3D coordinates for {sec}")

            # Store for a check later
            old_length = sec.L

            if sec is self.soma:
                new_length = self.process_soma_sec(sec, verbose)
            else:
                # Get the parent segment, sec
                pseg = sec.parentseg()
                if psec is None: raise RuntimeError("Section {sec} is attached to None.")
                psec = pseg.sec

                # Process and get the new length
                new_length = self.process_non_soma_sec(sec, psec, pseg)
            
            if int(new_length) != int(old_length):
                warnings.warn(f"Generation of 3D coordinates resulted in change of section length for {sec} from {old_length} to {sec.L}",
                              RuntimeWarning)

    def process_soma_sec(self, sec: NeuronSection, verbose: bool) -> NeuronAnyNumber:
        self.sec_angs.append(0)
        self.sec_rots.append(0)

        if sec.nseg != 1:
            if verbose:
                print(f'Changing soma nseg from {sec.nseg} to 1')
            sec.nseg = 1

        sec.pt3dclear()
        sec.pt3dadd(*[0., -1 * sec.L / 2., 0.], sec.diam)
        sec.pt3dadd(*[0., sec.L / 2., 0.], sec.diam)

        return sec.L

    def process_non_soma_sec(self, sec: NeuronSection, psec: NeuronSection, pseg: NeuronSegment) -> NeuronAnyNumber:
        # Get random theta and phi values for apical tuft and basal dendrites
        theta, phi = self.generate_phi_theta_for_apical_tuft_and_basal_dendrites(sec)

        # Find starting position using parent segment coordinates
        pt0 = self.find_starting_position_for_a_non_soma_sec(psec, pseg)

        # Calculate new coordinates using spherical coordinates
        xyz = [sec.L * np.sin(theta) * np.cos(phi), 
               sec.L * np.cos(theta), 
               sec.L * np.sin(theta) * np.sin(phi)]
        
        pt1 = [pt0[k] + xyz[k] for k in range(3)]

        sec.pt3dclear()
        sec.pt3dadd(*pt0, sec.diam)
        sec.pt3dadd(*pt1, sec.diam)

        return sec.L

    def generate_phi_theta_for_apical_tuft_and_basal_dendrites(self, sec: NeuronSection) -> tuple:
        if sec in self.apic:
            if sec != self.apic[0]: # Trunk
                theta, phi = np.random.uniform(0, np.pi / 2), np.random.uniform(0, 2 * np.pi)
            else:
                theta, phi = 0, np.pi/2
        elif sec in self.dend:
            theta, phi = np.random.uniform(np.pi / 2, np.pi), np.random.uniform(0, 2 * np.pi)
        else:
            theta, phi = 0, 0
        
        return theta, phi
    
    def find_starting_position_for_a_non_soma_sec(self, psec: NeuronSection, pseg: NeuronSegment) -> list:
        for i in range(psec.n3d() - 1):
            arc_length = (psec.arc3d(i), psec.arc3d(i + 1)) # Before, After
            if (arc_length[0] / psec.L) <= pseg.x <= (arc_length[1] / psec.L):
                # pseg.x is between 3d coordinates i and i+1
                psec_x_between_coordinates = (pseg.x * psec.L - arc_length[0]) / (arc_length[1] - arc_length[0])

                #  Calculate 3d coordinates at psec_x_between_coordinates
                xyz_before = [psec.x3d(i), psec.y3d(i), psec.z3d(i)]
                xyz_after = [psec.x3d(i + 1), psec.y3d(i+1), psec.z3d(i + 1)]
                xyz = [xyz_before[k] + (xyz_after[k] - xyz_before[k]) * psec_x_between_coordinates for k in range(3)]
                break

        return xyz
    
    # PRAGMA MARK: Segment Generation
    
    def calc_seg_coords(self) -> dict:

        nseg_total = sum(sec.nseg for sec in self.all)
        p0, p05, p1 = np.zeros((nseg_total, 3))
        r = np.zeros(nseg_total)

        seg_idx = 0
        for sec in self.all:

            seg_length = sec.L / sec.nseg

            for i in range(sec.n3d()-1):
                arc_length = [sec.arc3d(i), sec.arc3d(i+1)] # Before, after
                for seg in sec:
                    if (arc_length[0] / sec.L) <= seg.x < (arc_length[1] / sec.L):
                        seg_x_between_coordinates = (seg.x * sec.L - arc_length[0]) / (arc_length[1] - arc_length[0])
                        xyz_before = [sec.x3d(i), sec.y3d(i), sec.z3d(i)]
                        xyz_after = [sec.x3d(i+1), sec.y3d(i+1), sec.z3d(i+1)]

                        pt = np.array([xyz_before[k] + (xyz_after[k] - xyz_before[k]) * seg_x_between_coordinates for k in range(3)])
                        dxdydz = np.array([(xyz_after[k] - xyz_before[k]) * (seg_length / 2) / (arc_length[1] - arc_length[0]) for k in range(3)])
                        
                        pt_back, pt_forward = pt - dxdydz, pt + dxdydz

                        p0[seg_idx], p05[seg_idx], p1[seg_idx] = pt_back, pt, pt_forward
                        r[seg_idx] = seg.diam / 2

                        seg_idx += 1

        seg_coords = {'p0': p0, 'p1': p1, 'pc': p05, 'r': r, 'dl': p1 - p0}

        return seg_coords


    # PRAGMA MARK: Utils

    # TODO: CHECK
    def convert_section_list(self, section_list: NeuronAnyType) -> list:

        # If the section list is a hoc object, add its sections to the python list
        if str(type(section_list)) == "<class 'hoc.HocObject'>":
            new_section_list = [sec for sec in section_list]

        # Else, the section list is actually one section, add it to the list
        elif str(type(section_list)) == "<class 'nrn.Section'>":
            new_section_list = [section_list]

        else:
            raise TypeError

        return new_section_list
