import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from typing import Union, Optional, List, Tuple

from .utils.currents.ecp import move_position
from cells.simulation import Simulation
from cells.stylizedcell import StylizedCell


def plot_morphology(sim: Optional[Simulation] = None, cellid: Optional[int] = 0, cell: Optional[StylizedCell] = None,
                    seg_coords: Optional[dict] = None, sec_nseg: Optional[List] = None,
                    type_id: Optional[List] = None, electrodes: Optional[np.ndarray] = None,
                    axes: Union[List[int], Tuple[int]] = [2, 0, 1], clr: Optional[List[str]] = None,
                    elev: int = 20, azim: int = 10, move_cell: Optional[Union[List,np.ndarray]] = None,
                    figsize: Optional[Tuple[float, float]] = None) -> Tuple[Figure, Axes]:
    """
    Plot morphology in 3D.

    sim: simulation object
    cellid: cell id in the simulation object. Default: 0
    cell: stylized cell object. Ignore sim and cellid if specified
    seg_coords: if not using sim or cell, a dictionary that includes dl, pc, r
    sec_nseg: if not using sim or cell, list of number of segments in each section
    type_id:  if not using sim or cell, list of the swc type id of each section/segment
    electrodes: electrode positions. Default: None, not shown.
    axes: sequence of axes to display in 3d plot axes.
        Default: [2,0,1] show z,x,y in 3d plot x,y,z axes, so y is upward.
    clr: list of colors for each type of section
    Return Figure object, Axes object
    """
    if sim is None and cell is None:
        if seg_coords is None or sec_nseg is None or type_id is None:
            raise ValueError("If not using 'Simulation', input arguments 'seg_coords', 'sec_nseg', 'type_id' are required.")
        if clr is None:
            clr = ('g', 'r', 'b', 'c')
        if move_cell is None:
            move_cell = [0., 0., 0., 0., 1., 0.]
        sec_id_in_seg = np.cumsum([0] + list(sec_nseg[:-1]))
        type_id = np.asarray(type_id) - 1
        if type_id.size != len(sec_nseg):
            type_id = type_id = type_id[sec_id_in_seg]
        type_id = type_id.tolist()
        label_idx = np.array([type_id.index(i) for i in range(4)])
        lb_odr = np.argsort(label_idx)
        label_idx = label_idx[lb_odr].tolist()
        sec_name = np.array(('soma','axon','dend','apic'))[lb_odr]
    else:
        if clr is None:
            clr = ('g', 'b', 'pink', 'purple', 'r', 'c')
        if cell is None:
            if move_cell is None:
                move_cell = sim.loc_param[cellid, 0]
            cell = sim.cells[cellid]
        elif move_cell is None:
            move_cell = [0., 0., 0., 0., 1., 0.]
        seg_coords = cell.seg_coords
        sec_id_in_seg = cell.sec_id_in_seg
        sec_nseg = []
        sec_name = []
        label_idx = []
        type_id = []
        for i, sec in enumerate(cell.all):
            sec_nseg.append(sec.nseg)
            name = sec.name().split('.')[-1]
            if name not in sec_name:
                sec_name.append(name)
                label_idx.append(i)
            type_id.append(sec_name.index(name))
    label_idx.append(-1)

    move_cell = np.asarray(move_cell).reshape((2, 3))
    dl = move_position([0., 0., 0.], move_cell[1], seg_coords['dl'])
    pc = move_position(move_cell[0], move_cell[1], seg_coords['pc'])
    xyz = 'xyz'
    box = np.vstack([np.full(3, np.inf), np.full(3, np.NINF)])
    if electrodes is not None:
        box[0, axes[0:2]] = np.amin(electrodes[:, axes[0:2]], axis=0)
        box[1, axes[0:2]] = np.amax(electrodes[:, axes[0:2]], axis=0)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    lb_ptr = 0
    for i, itype in enumerate(type_id):
        label = sec_name[lb_ptr] if i == label_idx[lb_ptr] else None
        if label is not None: lb_ptr += 1
        i0 = sec_id_in_seg[i]
        i1 = i0 + sec_nseg[i] - 1
        if sec_name[itype] == 'soma':
            p05 = (pc[i0] + pc[i1]) / 2
            ax.scatter(*[p05[j] for j in axes], c=clr[itype], s=20, label=label)
        else:
            p0 = pc[i0] - dl[i0] / 2
            p1 = pc[i1] + dl[i1] / 2
            ax.plot3D(*[(p0[j], p1[j]) for j in axes], color=clr[itype], label=label)
            box[0, :] = np.minimum(box[0, :], np.minimum(p0, p1))
            box[1, :] = np.maximum(box[1, :], np.maximum(p0, p1))
    ctr = np.mean(box, axis=0)
    r = np.amax(box[1, :] - box[0, :]) / 2
    box = np.vstack([ctr - r, ctr + r])
    if electrodes is not None:
        idx = np.logical_and(np.all(electrodes >= box[0, :], axis=1), np.all(electrodes <= box[1, :], axis=1))
        ax.scatter(*[(electrodes[idx, j], electrodes[idx, j]) for j in axes], color='orange', s=5, label='electrodes')
    box = box[:, axes]
    ax.auto_scale_xyz(*box.T)
    ax.view_init(elev, azim)
    ax.legend(loc=1)
    ax.set_xlabel(xyz[axes[0]])
    ax.set_ylabel(xyz[axes[1]])
    ax.set_zlabel(xyz[axes[2]])
    plt.show()
    return fig, ax
