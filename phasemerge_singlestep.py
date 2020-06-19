import numpy as np
import genericio as gio
import h5py
from scipy import spatial #KD Tree
from itertools import combinations
import subhalo_mass_loss_model as SHMLM
from itk import h5_write_dict, many_to_one, pickle_save_dict
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
NUMRANKS = comm.Get_size()
step = 247

cc_data_dir = SHMLM.cc_data_dir
cc_output_dir = SHMLM.cc_output_dir

steps = SHMLM.steps

vars_cc = [
    'fof_halo_tag',
    'core_tag',
    'tree_node_index',
    'x',
    'y',
    'z',
    'vx',
    'vy',
    'vz',
    'radius',
    'infall_mass',
    'infall_step',
    'infall_fof_halo_tag',
    'infall_tree_node_index',
    'central',
    'merged',
    'vel_disp',
    'host_core'
]

def fname_cc(step, mode):
    if mode == 'input':
        return cc_data_dir + '09_03_2019.AQ.{}.coreproperties'.format(step)
    elif mode == 'output':
        return cc_output_dir + '09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step)

if rank==0:
    """Detects phase space core mergers at `step`."""
    print "Reading data from {}.".format(fname_cc(step, 'input'))

    # Read in cc for step
    cc = { v:gio.gio_read(fname_cc(step, 'input'), v)[0] for v in vars_cc }

    satellites_mask = cc['central'] == 0
    centrals_mask = cc['central'] == 1
    numSatellites = np.sum(satellites_mask)

    # Verify there are no satellites at first step
    if step == steps[0]:
        assert numSatellites == 0, 'Satellites found at first step.'

    #cc['mergedCoreTag'] = np.zeros_like(cc['core_tag'])
    #cc['mergedCoreStep'] = np.zeros_like(cc['infall_step'])
    # cc['phaseSpaceMerged'] = np.zeros_like(cc['central'])

    # If there are satellites (not applicable for first step)
    assert numSatellites != 0

    # Initialize M array (corresponds with cc[satellites_mask]) to be host tree node mass of each satellite
    idx_m21 = many_to_one( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask] )
    M, X, Y, Z = (cc[k][centrals_mask][idx_m21] for k in ['infall_mass', 'x', 'y', 'z'])

    x, y, z = cc['x'].copy(), cc['y'].copy(), cc['z'].copy()
    x[satellites_mask] = SHMLM.periodic_bcs(cc['x'][satellites_mask], X, SHMLM.BOXSIZE)
    y[satellites_mask] = SHMLM.periodic_bcs(cc['y'][satellites_mask], Y, SHMLM.BOXSIZE)
    z[satellites_mask] = SHMLM.periodic_bcs(cc['z'][satellites_mask], Z, SHMLM.BOXSIZE)

    srt_fht = np.argsort(cc['fof_halo_tag'])
    cc_srt = { k:cc[k][srt_fht].copy() for k in cc.keys() }
    cc_srt['x'] = x[srt_fht]
    cc_srt['y'] = y[srt_fht]
    cc_srt['z'] = z[srt_fht]
    _, idx, cnts = np.unique(cc_srt['fof_halo_tag'], return_index=True, return_counts=True)

comm.barrier()

if rank == 0:
    numCnts = len(cnts)
else:
    numCnts = None
numCnts = comm.bcast(numCnts, root=0)

if rank != 0:
    idx = np.empty(numCnts, dtype=np.int64)
    cnts = np.empty(numCnts, dtype=np.int64)

comm.Bcast(idx, root=0)
comm.Bcast(cnts, root=0)

comm.barrier()

hostMask = cnts>1
print 'Found {} hosts with over 1 core, I am rank {}.'.format(np.sum(hostMask), rank)

c1, c2 = [], []
indices = np.array_split(np.arange(np.sum(hostMask)), NUMRANKS)[rank]
for i in indices:
    i1, i2 = idx[hostMask][i], idx[hostMask][i] + cnts[hostMask][i]
    c1i, c2i = np.array( list(combinations(range(i1, i2), 2)) ).T #c1, c2 are indices relative to cc_srt
    c1.append(c1i)
    c2.append(c2i)
    if rank == 0:
        print float(i)/len(indices)*100, '%'
comm.barrier()

c1, c2 = np.hstack(c1), np.hstack(c2)
c1 = np.array(c1, dtype=np.int64, order='C')
c2 = np.array(c2, dtype=np.int64, order='C')
send_counts = np.array(comm.gather(len(c1), root=0))
if rank==0:
    recv_c1 = np.empty(np.sum(send_counts), dtype=np.int64)
    recv_c2 = np.empty(np.sum(send_counts), dtype=np.int64)
else:
    recv_c1 = None
    recv_c2 = None

comm.Gatherv( sendbuf=c1, recvbuf=(recv_c1, send_counts), root=0 )
comm.Gatherv( sendbuf=c2, recvbuf=(recv_c2, send_counts), root=0 )

if rank == 0:
    cc_srt['phaseSpaceMerged'] = np.zeros_like(cc_srt['central'])
    Delta_x = ((x[recv_c1]-x[recv_c2])**2 + (y[recv_c1]-y[recv_c2])**2 + (z[recv_c1]-z[recv_c2])**2)**0.5
    Delta_v = ((cc_srt['vx'][recv_c1]-cc_srt['vx'][recv_c2])**2 + (cc_srt['vy'][recv_c1]-cc_srt['vy'][recv_c2])**2 + (cc_srt['vz'][recv_c1]-cc_srt['vz'][recv_c2])**2)**0.5

    mass1bigger = cc_srt['infall_mass'][recv_c1] > cc_srt['infall_mass'][recv_c2]
    massbiggeridx = np.where(mass1bigger, recv_c1, recv_c2)
    masssmalleridx = np.where(mass1bigger, recv_c2, recv_c1)
    sigma_x = 3*cc_srt['radius'][massbiggeridx]
    sigma_v = 3*cc_srt['vel_disp'][massbiggeridx]
    mergeFound = ((Delta_x/sigma_x + Delta_v/sigma_v) < 2)
    cc_srt['phaseSpaceMerged'][np.unique(masssmalleridx[mergeFound])] = 1

    print "Step", step
    print "Merged pairs found: {} out of {} pairs ({}%)".format( np.sum(mergeFound), len(mergeFound), float(np.sum(mergeFound))/len(mergeFound)*100. )
    print "Total number of cores marked as merged:", np.sum(cc_srt['phaseSpaceMerged']==1)

    centrals_mask = cc_srt['central']==1
    print "Number of central cores marked as merged:", np.sum(cc_srt['phaseSpaceMerged'][centrals_mask]==1)

    cc_srt['phaseSpaceMerged'][centrals_mask] = 0

    pickle_save_dict('phasemerge_{}.pkl'.format(step), cc_srt)
comm.barrier()
