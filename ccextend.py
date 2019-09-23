import numpy as np
import genericio as gio
import h5py
import subhalo_mass_loss_model as SHMLM

from tqdm import tqdm # progress bar

cc_data_dir = '/home/isultan/data/AlphaQ/core_catalog/'
cc_output_dir = '/home/isultan/projects/halomassloss/ccextend/output/'

steps = [43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88, 90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121, 124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167, 171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230, 235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315, 323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432, 442, 453, 464, 475, 487, 499]

vars_cc = [
#    'fof_halo_tag',
   'core_tag', 
   'tree_node_index', 
#     'x', 
#     'y', 
#     'z', 
#     'vx', 
#     'vy', 
#     'vz', 
#     'radius', 
    'infall_mass', 
#     'infall_step', 
#    'infall_fof_halo_tag',
#    'infall_tree_node_index',
   'central', 
#    'vel_disp'
]

def write_dict_to_disk(step, cc):
    outfile = cc_output_dir + '09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step)

    f = h5py.File(outfile, 'w')
    grp = f.create_group('coredata')
    for k, v in cc.items():
        grp[k] = v

    f.close()

def fname_cc(step):
    return cc_data_dir + '09_03_2019.AQ.{}.coreproperties'.format(step)

if __name__ == '__main__':
    
    cc = {}
    cc_prev = {}
    
    for step in tqdm(steps):
        # Read in cc for step
        cc = { v:gio.gio_read(fname_cc(step), v)[0] for v in vars_cc }
        
        satellites_mask = cc['central'] == 0
        centrals_mask = cc['central'] == 1
        assert np.sum(satellites_mask) + np.sum(centrals_mask) == len(cc['infall_mass'])
        numSatellites = np.sum(satellites_mask)
        
        # Verify there are no satellites at first step
        if step == steps[0]:
            assert numSatellites == 0, 'Satellites found at first step.'

        # Add column for m_evolved and initialize to 0
        cc['m_evolved'] = np.zeros_like(cc['infall_mass'])
        
        # If there are satellites (not applicable for first step)
        if numSatellites != 0:
            
            # Match satellites to prev step cores using core tag.
            vals, idx1, idx2 = np.intersect1d( cc['core_tag'][satellites_mask], cc_prev['core_tag'], return_indices=True )

            # assert len(vals) == len(cc['core_tag'][satellites_mask]), 'All cores from prev step did not carry over.'
            # assert len(cc['core_tag'][satellites_mask]) == len(np.unique(cc['core_tag'][satellites_mask])), 'Satellite core tags not unique for this time step.'
            
            # Set m_evolved of all satellites that have core tag match on prev step to m_evolved of prev step.
            cc['m_evolved'][satellites_mask][idx1] = cc_prev['m_evolved'][idx2]
            
            # Initialize m array (corresponds with cc[satellites_mask]) to be either infall_mass (step after infall) or m_evolved (subsequent steps).
            m = (cc['m_evolved'][satellites_mask] == 0)*cc['infall_mass'][satellites_mask] + (cc['m_evolved'][satellites_mask] != 0)*cc['m_evolved'][satellites_mask]

            # Match satellites tni with centrals tni.
            vals2, idx3, idx4 = np.intersect1d( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask], return_indices=True)

            # Unique satellites tni array with inverse indices
            vals3, idx_inv = np.unique(cc['tree_node_index'][satellites_mask], return_inverse=True)
            
            assert np.array_equal(vals2, vals3), "All satellites don't have a central match."
            assert np.array_equal(vals3[idx_inv], cc['tree_node_index'][satellites_mask]), 'np.unique inverse indices: original array recreation failure'
            
            assert np.array_equal(cc['tree_node_index'][centrals_mask][idx4], np.sort(cc['tree_node_index'][centrals_mask][idx4])), 'Centrals with satellites: array sorting failure'


            # Initialize M array (corresponds with cc[satellites_mask]) to be host tree node mass of each satellite
            M = cc['infall_mass'][centrals_mask][idx4][idx_inv]
            assert len(M) == len(m) == len(cc['m_evolved'][satellites_mask]), 'M, m, cc[satellites_mask] lengths not equal.'

            # Compute m_evolved of satellites according to SHMLModel.
            cc['m_evolved'][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M, step=step, step_prev=steps[steps.index(step)-1], A=0.86, zeta=0.07)

        
        # write output to disk
        write_dict_to_disk(step, cc)
        
        cc_prev = { k:v.copy() for k,v in cc.items() }
        