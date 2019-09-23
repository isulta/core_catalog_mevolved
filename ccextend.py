import numpy as np
import genericio as gio
import h5py
from tqdm import tqdm #progress bar

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

def write_cc_to_disk(step, cc):
    outfile = cc_output_dir + '09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step)

    f = h5py.File(outfile, 'w')
    grp = f.create_group('coredata')
    for k, v in cc.items():
        grp[k] = v

    f.close()

def fname_cc(step):
    return cc_data_dir + '09_03_2019.AQ.{}.coreproperties'.format(step)

if __name__ == '__main__':
    
    for step in tqdm(steps[:20]):
        # read in cc for step
        cc = { v:gio.gio_read(fname_cc(step), v)[0] for v in vars_cc }
        
        noncentrals_mask = cc['central'] != 1
        numSatellites = np.sum(noncentrals_mask)
        
        # verify there are no satellites at first step
        if step == steps[0]:
            assert numSatellites == 0

        # add column for m_evolved and initialize to 0
        cc['m_evolved'] = np.zeros_like(cc['infall_mass'])
        
        # if there are satellites (not applicable for first step)
        if numSatellites != 0:
            pass
        
        # write output to disk
        write_cc_to_disk(step, cc)