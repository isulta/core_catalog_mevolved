import numpy as np
import genericio as gio
import h5py
import subhalo_mass_loss_model as SHMLM
from itk import h5_write_dict

from tqdm import tqdm # progress bar

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

def fname_cc(step):
    return cc_data_dir + '09_03_2019.AQ.{}.coreproperties'.format(step)

def create_core_catalog_mevolved(virialFlag, A=None, zeta=None, writeOutputFlag=True):
    """
    Appends mevolved to core catalog and saves output in HDF5.
    Works  by computing mevolved for step+1 at each step and saving that in memory.
    """
    if writeOutputFlag:
        print 'Reading data from {} and writing output to {}'.format(cc_data_dir, cc_output_dir)

    cc = {}
    cc_prev = {}
    
    for step in tqdm(steps):
        # Read in cc for step
        cc = { v:gio.gio_read(fname_cc(step), v)[0] for v in vars_cc }
        
        if virialFlag:
            # Convert all mass to virial
            cc['infall_mass'] = SHMLM.m_vir(cc['infall_mass'], step)

        satellites_mask = cc['central'] == 0
        centrals_mask = cc['central'] == 1
        # assert np.sum(satellites_mask) + np.sum(centrals_mask) == len(cc['infall_mass']), 'central flag not 0 or 1.'
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

            # Set m_evolved of all satellites that have core tag match on prev step to next_m_evolved of prev step.
            # DOUBLE MASK ASSIGNMENT: cc['m_evolved'][satellites_mask][idx1] = cc_prev['next_m_evolved'][idx2] 
            cc['m_evolved'][ np.flatnonzero(satellites_mask)[idx1] ] = cc_prev['next_m_evolved'][idx2]

            # Initialize m array (corresponds with cc[satellites_mask]) to be either infall_mass (step after infall) or m_evolved (subsequent steps).
            m = (cc['m_evolved'][satellites_mask] == 0)*cc['infall_mass'][satellites_mask] + (cc['m_evolved'][satellites_mask] != 0)*cc['m_evolved'][satellites_mask]
            # Initialize m array (corresponds with cc[satellites_mask]) to be either virial_infall_mass (step after infall) or m_evolved (subsequent steps).
            #m = (cc['m_evolved'][satellites_mask] == 0)*SHMLM.m_vir(cc['infall_mass'][satellites_mask], step) + (cc['m_evolved'][satellites_mask] != 0)*cc['m_evolved'][satellites_mask]

            # Set m_evolved of satellites with m_evolved=0 to infall mass.
            cc['m_evolved'][satellites_mask] = m

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
         
        
        if writeOutputFlag:
            # Write output to disk
            h5_write_dict( cc_output_dir + '09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), cc, 'coredata' )

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
       # Mass loss assumed to begin at step AFTER infall detected.
        if step != steps[-1]:
            cc_prev = { 'core_tag':cc['core_tag'].copy() }
            cc_prev['next_m_evolved'] = np.zeros_like(cc['infall_mass'])
            
            if numSatellites != 0: # If there are satellites (not applicable for first step)
                cc_prev['next_m_evolved'][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                #cc_prev['next_m_evolved'][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M, step=steps[steps.index(step)+1], step_prev=step, A=23.7, zeta=0.36)

    return cc

if __name__ == '__main__':
    create_core_catalog_mevolved(virialFlag=False)