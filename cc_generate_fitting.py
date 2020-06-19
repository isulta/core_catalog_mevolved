import numpy as np
import genericio as gio
import h5py
import subhalo_mass_loss_model as SHMLM
from itk import h5_write_dict, many_to_one

from tqdm import tqdm # progress bar

cc_data_dir = SHMLM.cc_data_dir
cc_output_dir = SHMLM.cc_output_dir

# A_arr = [0.25,0.5,1.0,1.2,1.4,1.5,1.6,2.0,4.0,10]
# zeta_arr = [0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.2]
A_arr = [0.8, 0.9, 1.0 , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
# zeta_arr = [0.01, 0.039, 0.068, 0.097, 0.126, 0.155, 0.184, 0.213, 0.242, 0.271, 0.3]
zeta_arr = [0.01, 0.02, 0.039, 0.068, 0.08, 0.097, 0.126, 0.155, 0.184, 0.213, 0.242, 0.271, 0.3]

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

def m_evolved_col(A, zeta, next=False):
    if next:
        return 'next_m_evolved_{}_{}'.format(A, zeta)
    else:
        return 'm_evolved_{}_{}'.format(A, zeta)

def create_core_catalog_mevolved(virialFlag, writeOutputFlag=True, useLocalHost=False):
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
        cc = { v:gio.gio_read(fname_cc(step, 'input'), v)[0] for v in vars_cc }

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
        for A in A_arr:
            for zeta in zeta_arr:
                cc[m_evolved_col(A, zeta)] = np.zeros_like(cc['infall_mass'])

        # If there are satellites (not applicable for first step)
        if numSatellites != 0:

            # Match satellites to prev step cores using core tag.
            vals, idx1, idx2 = np.intersect1d( cc['core_tag'][satellites_mask], cc_prev['core_tag'], return_indices=True )

            # assert len(vals) == len(cc['core_tag'][satellites_mask]), 'All cores from prev step did not carry over.'
            # assert len(cc['core_tag'][satellites_mask]) == len(np.unique(cc['core_tag'][satellites_mask])), 'Satellite core tags not unique for this time step.'

            for A in A_arr:
                for zeta in zeta_arr:
                    # Set m_evolved of all satellites that have core tag match on prev step to next_m_evolved of prev step.
                    # DOUBLE MASK ASSIGNMENT: cc['m_evolved'][satellites_mask][idx1] = cc_prev['next_m_evolved'][idx2]
                    cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[idx1] ] = cc_prev[m_evolved_col(A, zeta, next=True)][idx2]

                    # Initialize m array (corresponds with cc[satellites_mask]) to be either infall_mass (step after infall) or m_evolved (subsequent steps).
                    m = (cc[m_evolved_col(A, zeta)][satellites_mask] == 0)*cc['infall_mass'][satellites_mask] + (cc[m_evolved_col(A, zeta)][satellites_mask] != 0)*cc[m_evolved_col(A, zeta)][satellites_mask]
                    # Initialize m array (corresponds with cc[satellites_mask]) to be either virial_infall_mass (step after infall) or m_evolved (subsequent steps).
                    #m = (cc[m_evolved_col(A, zeta)][satellites_mask] == 0)*SHMLM.m_vir(cc['infall_mass'][satellites_mask], step) + (cc[m_evolved_col(A, zeta)][satellites_mask] != 0)*cc[m_evolved_col(A, zeta)][satellites_mask]

                    # Set m_evolved of satellites with m_evolved=0 to infall mass.
                    cc[m_evolved_col(A, zeta)][satellites_mask] = m

            # Initialize M array (corresponds with cc[satellites_mask]) to be host tree node mass of each satellite
            M = cc['infall_mass'][centrals_mask][ many_to_one( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask] ) ]
            # assert len(M) == len(m) == len(cc['m_evolved'][satellites_mask]), 'M, m, cc[satellites_mask] lengths not equal.'

        # wasInFragment: persistent flag that is False by default and becomes True when core is inside a fragment halo
        cc['wasInFragment'] = cc['fof_halo_tag']<0
        if step != steps[0]:
            _, i1, i2 = np.intersect1d( cc['core_tag'], cc_prev['core_tag'], return_indices=True )
            cc['wasInFragment'][i1] = np.logical_or( cc['wasInFragment'][i1], cc_prev['wasInFragment'][i2] )

        if writeOutputFlag:
            # Write output to disk
            h5_write_dict( fname_cc(step, 'output'), cc, 'coredata' )

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
       # Mass loss assumed to begin at step AFTER infall detected.
        if step != steps[-1]:
            cc_prev = { 'core_tag':cc['core_tag'].copy(), 'wasInFragment':cc['wasInFragment'].copy() }
            if numSatellites != 0:
                localhost_m21 = many_to_one( cc['host_core'][satellites_mask], cc['core_tag'] )

            for A in A_arr:
                for zeta in zeta_arr:
                    cc_prev[m_evolved_col(A, zeta, next=True)] = np.zeros_like(cc['infall_mass'])

                    if numSatellites != 0: # If there are satellites (not applicable for first step)
                        m = cc[m_evolved_col(A, zeta)][satellites_mask]
                        if useLocalHost:
                            Mlocal = cc[m_evolved_col(A, zeta)][localhost_m21]
                            M_A_zeta = (Mlocal==0)*M + (Mlocal!=0)*Mlocal
                            cc_prev[m_evolved_col(A, zeta, next=True)][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M_A_zeta, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                        else:
                            cc_prev[m_evolved_col(A, zeta, next=True)]][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M, step=steps[steps.index(step)+1], step_prev=step, A=23.7, zeta=0.36)

    return cc

if __name__ == '__main__':
    create_core_catalog_mevolved(virialFlag=False, useLocalHost=True)
