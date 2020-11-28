from __future__ import division
import numpy as np
import genericio as gio
import h5py
import subhalo_mass_loss_model_SV as SHMLM
from itk import h5_write_dict, many_to_one, h5_read_dict

from tqdm import tqdm # progress bar

cc_data_dir = SHMLM.cc_data_dir
cc_output_dir = SHMLM.cc_output_dir

# A_arr, zeta_arr = [0.9], [0.005]
#A_arr = [0.4, 0.5, 0.6 , 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
A_arr = [0.4, 0.5, 0.6 , 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
# zeta_arr = [0.001, 0.005, 0.01, 0.02, 0.04, 0.07, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2]
zeta_arr = [0.001, 0.005, 0.01, 0.02, 0.04, 0.07, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]

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
    'infall_tree_node_mass',
    'infall_step',
    'infall_fof_halo_tag',
    'infall_tree_node_index',
    'central',
    'merged',
    'vel_disp',
    'host_core',
    'infall_sod_halo_max_cir_vel',
    'infall_fof_halo_max_cir_vel',
    'infall_fof_halo_angmom_x',
    'infall_fof_halo_angmom_y',
    'infall_fof_halo_angmom_z',
    'infall_sod_halo_angmom_x',
    'infall_sod_halo_angmom_y',
    'infall_sod_halo_angmom_z',
    'infall_sod_halo_mass',
    'infall_fof_halo_mass'
]

def fname_cc(step, mode):
    if mode == 'input':
        return cc_data_dir + 'm000p-{}.coreproperties'.format(step)
    elif mode == 'output':
        return cc_output_dir + 'm000p-{}.corepropertiesextend.hdf5'.format(step)

def m_evolved_col(A, zeta, next=False):
    if next:
        return 'next_m_evolved_{}_{}'.format(A, zeta)
    else:
        return 'm_evolved_{}_{}'.format(A, zeta)

def create_core_catalog_mevolved(virialFlag=False, writeOutputFlag=True, useLocalHost=False, checkpointRestart=None):
    """
    Appends mevolved to core catalog and saves output in HDF5.
    Works  by computing mevolved for step+1 at each step and saving that in memory.
    """
    if writeOutputFlag:
        print 'Reading data from {} and writing output to {}'.format(cc_data_dir, cc_output_dir)
    cc = {}
    cc_prev = {}

    if checkpointRestart is None:
        itersteps = steps
    else:
        itersteps = steps[steps.index(checkpointRestart):]

    for step in tqdm(itersteps):
        # Read in cc for step
        if checkpointRestart is None:
            cc = { v:gio.gio_read(fname_cc(step, 'input'), v)[0] for v in vars_cc }

            if virialFlag:
                # Convert all mass to virial
                cc['infall_tree_node_mass'] = SHMLM.m_vir(cc['infall_tree_node_mass'], step)
        else:
            cc = h5_read_dict(fname_cc(checkpointRestart, 'output'), 'coredata')

        satellites_mask = cc['central'] == 0
        centrals_mask = cc['central'] == 1
        # assert np.sum(satellites_mask) + np.sum(centrals_mask) == len(cc['infall_tree_node_mass']), 'central flag not 0 or 1.'
        numSatellites = np.sum(satellites_mask)

        # Verify there are no satellites at first step
        if step == steps[0]:
            assert numSatellites == 0, 'Satellites found at first step.'

        # Add column for m_evolved and initialize to 0
        if checkpointRestart is None:
            for A in A_arr:
                for zeta in zeta_arr:
                    cc[m_evolved_col(A, zeta)] = np.zeros_like(cc['infall_tree_node_mass'])

        # If there are satellites (not applicable for first step)
        if numSatellites != 0:

            if checkpointRestart is None:
                # Match satellites to prev step cores using core tag.
                vals, idx1, idx2 = np.intersect1d( cc['core_tag'][satellites_mask], cc_prev['core_tag'], return_indices=True )

            # assert len(vals) == len(cc['core_tag'][satellites_mask]), 'All cores from prev step did not carry over.'
            # assert len(cc['core_tag'][satellites_mask]) == len(np.unique(cc['core_tag'][satellites_mask])), 'Satellite core tags not unique for this time step.'

            # Initialize M array (corresponds with cc[satellites_mask]) to be host tree node mass of each satellite
            idx_m21 = many_to_one( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask] )
            M, X, Y, Z = (cc[k][centrals_mask][idx_m21] for k in ['infall_tree_node_mass', 'x', 'y', 'z'])

            if checkpointRestart is None:
                for A in A_arr:
                    for zeta in zeta_arr:
                        # Set m_evolved of all satellites that have core tag match on prev step to next_m_evolved of prev step.
                        # DOUBLE MASK ASSIGNMENT: cc['m_evolved'][satellites_mask][idx1] = cc_prev['next_m_evolved'][idx2]
                        cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[idx1] ] = cc_prev[m_evolved_col(A, zeta, next=True)][idx2]

                        initMask = cc[m_evolved_col(A, zeta)][satellites_mask] == 0
                        minfall = cc['infall_tree_node_mass'][satellites_mask][initMask]
                        cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[initMask] ] = SHMLM.m_evolved(m0=minfall, M0=M[initMask], step=step, step_prev=steps[steps.index(step)-1], A=A, zeta=zeta, dtFactorFlag=True)

        if writeOutputFlag and (checkpointRestart is None):
            # Write output to disk
            h5_write_dict( fname_cc(step, 'output'), cc, 'coredata' )

        #cc should be checkpointRestart step file if not None
        if checkpointRestart is not None:
            checkpointRestart = None

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
        if step != steps[-1]:
            cc_prev = { 'core_tag':cc['core_tag'].copy() }
            if numSatellites != 0:
                localhost_m21 = many_to_one( cc['host_core'][satellites_mask], cc['core_tag'] )

            for A in A_arr:
                for zeta in zeta_arr:
                    cc_prev[m_evolved_col(A, zeta, next=True)] = np.zeros_like(cc['infall_tree_node_mass'])

                    if numSatellites != 0: # If there are satellites (not applicable for first step)
                        m = cc[m_evolved_col(A, zeta)][satellites_mask]
                        if useLocalHost:
                            Mlocal = cc[m_evolved_col(A, zeta)][localhost_m21]
                            M_A_zeta = (Mlocal==0)*M + (Mlocal!=0)*Mlocal
                            cc_prev[m_evolved_col(A, zeta, next=True)][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M_A_zeta, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                        else:
                            cc_prev[m_evolved_col(A, zeta, next=True)][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)

    return cc

if __name__ == '__main__':
    create_core_catalog_mevolved(virialFlag=False, writeOutputFlag=True, useLocalHost=True, checkpointRestart=None)
