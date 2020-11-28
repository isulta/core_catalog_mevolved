from __future__ import division
import numpy as np
import genericio as gio
import subhalo_mass_loss_model_ALCC as SHMLM
from itk import h5_write_dict, many_to_one
from tqdm import tqdm # progress bar

cc_data_dir = SHMLM.cc_data_dir
cc_output_dir = SHMLM.cc_output_dir

A, zeta = AFID, ZETAFID

steps = SHMLM.steps

vars_cc_all = [
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
    'infall_fof_halo_mass',
    'infall_sod_halo_cdelta'
]
vars_cc_min = [
    'core_tag',
    'tree_node_index',
    'infall_tree_node_mass',
    'central',
    'host_core'
]

def vars_cc(step):
    if step==499 or step==247:
        return vars_cc_all
    else:
        return vars_cc_min

def fname_cc(step, mode):
    if mode == 'input':
        return cc_data_dir + '{}.coreproperties'.format(step)
    elif mode == 'output':
        return cc_output_dir + '{}.corepropertiesextend.hdf5'.format(step)

def m_evolved_col(A, zeta, next=False):
    if next:
        return 'next_m_evolved_{}_{}'.format(A, zeta)
    else:
        return 'm_evolved_{}_{}'.format(A, zeta)

def create_core_catalog_mevolved(writeOutputFlag=True, useLocalHost=False):
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
        cc = { v:gio.gio_read(fname_cc(step, 'input'), v)[0] for v in vars_cc(step) }

        satellites_mask = cc['central'] == 0
        centrals_mask = cc['central'] == 1
        # assert np.sum(satellites_mask) + np.sum(centrals_mask) == len(cc['infall_tree_node_mass']), 'central flag not 0 or 1.'
        numSatellites = np.sum(satellites_mask)

        # Verify there are no satellites at first step
        if step == steps[0]:
            assert numSatellites == 0, 'Satellites found at first step.'

        # Add column for m_evolved and initialize to 0
        cc[m_evolved_col(A, zeta)] = np.zeros_like(cc['infall_tree_node_mass'])

        # If there are satellites (not applicable for first step)
        if numSatellites != 0:

            # Match satellites to prev step cores using core tag.
            vals, idx1, idx2 = np.intersect1d( cc['core_tag'][satellites_mask], cc_prev['core_tag'], return_indices=True )

            idx_m21 = many_to_one( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask] )
            M = cc['infall_tree_node_mass'][centrals_mask][idx_m21]

            # Set m_evolved of all satellites that have core tag match on prev step to next_m_evolved of prev step.
            cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[idx1] ] = cc_prev[m_evolved_col(A, zeta, next=True)][idx2]

            # Initialize m array (corresponds with cc[satellites_mask]) to be either infall_mass (step after infall) or m_evolved (subsequent steps).
            initMask = cc[m_evolved_col(A, zeta)][satellites_mask] == 0
            minfall = cc['infall_tree_node_mass'][satellites_mask][initMask]
            cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[initMask] ] = SHMLM.m_evolved(m0=minfall, M0=M[initMask], step=step, step_prev=steps[steps.index(step)-1], A=A, zeta=zeta, dtFactorFlag=True)
            
        if writeOutputFlag and (step==499 or step==247):
            # Write output to disk
            h5_write_dict( fname_cc(step, 'output'), cc, 'coredata' )

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
       # Mass loss assumed to begin at step AFTER infall detected.
        if step != steps[-1]:
            cc_prev = { 'core_tag':cc['core_tag'].copy() }
            if numSatellites != 0:
                localhost_m21 = many_to_one( cc['host_core'][satellites_mask], cc['core_tag'] )

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
    create_core_catalog_mevolved(writeOutputFlag=True, useLocalHost=True)
