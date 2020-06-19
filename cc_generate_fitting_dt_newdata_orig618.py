from __future__ import division
import numpy as np
import genericio as gio
import h5py
from scipy import spatial #KD Tree
from itertools import combinations
import subhalo_mass_loss_model_newdata as SHMLM
from itk import h5_write_dict, many_to_one, h5_read_dict

from tqdm import tqdm # progress bar

cc_data_dir = SHMLM.cc_data_dir
cc_output_dir = SHMLM.cc_output_dir

A_arr, zeta_arr = [0.9], [0.005]

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
        return cc_data_dir + '{}.coreproperties'.format(step)
    elif mode == 'output':
        return cc_output_dir + '{}.corepropertiesextend.hdf5'.format(step)

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
        '''
        #cc['mergedCoreTag'] = np.zeros_like(cc['core_tag'])
        #cc['mergedCoreStep'] = np.zeros_like(cc['infall_step'])
        cc['phaseSpaceMerged'] = np.zeros_like(cc['central'])

        # wasInFragment: persistent flag that is False by default and becomes True when core is inside a fragment halo
        cc['wasInFragment'] = cc['fof_halo_tag']<0
        if step != steps[0]:
            _, i1, i2 = np.intersect1d( cc['core_tag'], cc_prev['core_tag'], return_indices=True )
            cc['wasInFragment'][i1] = np.logical_or( cc['wasInFragment'][i1], cc_prev['wasInFragment'][i2] )
            #cc['mergedCoreTag'][i1] = cc_prev['mergedCoreTag'][i2]
            #cc['mergedCoreStep'][i1] = cc_prev['mergedCoreStep'][i2]
            cc['phaseSpaceMerged'][i1] = cc_prev['phaseSpaceMerged'][i2]
        '''

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

                        # Initialize m array (corresponds with cc[satellites_mask]) to be either infall_mass (step after infall) or m_evolved (subsequent steps).
                        initMask = cc[m_evolved_col(A, zeta)][satellites_mask] == 0
                        minfall = cc['infall_tree_node_mass'][satellites_mask][initMask]
                        cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[initMask] ] = SHMLM.m_evolved(m0=minfall, M0=M[initMask], step=step, step_prev=steps[steps.index(step)-1], A=A, zeta=zeta, dtFactorFlag=True)
                        # m = (cc[m_evolved_col(A, zeta)][satellites_mask] == 0)*cc['infall_tree_node_mass'][satellites_mask] + (cc[m_evolved_col(A, zeta)][satellites_mask] != 0)*cc[m_evolved_col(A, zeta)][satellites_mask]

                        # Set m_evolved of satellites with m_evolved=0 to infall mass.
                        # cc[m_evolved_col(A, zeta)][satellites_mask] = m
            '''
            x, y, z = cc['x'].copy(), cc['y'].copy(), cc['z'].copy()
            x[satellites_mask] = SHMLM.periodic_bcs(cc['x'][satellites_mask], X, SHMLM.BOXSIZE)
            y[satellites_mask] = SHMLM.periodic_bcs(cc['y'][satellites_mask], Y, SHMLM.BOXSIZE)
            z[satellites_mask] = SHMLM.periodic_bcs(cc['z'][satellites_mask], Z, SHMLM.BOXSIZE)
            nonMergeMask = np.flatnonzero(cc['phaseSpaceMerged']==0)

            c1, c2 = np.array( list(combinations(nonMergeMask, 2)) ).T #c1, c2 are indices relative of cc
            Delta_x = ((x[c1]-x[c2])**2 + (y[c1]-y[c2])**2 + (z[c1]-z[c2])**2)**0.5
            Delta_v = ((cc['vx'][c1]-cc['vx'][c2])**2 + (cc['vy'][c1]-cc['vy'][c2])**2 + (cc['vz'][c1]-cc['vz'][c2])**2)**0.5

            mass1bigger = cc['infall_tree_node_mass'][c1] > cc['infall_tree_node_mass'][c2]
            massbiggeridx = np.where(mass1bigger, c1, c2)
            masssmalleridx = np.where(mass1bigger, c2, c1)
            sigma_x = 3*cc['radius'][massbiggeridx]
            sigma_v = 3*cc['vel_disp'][massbiggeridx]
            mergeFound = ((Delta_x/sigma_x + Delta_v/sigma_v) < 2)
            cc['phaseSpaceMerged'][np.unique(masssmalleridx[mergeFound])] = 1

            print "Step", step
            print "Merged pairs found: {} out of {} pairs ({}%)".format( np.sum(mergeFound), len(mergeFound), np.sum(mergeFound)/len(mergeFound)*100. )
            print "Total number of cores marked as merged:", np.sum(cc['phaseSpaceMerged']==1)
            print "Number of central cores marked as merged:", np.sum(cc['phaseSpaceMerged'][centrals_mask]==1)

            cc['phaseSpaceMerged'][centrals_mask] = 0
            '''
        if writeOutputFlag and (checkpointRestart is None):
            # Write output to disk
            h5_write_dict( fname_cc(step, 'output'), cc, 'coredata' )

        #cc should be checkpointRestart step file if not None
        if checkpointRestart is not None:
            checkpointRestart = None

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
       # Mass loss assumed to begin at step AFTER infall detected.
        if step != steps[-1]:
            # cc_prev = { k:cc[k].copy() for k in ['core_tag', 'wasInFragment', 'phaseSpaceMerged'] }
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
