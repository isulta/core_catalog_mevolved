''' CHANGES
- many_to_one: verbose=True, assert_x0_unique=False, assert_x1_in_x0=False
- save_cc_prev
- h5distread
- resumeStep
- verbose/timekeeping
- useGPU, intersect1d and many-to-one
'''
from __future__ import division
import numpy as np
import subhalo_mass_loss_model_LastJourney as SHMLM
from itk import h5_write_dict, h5_read_dict, many_to_one, many_to_one_GPU, intersect1d_GPU
from tqdm import tqdm # progress bar
import os
import glob
import time

cc_data_dir = SHMLM.cc_data_dir
cc_output_dir = SHMLM.cc_output_dir

A, zeta = SHMLM.AFID, SHMLM.ZETAFID

steps = SHMLM.steps

vars_cc_all = [
    'fof_halo_tag',
    'core_tag',
    'tree_node_index',
    'infall_tree_node_mass',
    'central',
    'host_core'
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
        return cc_data_dir + 'm000p-{}'.format(step)
    elif mode == 'output':
        return cc_output_dir + 'm000p-{}.corepropertiesextend.hdf5'.format(step)
    elif mode=='output_ccprev':
        return cc_output_dir + 'm000p-{}.ccprev.hdf5'.format(step)

def h5distread(step):
    basename = fname_cc(step, 'input')
    ccall = {k:[] for k in vars_cc_min}
    for f in glob.glob(basename+'#*'):
        ccf = h5_read_dict(f)
        for k in vars_cc_min:
            ccall[k].append(ccf[k])
    return { k:np.concatenate(ccall[k]) for k in vars_cc_min }

def m_evolved_col(A, zeta, next=False):
    if next:
        return 'next_m_evolved_{}_{}'.format(A, zeta)
    else:
        return 'm_evolved_{}_{}'.format(A, zeta)

def create_core_catalog_mevolved(writeOutputFlag=True, useLocalHost=False, save_cc_prev=False, resumeStep=None, useGPU=False):
    """
    Appends mevolved to core catalog and saves output in HDF5.
    Works  by computing mevolved for step+1 at each step and saving that in memory.
    """
    if writeOutputFlag:
        print('Reading data from {} and writing output to {}'.format(cc_data_dir, cc_output_dir))
    cc = {}
    cc_prev = {}

    for step in tqdm(steps):
        # Start at step `resumeStep`. Assumes code finished running and saved ccprev for step `resumeStep-1`.
        if resumeStep is not None:
            if step<resumeStep:
                continue
            elif step==resumeStep:
                prevstep = steps[steps.index(step)-1]
                fname_ccprev_prevstep = fname_cc(prevstep, 'output_ccprev')
                print(f'Resuming at step {step} using {fname_ccprev_prevstep}.'); start=time.time()
                cc_prev = h5_read_dict(fname_ccprev_prevstep, 'coredata')
                cc_prev['core_tag'] = h5distread(prevstep)['core_tag']
                print(f'Finished reading {fname_ccprev_prevstep} and input hdf5 in {time.time()-start} seconds.')

        print(f'Beginning step {step}. Reading hdf5 files..'); start = time.time()
        # Read in cc for step
        cc = h5distread(step)
        print(f'Finished reading hdf5 files in {time.time()-start} seconds.')

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
            print('intersect1d cc to cc_prev...'); start=time.time()
            if useGPU:
                vals, idx1, idx2 = intersect1d_GPU( cc['core_tag'][satellites_mask], cc_prev['core_tag'], return_indices=True )
            else:
                vals, idx1, idx2 = np.intersect1d( cc['core_tag'][satellites_mask], cc_prev['core_tag'], return_indices=True )
            print(f'Matches found between cc and cc_prev: {len(vals)}.')
            print(f'Finished intersect1d cc to cc_prev in {time.time()-start} seconds.')

            print('satellites:centrals m21...'); start=time.time()
            if useGPU:
                idx_m21 = many_to_one_GPU( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask] )
            else:
                idx_m21 = many_to_one( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask], verbose=True, assert_x0_unique=False, assert_x1_in_x0=False )
            print(f'Finished satellites:centrals m21 in {time.time()-start} seconds.')
            M = cc['infall_tree_node_mass'][centrals_mask][idx_m21]

            # Set m_evolved of all satellites that have core tag match on prev step to next_m_evolved of prev step.
            cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[idx1] ] = cc_prev[m_evolved_col(A, zeta, next=True)][idx2]
            
            # Initialize m array (corresponds with cc[satellites_mask]) to be either infall_mass (step after infall) or m_evolved (subsequent steps).
            print('SHMLM (new satellites)...'); start=time.time()
            initMask = cc[m_evolved_col(A, zeta)][satellites_mask] == 0
            minfall = cc['infall_tree_node_mass'][satellites_mask][initMask]
            cc[m_evolved_col(A, zeta)][ np.flatnonzero(satellites_mask)[initMask] ] = SHMLM.m_evolved(m0=minfall, M0=M[initMask], step=step, step_prev=steps[steps.index(step)-1], A=A, zeta=zeta, dtFactorFlag=True)
            print(f'Finished SHMLM (new satellites) in {time.time()-start} seconds.')

        if writeOutputFlag and (step==499 or step==247):
            # Write output to disk
            cc_save = {
                # 'fof_halo_tag':cc['fof_halo_tag'],
                # 'central':cc['central'],
                m_evolved_col(A, zeta):cc[m_evolved_col(A, zeta)]
                # 'M':M
            }
            h5_write_dict( fname_cc(step, 'output'), cc_save, 'coredata' )

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
       # Mass loss assumed to begin at step AFTER infall detected.
        if step != steps[-1]:
            cc_prev = { 'core_tag':cc['core_tag'].copy() }
            if numSatellites != 0:
                print('host_core:core_tag m21...'); start=time.time()
                if useGPU:
                    localhost_m21 = many_to_one_GPU( cc['host_core'][satellites_mask], cc['core_tag'] )
                else:
                    localhost_m21 = many_to_one( cc['host_core'][satellites_mask], cc['core_tag'], verbose=True, assert_x0_unique=False, assert_x1_in_x0=False )
                print(f'Finished host_core:core_tag m21 in {time.time()-start} seconds.')

            cc_prev[m_evolved_col(A, zeta, next=True)] = np.zeros_like(cc['infall_tree_node_mass'])

            if numSatellites != 0: # If there are satellites (not applicable for first step)
                print('SHMLM (next step)...'); start=time.time()
                m = cc[m_evolved_col(A, zeta)][satellites_mask]
                if useLocalHost:
                    Mlocal = cc[m_evolved_col(A, zeta)][localhost_m21]
                    M_A_zeta = (Mlocal==0)*M + (Mlocal!=0)*Mlocal
                    cc_prev[m_evolved_col(A, zeta, next=True)][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M_A_zeta, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                else:
                    cc_prev[m_evolved_col(A, zeta, next=True)][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)
                print(f'Finished SHMLM (next step) in {time.time()-start} seconds.')
            if save_cc_prev:
                print('writing ccprev hdf5...'); start=time.time()
                h5_write_dict( fname_cc(step, 'output_ccprev'), {m_evolved_col(A, zeta, next=True):cc_prev[m_evolved_col(A, zeta, next=True)]}, 'coredata' )
                # if os.path.exists( fname_cc(steps[steps.index(step)-1], 'output_ccprev') ):
                #     os.remove( fname_cc(steps[steps.index(step)-1], 'output_ccprev') )
                print(f'Finished writing ccprev hdf5 in {time.time()-start} seconds.')

    return cc

if __name__ == '__main__':
    create_core_catalog_mevolved(writeOutputFlag=True, useLocalHost=True, save_cc_prev=True, resumeStep=442, useGPU=False)
