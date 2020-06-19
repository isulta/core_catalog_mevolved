import numpy as np
import genericio as gio
import h5py
from scipy import spatial #KD Tree
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

from multiprocessing import Pool
import multiprocessing
cc, coresMaskedArray, distance_upper_bound, pool_mergedCoreTag, coresKDTree, KDTree_idx = {}, np.array([]), np.array([]), {}, spatial.cKDTree([[1],[2]]), np.array([])
pool_size = 16

def query(j):
    rank = ((multiprocessing.current_process()._identity[0] - 1) % pool_size) + 1
    for i in j:
        # print multiprocessing.current_process()._identity
        qres = coresKDTree.query_ball_point(coresMaskedArray[i], r=distance_upper_bound[i])
        assert len(qres)>0, 'Merge with self not detected!'
        if len(qres)>1:
            # idxmax = qres[ np.argmax(cc['infall_mass'][satellites_mask][KDTreemask][qres]) ]
            idxmax = qres[ np.argmax(cc['infall_mass'][KDTree_idx][qres]) ]
            
            #qres = np.array(qres)
            #qres2 = qres[qres!=idxmax]
            # cc['mergedCoreTag'][ KDTree_idx[qres2] ] = cc['core_tag'][KDTree_idx][idxmax]
            # cc['mergedCoreStep'][ KDTree_idx[qres2] ] = step
            for iq in qres:
		if iq == idxmax:
			continue
		else:
                	pool_mergedCoreTag[rank][iq] = cc['core_tag'][KDTree_idx][idxmax]
            # pool_mergedCoreTag[rank][qres2] = cc['core_tag'][KDTree_idx][idxmax]
def create_core_catalog_mevolved(virialFlag, writeOutputFlag=True, useLocalHost=False):
    """
    Appends mevolved to core catalog and saves output in HDF5.
    Works  by computing mevolved for step+1 at each step and saving that in memory.
    """
    if writeOutputFlag:
        print 'Reading data from {} and writing output to {}'.format(cc_data_dir, cc_output_dir)
    global cc, coresMaskedArray, distance_upper_bound, pool_mergedCoreTag, coresKDTree, KDTree_idx
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
        
        cc['mergedCoreTag'] = np.zeros_like(cc['core_tag'])
        cc['mergedCoreStep'] = np.zeros_like(cc['infall_step'])

        # wasInFragment: persistent flag that is False by default and becomes True when core is inside a fragment halo
        cc['wasInFragment'] = cc['fof_halo_tag']<0
        if step != steps[0]:
            _, i1, i2 = np.intersect1d( cc['core_tag'], cc_prev['core_tag'], return_indices=True )
            cc['wasInFragment'][i1] = np.logical_or( cc['wasInFragment'][i1], cc_prev['wasInFragment'][i2] )
            cc['mergedCoreTag'][i1] = cc_prev['mergedCoreTag'][i2]
            cc['mergedCoreStep'][i1] = cc_prev['mergedCoreStep'][i2]

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

                    # Set m_evolved of satellites with m_evolved=0 to infall mass.
                    cc[m_evolved_col(A, zeta)][satellites_mask] = m
            
            # Initialize M array (corresponds with cc[satellites_mask]) to be host tree node mass of each satellite
            idx_m21 = many_to_one( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask] )
            M, X, Y, Z = (cc[k][centrals_mask][idx_m21] for k in ['infall_mass', 'x', 'y', 'z'])
            
            KDTreemask = cc['mergedCoreTag'][satellites_mask] == 0
            x = SHMLM.periodic_bcs(cc['x'][satellites_mask][KDTreemask], X[KDTreemask], SHMLM.BOXSIZE)
            y = SHMLM.periodic_bcs(cc['y'][satellites_mask][KDTreemask], Y[KDTreemask], SHMLM.BOXSIZE)
            z = SHMLM.periodic_bcs(cc['z'][satellites_mask][KDTreemask], Z[KDTreemask], SHMLM.BOXSIZE)

            # coresMaskedArray = np.vstack((x, y, z)).T
            
            coresMaskedArray_Tree = np.vstack((np.r_[x, cc['x'][centrals_mask]], np.r_[y, cc['y'][centrals_mask]], np.r_[z, cc['z'][centrals_mask]])).T
            coresMaskedArray = coresMaskedArray_Tree

            KDTree_idx = np.r_[np.flatnonzero(satellites_mask)[KDTreemask], np.flatnonzero(centrals_mask)]
            coresKDTree = spatial.cKDTree(coresMaskedArray_Tree)
            
            # Set mergedCoreTag as central core of KDTreemask cores with no mergedCoreTag that are `merged`
            '''
            disruptMask = cc['merged'][satellites_mask][KDTreemask]|(cc['radius'][satellites_mask][KDTreemask] >= SHMLM.CORERADIUSCUT)
            cc['mergedCoreTag'][ np.flatnonzero(satellites_mask)[KDTreemask] ] += disruptMask*cc['core_tag'][centrals_mask][idx_m21][KDTreemask]
            cc['mergedCoreStep'][ np.flatnonzero(satellites_mask)[KDTreemask] ] += disruptMask*step
            '''

            ### core merging detection
            # rvir_KDTreemask = SHMLM.getRvir(cc[m_evolved_col(1.1, 0.068)][satellites_mask][KDTreemask], SHMLM.step2z[step]) #infall should be exact m_evolved in production
            # rvir_KDTreemask = SHMLM.getRvir(cc['infall_mass'][satellites_mask][KDTreemask], SHMLM.step2z[step]) #infall should be exact m_evolved in production
            # distance_upper_bound = SHMLM.COREVIRIALRADIUSFACTOR * rvir_KDTreemask
            distance_upper_bound = SHMLM.COREVIRIALRADIUSFACTOR * SHMLM.getRvir(cc['infall_mass'][KDTree_idx], SHMLM.step2z[step])
            import ctypes
            
            # pool_mergedCoreTag = { i:np.full(len(KDTree_idx), 0, dtype=np.int64) for i in range(1,pool_size+1) }
            pool_mergedCoreTag = { i:multiprocessing.RawArray(ctypes.c_int64, np.zeros(len(KDTree_idx), dtype=np.int64)) for i in range(1,pool_size+1) }
            # Search radius of 2*rvir around each core
            from datetime import datetime

            tstart = datetime.now()
            work = np.arange(len(distance_upper_bound))
            t1 = datetime.now() - tstart
            print "work=",t1
            tstart = datetime.now()
            p = Pool(pool_size)
            t2 = datetime.now() - tstart
            print "Pool=",t2
            tstart = datetime.now()
            p.map(query, np.array_split(work, pool_size))
            t3 = datetime.now() - tstart
            print "mapp=",t3
            tstart = datetime.now()
            p.close()
            t4 = datetime.now() - tstart
            print "close=",t4
            tstart = datetime.now()
            p.join()
            t5 = datetime.now() - tstart
            print "join=",t5
            tstart = datetime.now()
            
            
            # mergedCoreTag = np.full_like(pool_mergedCoreTag[1], 0)
            mergedCoreTag = np.zeros(len(KDTree_idx), dtype=np.int64)
            
            # pool_mergedCoreTag = {k:np.array(pool_mergedCoreTag[k]) for k in pool_mergedCoreTag.keys()}
            
            for i in range(1,pool_size+1):
                mCT = np.array(pool_mergedCoreTag[i])
                newMergedMask = mCT!=0
                # print np.sum(newMergedMask)
                mergedCoreTag[newMergedMask] = mCT[newMergedMask]
            t6 = datetime.now() - tstart
            newMergedMask = mergedCoreTag != 0
            cc['mergedCoreTag'][ KDTree_idx[newMergedMask] ] = mergedCoreTag[newMergedMask]
            cc['mergedCoreStep'][ KDTree_idx[newMergedMask] ] = step
            print np.sum(cc['mergedCoreTag']!=0)
            t7 = datetime.now() - tstart
            print t1, t2, t3, t4, t5, t6, t7
            ###
            print 'Merged centrals: ', np.sum(cc['mergedCoreTag'][centrals_mask]!=0)
            cc['mergedCoreTag'][centrals_mask] = 0
            cc['mergedCoreStep'][centrals_mask] = 0
            
        if writeOutputFlag:
            # Write output to disk
            h5_write_dict( fname_cc(step, 'output'), cc, 'coredata' )

       # Compute m_evolved of satellites according to SHMLModel for NEXT time step and save as cc_prev['next_m_evolved'] in memory.
       # Mass loss assumed to begin at step AFTER infall detected.
        if step != steps[-1]:
            # cc_prev = { 'core_tag':cc['core_tag'].copy(), 'wasInFragment':cc['wasInFragment'].copy() }
            cc_prev = { k:cc[k].copy() for k in ['core_tag', 'wasInFragment', 'mergedCoreTag', 'mergedCoreStep'] }
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
                            cc_prev[m_evolved_col(A, zeta, next=True)][satellites_mask] = SHMLM.m_evolved(m0=m, M0=M, step=steps[steps.index(step)+1], step_prev=step, A=A, zeta=zeta)

    return cc

if __name__ == '__main__':
    create_core_catalog_mevolved(virialFlag=False, useLocalHost=True)
