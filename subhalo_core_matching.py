from __future__ import division
from itk import gio_read_dict, hist, plt_latex, h5_read_dict, many_to_one
import numpy as np
import matplotlib.pyplot as plt
import os
import subhalo_mass_loss_model as SHMLM
from tqdm import trange, tqdm
plt_latex()

# data_dir = '/data/a/cpac/erangel/AlphaQ/output/'
# data_dir = '/data/a/cpac/dkorytov/data/AlphaQ/merger_trees/'
# data_dir = '/home/isultan/data/AlphaQ/MergerTrees_bestMatch_cooley/'
# data_dir = '/home/isultan/data/AlphaQ/cooley_MergerTrees_updated/'

# data_dir = '/home/isultan/data/AlphaQ/cooley_MergerTrees_updated_rev4070/'
# steps = sorted([int(f.split('.')[0]) for f in os.listdir(data_dir) if '#' not in f])
# fname = lambda s : data_dir + '{}.treenodes'.format(s)

data_dir = '/home/isultan/data/AlphaQ/updated_tree_nodes/'
steps = sorted([int(f.split('.')[2]) for f in os.listdir(data_dir) if '#' not in f])
fname_mt = lambda s : data_dir + '09_03_2019.AQ.{}.treenodes'.format(s)
# steps = [44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88, 90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121, 124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167, 171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230, 235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315, 323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432, 442, 453, 464, 475, 487, 499]

steps=steps[steps.index(163)+1:]
fname_sh = lambda s: '/home/isultan/data/AlphaQ/subhalos/m000-{}.subhaloproperties'.format(s)
fname_cc = lambda s: '/home/isultan/data/AlphaQ/core_catalog_merg/09_03_2019.AQ.{}.coreproperties'.format(s)

cols_mt = ['tree_node_index', 'desc_node_index', 'fof_halo_tag', 'fof_halo_mass', 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z']#'tree_node_mass']
cols_sh = ['fof_halo_tag', 'subhalo_tag', 'subhalo_count', 'subhalo_mean_x','subhalo_mean_y','subhalo_mean_z','subhalo_mean_vx', 'subhalo_mean_vy', 'subhalo_mean_vz', 'subhalo_mass']
cols_cc = ['fof_halo_tag', 'x', 'y', 'z', 'radius', 'infall_mass', 'central', 'host_core', 'core_tag', 'merged', 'tree_node_index']

from scipy import spatial #KD Tree for subhalo-core matching
# fn='/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof/09_03_2019.AQ.499.corepropertiesextend.hdf5'
# cc = h5_read_dict(fn, 'coredata')
# sh = gio_read_dict( fname_sh(499), cols_sh )
# mt = gio_read_dict( fname(499), cols_mt )

def generate_cores_kdtree(cc, M1, M2, s1=False, disrupt=None, onlyFiltered=False, giveFragmentsFofMass=False, z=0):
    idx_filteredsatcores, M, X, Y, Z, nHalo = SHMLM.core_mask(cc, M1=M1, M2=M2, s1=s1, disrupt=disrupt, z=z)
    cc_filtered = { k:cc[k][idx_filteredsatcores].copy() for k in cc.keys() }
    cc_filtered['M'] = M
    if giveFragmentsFofMass:
        fragmask = cc_filtered['fof_halo_tag']<0
        frag_realfht = np.bitwise_and(cc_filtered['fof_halo_tag'][fragmask]*-1, 0xffffffffffff)
        idx_m21 = many_to_one( frag_realfht, mt_cores['fof_halo_tag'] )
        cc_filtered['M'][fragmask] = mt_cores['fof_halo_mass'][idx_m21]

    cc_filtered['x'] = SHMLM.periodic_bcs(cc_filtered['x'], X, SHMLM.BOXSIZE)
    cc_filtered['y'] = SHMLM.periodic_bcs(cc_filtered['y'], Y, SHMLM.BOXSIZE)
    cc_filtered['z'] = SHMLM.periodic_bcs(cc_filtered['z'], Z, SHMLM.BOXSIZE)
    Mmask = (M1 <= cc_filtered['M']) & (cc_filtered['M'] <= M2)
    cc_filtered = { k:cc_filtered[k][Mmask].copy() for k in cc_filtered.keys() }
    if onlyFiltered:
        return cc_filtered
    else:
        return spatial.cKDTree( np.vstack((cc_filtered['x'], cc_filtered['y'], cc_filtered['z'])).T ), cc_filtered, nHalo

def sh_cores_match(cc, sh, mt, z):
    cores_tree, cc_filtered, _ = generate_cores_kdtree(cc, M1=10**9, M2=10**16, s1=False, disrupt=None, z=z)

    sh['rvir'] = SHMLM.getRvir(sh['subhalo_mass'], z)

    idx_m21_sh = many_to_one(sh['fof_halo_tag'], mt['fof_halo_tag'])
    sh['M'] = mt['fof_halo_mass'][idx_m21_sh]

    sh['X'] = mt['fof_halo_center_x'][idx_m21_sh]
    sh['Y'] = mt['fof_halo_center_y'][idx_m21_sh]
    sh['Z'] = mt['fof_halo_center_z'][idx_m21_sh]

    subhalo_mean_x = SHMLM.periodic_bcs(sh['subhalo_mean_x'], sh['X'], SHMLM.BOXSIZE)
    subhalo_mean_y = SHMLM.periodic_bcs(sh['subhalo_mean_y'], sh['Y'], SHMLM.BOXSIZE)
    subhalo_mean_z = SHMLM.periodic_bcs(sh['subhalo_mean_z'], sh['Z'], SHMLM.BOXSIZE)

    sh_mask = (sh['subhalo_tag']!=0)
    sh_arr = np.vstack((subhalo_mean_x[sh_mask], subhalo_mean_y[sh_mask], subhalo_mean_z[sh_mask])).T
    distance_upper_bound = 2 * sh['rvir'][sh_mask]

    # Search radius of 2*rvir around each subhalo, only look for 1:1 matches

    dist, idx = [], []
    for i in range(len(distance_upper_bound)):
        dv, iv = cores_tree.query(sh_arr[i], k=2, distance_upper_bound=distance_upper_bound[i])
        dist.append(dv)
        idx.append(iv)
    dist = np.array(dist)
    idx = np.array(idx)

    f1, f2 = (dist != np.inf).T
    fmask = f1^f2

    percentmatch = np.sum(fmask)/np.sum(sh_mask)*100
    f1i = f1[np.invert(fmask)]
    percentmany = np.sum(f1i)/len(f1i)*100
    percentnone = np.sum(np.invert(f1i))/len(f1i)*100
    print '{}% of masked subhalos have 1:1 core match. Of the unmatched subhalos, {}% have mutliple cores and {}% have no core.'.format(percentmatch, percentmany, percentnone)

    return np.flatnonzero(sh_mask)[fmask], idx[:,0][fmask], cc_filtered
    #return sh indicies, cc_filtered indices, cc_filtered
    # sh['subhalo_mass'][sh_mask][fmask]
    # cc_filtered['infall_mass'][idx[:,0][fmask]]

# not: sum(Mp)/MN for every node N with at least 1 progenitor p, in mass bins of MN
# not: sum(Mp)/MN for every node N with at least 1 fragment progenitor p, in mass bins of MN

# sumMp = np.array([], dtype=np.float32)
# MN = np.array([], dtype=np.float32)
# hasFragment = np.array([], dtype=bool)
countStep = np.array([], dtype=np.int32)
msh_prog = np.array([], dtype=np.float32)
msh_desc = np.array([], dtype=np.float32)
M_host = np.array([], dtype=np.float32)

countHaloMatches = np.array([], dtype=np.int32)
count2Subhalo = np.array([], dtype=np.int32)

mt = None
sh = None
cc = None
sh_indices, cc_filtered_indices, cc_filtered = None, None, None
for s, s_next in tqdm(zip(steps[:-1], steps[1:]), total=len(steps)-1):
    #readin
    if not mt:
        mt = gio_read_dict( fname_mt(s), cols_mt )
        sh = gio_read_dict( fname_sh(s), cols_sh )
        cc = gio_read_dict( fname_cc(s), cols_cc )
        sh_indices, cc_filtered_indices, cc_filtered = sh_cores_match(cc, sh, mt, z=SHMLM.step2z[s])

    mt_next = gio_read_dict( fname_mt(s_next), cols_mt )
    sh_next = gio_read_dict( fname_sh(s_next), cols_sh )
    cc_next = gio_read_dict( fname_cc(s_next), cols_cc )
    sh_indices_next, cc_filtered_indices_next, cc_filtered_next = sh_cores_match(cc_next, sh_next, mt_next, z=SHMLM.step2z[s_next])

    _, matchcoresidx, matchcoresidx_next = np.intersect1d(cc_filtered['core_tag'][cc_filtered_indices], cc_filtered_next['core_tag'][cc_filtered_indices_next], assume_unique=False, return_indices=True)
    countStep = np.append( countStep, len(matchcoresidx) )
    print 'Matches found:', countStep[-1], 'on step', s

    M_host = np.r_[ M_host, sh['M'][sh_indices][matchcoresidx] ]

    msh_prog = np.r_[ msh_prog, sh['subhalo_count'][sh_indices][matchcoresidx] ]
    msh_desc = np.r_[ msh_desc, sh_next['subhalo_count'][sh_indices_next][matchcoresidx_next] ]
#     fragmask = mt['fof_halo_tag']<0
    '''
    _, idx_un, cnt_un = np.unique(mt['desc_node_index'], return_index=True, return_counts=True)
    isin = np.intersect1d( np.flatnonzero(np.isin(mt['desc_node_index'], mt_next['tree_node_index'])), idx_un[cnt_un==1] )
    print 'isin1:', len(isin)

    arst = np.argsort(mt['desc_node_index'][isin])
    mt_prog = { k:mt[k][isin][arst] for k in cols_mt }

    vals, idx1, idx2 = np.intersect1d(mt_prog['desc_node_index'], mt_next['tree_node_index'], assume_unique=False, return_indices=True)
    assert len(vals) == len(mt_prog['desc_node_index'])

    sh_satmask = np.flatnonzero(sh['subhalo_tag']!=0)
    _, idx_un2, cnt_un2 = np.unique(sh['fof_halo_tag'][sh_satmask], return_index=True, return_counts=True)
    sh_satmask = sh_satmask[idx_un2[cnt_un2==1]]

    sh_next_satmask = np.flatnonzero(sh_next['subhalo_tag']!=0)
    _, idx_un3, cnt_un3 = np.unique(sh_next['fof_halo_tag'][sh_next_satmask], return_index=True, return_counts=True)
    sh_next_satmask = sh_next_satmask[idx_un3[cnt_un3==1]]

    # checks to see if all fhts are unique
    assert len(np.unique(sh['fof_halo_tag'][sh_satmask])) == len(sh_satmask)
    assert len(np.unique(sh_next['fof_halo_tag'][sh_next_satmask])) == len(sh_next_satmask)

    assert len(np.unique(mt_prog['fof_halo_tag'][idx1])) == len(idx1)
    assert len(np.unique(mt_next['fof_halo_tag'][idx2])) == len(idx2)

    _, idx3, idx4 = np.intersect1d(mt_prog['fof_halo_tag'][idx1], sh['fof_halo_tag'][sh_satmask], assume_unique=True, return_indices=True)
    _, idx5, idx6 = np.intersect1d(mt_next['fof_halo_tag'][idx2], sh_next['fof_halo_tag'][sh_next_satmask], assume_unique=True, return_indices=True)
    vals2, idx7, idx8 = np.intersect1d(idx3, idx5, assume_unique=True, return_indices=True)

    assert len(mt_prog['fof_halo_tag'][idx1][vals2]) == len(mt_next['fof_halo_tag'][idx2][vals2]) == len(sh['fof_halo_tag'][sh_satmask][idx4][idx7]) == len(sh_next['fof_halo_tag'][sh_next_satmask][idx6][idx8])
    assert np.array_equal(mt_prog['fof_halo_tag'][idx1][vals2], sh['fof_halo_tag'][sh_satmask][idx4][idx7])
    assert np.array_equal(mt_next['fof_halo_tag'][idx2][vals2], sh_next['fof_halo_tag'][sh_next_satmask][idx6][idx8])

    M_host = np.r_[ M_host, mt_prog['fof_halo_mass'][idx1][vals2] ]

    msh_prog = np.r_[ msh_prog, sh['subhalo_count'][sh_satmask][idx4][idx7] ]
    msh_desc = np.r_[ msh_desc, sh_next['subhalo_count'][sh_next_satmask][idx6][idx8] ]

    countStep = np.append( countStep, len(mt_prog['fof_halo_tag'][idx1][vals2]) )
    print countStep[-1]

    countHaloMatches = np.append( countHaloMatches, len(idx1) )
    count2Subhalo = np.append( count2Subhalo, len(sh_satmask) )
    '''

    mt = { k:mt_next[k].copy() for k in mt_next.keys() }
    sh = { k:sh_next[k].copy() for k in sh_next.keys() }
    cc = { k:cc_next[k].copy() for k in cc_next.keys() }
    sh_indices, cc_filtered_indices, cc_filtered = sh_indices_next.copy(), cc_filtered_indices_next.copy(), { k:cc_filtered_next[k].copy() for k in cc_filtered_next.keys() }
    import sys
    sys.stdout.flush()
