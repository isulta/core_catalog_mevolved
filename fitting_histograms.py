from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subhalo_mass_loss_model as SHMLM
import genericio as gio
from tqdm import tqdm
from itk import hist, h5_read_dict, gio_read_dict, loadpickle, plt_latex, periodic_bcs, many_to_one
from cc_generate_fitting import m_evolved_col, A_arr, zeta_arr
from scipy import spatial #KD Tree for subhalo-core matching
from scipy.stats import norm # Gaussian fitting
import itertools as it

step = 499
z = SHMLM.step2z[step]
cc = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting/09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), 'coredata')
sh = gio_read_dict('/home/isultan/data/AlphaQ/subhalos/m000-{}.subhaloproperties'.format(step), ['fof_halo_tag','subhalo_mean_x','subhalo_mean_y','subhalo_mean_z','subhalo_mean_vx', 'subhalo_mean_vy', 'subhalo_mean_vz', 'subhalo_count', 'subhalo_tag', 'subhalo_mass'])
mt = gio_read_dict('/home/isultan/data/AlphaQ/updated_tree_nodes/09_03_2019.AQ.{}.treenodes'.format(step), ['fof_halo_tag', 'fof_halo_mass'])

M1, M2 = 10**9., 10**15.5

# Subhalo-core matching: to look at the subhalo mass:core evolved mass ratio
def generate_cores_kdtree(M1, M2, s1=False, disrupt=None):
    idx_filteredsatcores, M, nHalo = SHMLM.core_mask(cc, M1, M2, s1=s1, disrupt=disrupt)
    cc_filtered = { k:cc[k][idx_filteredsatcores].copy() for k in cc.keys() }
    cc_filtered['M'] = M
    return spatial.cKDTree( np.vstack((cc_filtered['x'], cc_filtered['y'], cc_filtered['z'])).T ), cc_filtered, nHalo

# def residual(A, zeta, plotFlag=False):
#     """Returns Euclidean norm of subhalo and core mass functions, where only subhalos and cores in 1:1 match are considered."""
#     # residual plots
#     print len(sh['subhalo_mass'][sh_mask][fmask])
#     print len(cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]])
#     realmask = cc_filtered['fof_halo_tag'][idx[:,0][fmask]]>=0
#     shmf = sh['subhalo_mass'][sh_mask][fmask][realmask] / sh['M'][sh_mask][fmask][realmask]
#     r = (-3, 0)
#     r_res = (-3, -2)
    
#     sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=True, normCnts=False, normLogCnts=True, normScalar=nHalo, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)
#     res_mask = (r_res[0]<=sh_xarr)&(sh_xarr<=r_res[1])

#     cmf = cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]][realmask] / cc_filtered['M'][idx[:,0][fmask]][realmask]
#     cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=nHalo, normCnts=False, normBinsize=True, normLogCnts=True)
    
#     return np.linalg.norm((sh_cnts-cores_cnts)[res_mask])

def residual(A, zeta, plotFlag=False):
    """Returns Euclidean norm of subhalo and core mass functions, where only subhalos and cores in 1:1 match are considered."""
    # residual plots
    print len(sh['subhalo_mass'][sh_mask][fmask])
    print len(cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]])

    shmf = sh['subhalo_mass'][sh_mask][fmask] / cc_filtered['M'][idx[:,0][fmask]]#sh['M'][sh_mask][fmask]
    r = (-3, 0)
    r_res = (-3, -2)
    
    sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=True, normCnts=False, normLogCnts=True, normScalar=nHalo, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)
    res_mask = (r_res[0]<=sh_xarr)&(sh_xarr<=r_res[1])

    cmf = cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]] / cc_filtered['M'][idx[:,0][fmask]]
    cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=nHalo, normCnts=False, normBinsize=True, normLogCnts=True)
    
    return np.linalg.norm((sh_cnts-cores_cnts)[res_mask])

"""Save histograms"""
# for A, zeta in tqdm(it.product(A_arr, zeta_arr)):
#     plt.figure()
#     plt.hist((sh['subhalo_mass'][sh_mask][fmask] - cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]])/sh['subhalo_mass'][sh_mask][fmask], range=(-5,5), bins=100)#, alpha=.5)
#     plt.axvline(x=0, c='k')
#     plt.title( 'z=0, {} $\leq$ log(M/$h^{{-1}}M_\odot$)$\leq$ {}'.format(np.log10(M1),np.log10(M2)) )
#     plt.savefig( 'hists/hist_{}_{}_A_{}_zeta_{}.png'.format(int(np.log10(M1)), int(np.log10(M2)), A, zeta) )
#     plt.close()

"""2 norm of sh-core 1:1 matches """
# M = np.empty((len(A_arr), len(zeta_arr)))
# for i in range(len(A_arr)):
#     for j in range(len(zeta_arr)):
#         A, zeta = A_arr[i], zeta_arr[j]
#         # M[i,j] = np.polyfit( sh['subhalo_mass'][sh_mask][fmask], cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]], 1 )[0]
#         # M[i,j] = np.linalg.norm(sh['subhalo_mass'][sh_mask][fmask] - cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]])
#         M[i,j] = residual(A, zeta)
#         print M[i,j]
# plt.pcolormesh(A_arr, zeta_arr, M)
# plt.colorbar()
# plt.show()

"""Mean/std plot"""
# plt.figure()
# mns = []
# stds = []

# for A, zeta in it.product(A_arr, zeta_arr):
#     data = (sh['subhalo_mass'][sh_mask][fmask] - cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]])/sh['subhalo_mass'][sh_mask][fmask]
#     mean,std=norm.fit(data)
#     mns.append(mean)
#     stds.append(std)
#     # plt.text(mean, std, '({}, {})'.format(A, zeta), fontsize=10)

# plt.scatter(mns, stds, s=1)
# plt.show()

if __name__ == "__main__":
    plt_latex()
    cores_tree, cc_filtered, nHalo = generate_cores_kdtree(M1=M1, M2=M2, s1=False, disrupt=None)

    sh['rvir'] = SHMLM.getRvir(sh['subhalo_mass'], z)

    idx_m21_sh = many_to_one(sh['fof_halo_tag'], mt['fof_halo_tag'])
    sh['M'] = mt['fof_halo_mass'][idx_m21_sh]
    sh_mask = (sh['subhalo_tag']!=0)&(M1<=sh['M'])&(sh['M']<=M2)

    sh_arr = np.vstack((sh['subhalo_mean_x'][sh_mask], sh['subhalo_mean_y'][sh_mask], sh['subhalo_mean_z'][sh_mask])).T
    distance_upper_bound = 2 * sh['rvir'][sh_mask]

    # Search radius of 2*rvir around each subhalo, only look for 1:1 matches
    dist, idx = [], []
    for i in tqdm(range(len(distance_upper_bound))):
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