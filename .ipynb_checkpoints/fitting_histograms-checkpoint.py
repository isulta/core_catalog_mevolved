from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subhalo_mass_loss_model as SHMLM
import genericio as gio
from tqdm import tqdm
from itk import hist, h5_read_dict, gio_read_dict, loadpickle, plt_latex, periodic_bcs, many_to_one
# from cc_generate_fitting import m_evolved_col, A_arr, zeta_arr
#from cc_generate_fitting_phasemerge import m_evolved_col, A_arr, zeta_arr
from cc_generate_fitting_dt_fiducial import m_evolved_col, A_arr, zeta_arr
from scipy import spatial #KD Tree for subhalo-core matching
from scipy.stats import norm # Gaussian fitting
import itertools as it
massbinsize = 0.5

step = 499
#step = 247
z = SHMLM.step2z[step]
# cc = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting_localhost_mergedcoretag/09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), 'coredata')
# cc = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting_localhost_mergedcoretag_parallel/09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), 'coredata')
#cc = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting_localhost_phasemerge/09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), 'coredata')

cc = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting_localhost_dtfactor_0.5/09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), 'coredata')
#cc = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting_localhost_dtfactor_0.5_fiducial_A_0.9_zeta_0.005_FIDUCIALPARTICLECUTMASS_PARTICLES100MASS_spline/09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), 'coredata')
#cc = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting_localhost_dtfactor_0.5_fiducial_A_0.9_zeta_0.005_FIDUCIALPARTICLECUTMASS_PARTICLES100MASS_spline_zbin_1.5_2/09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), 'coredata')
#cc = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting_localhost_dtfactor_0.5_fiducial_A_0.9_zeta_0.005_FIDUCIALPARTICLECUTMASS_PARTICLES100MASS_spline_match_zbin_1.5_2/09_03_2019.AQ.{}.corepropertiesextend.hdf5'.format(step), 'coredata')

#cc_pm = loadpickle('phasemerge_{}.pkl'.format(step))
#_, idx1, idx2 = np.intersect1d(cc['core_tag'], cc_pm['core_tag'], assume_unique=False, return_indices=True)
#cc['phaseSpaceMerged'] = np.zeros_like(cc['central'])
#cc['phaseSpaceMerged'][idx1] = cc_pm['phaseSpaceMerged'][idx2]

# sh = gio_read_dict('/home/isultan/data/AlphaQ/subhalos/m000-{}.subhaloproperties'.format(step), ['fof_halo_tag','subhalo_mean_x','subhalo_mean_y','subhalo_mean_z','subhalo_mean_vx', 'subhalo_mean_vy', 'subhalo_mean_vz', 'subhalo_count', 'subhalo_tag', 'subhalo_mass'])
mt_cores = gio_read_dict('/home/isultan/data/AlphaQ/updated_tree_nodes/09_03_2019.AQ.{}.treenodes'.format(step), ['fof_halo_tag', 'fof_halo_mass', 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z'])
sh = gio_read_dict('/home/isultan/data/AlphaQ/100P/m000-{}.subhaloproperties'.format(step), ['fof_halo_tag','subhalo_mean_x','subhalo_mean_y','subhalo_mean_z','subhalo_mean_vx', 'subhalo_mean_vy', 'subhalo_mean_vz', 'subhalo_count', 'subhalo_tag', 'subhalo_mass'])
mt = gio_read_dict('/home/isultan/data/AlphaQ/100P/m000-{}.haloproperties'.format(step), ['fof_halo_tag', 'fof_halo_mass', 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z'])

# M1, M2 = 10**12, 10**15.5

# Subhalo-core matching: to look at the subhalo mass:core evolved mass ratio
def generate_cores_kdtree(M1, M2, s1=False, disrupt=None, onlyFiltered=False, giveFragmentsFofMass=False):
    idx_filteredsatcores, M, X, Y, Z, nHalo = SHMLM.core_mask(cc, M1=10**8, M2=10**16, s1=s1, disrupt=disrupt, z=z)
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

def shmass(A, zeta, realFlag=False, plotFlag=False, mergedCoreTagFlag=False):
    """Returns Euclidean norm of subhalo and core mass functions, where only subhalos and cores in 1:1 match are considered."""
    # residual plots
    # print len(sh['subhalo_mass'][sh_mask][fmask])
    # print len(cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]])
    # realmask = cc_filtered['fof_halo_tag'][idx[:,0][fmask]]>=0

    realmask = np.full( np.sum(fmask), True, dtype=bool )
    if realFlag:
        realmask = np.invert( cc_filtered['wasInFragment'][idx[:,0][fmask]] )
    if mergedCoreTagFlag:
        realmask = realmask&(cc_filtered['mergedCoreTag'][idx[:,0][fmask]]==0)

    shmf = sh['subhalo_mass'][sh_mask][fmask][realmask]
    r = (9, 15)

    sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=False, normCnts=True, normLogCnts=True, normScalar=1, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)

    cmf = cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]][realmask]
    cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=1, normCnts=True, normBinsize=False, normLogCnts=True)

def paramSweep(dir, mostMassiveCoreFlag, mergedCoreTagFlag, plots100PartFlag, M1, M2, phaseMergedFlag=False, normCnts=False, MdepPlots=False, logmOverMFlag=True):
    for A, zeta in it.product(A_arr, zeta_arr):
        if not MdepPlots:
            plt.figure()
        if mostMassiveCoreFlag:
            mostMassiveCore(A, zeta, plotFlag=True)
        elif plots100PartFlag:
            subhalo_core_mass_plots(A, zeta, M1, M2, plotFlag=True, residualRange=(11.2,12.5), normLogCntsFlag=True, logmOverMFlag=logmOverMFlag, phaseMergedFlag=phaseMergedFlag, normCnts=normCnts, MdepPlots=MdepPlots)
        else:
            shmass(A, zeta, realFlag=False, plotFlag=True, mergedCoreTagFlag=mergedCoreTagFlag)
        if not MdepPlots:
            plt.title( 'z={}, {} $\leq$ log(M/$h^{{-1}}M_\odot$)$\leq$ {}'.format(z, np.log10(M1),np.log10(M2)) )
        ftitle = dir+'A_{}_zeta_{}.png'.format(A, zeta)
        plt.savefig( ftitle )
        print ftitle
        plt.close()

def subhalo_core_mass_plots(A, zeta, M1, M2, plotFlag, residualRange, normLogCntsFlag=True, logmOverMFlag=False, phaseMergedFlag=False, normCnts=False, MdepPlots=False, vlinepart=True, normScalar=False, normBinsize=False):
    """Plots subhalo and core mass histogram for all satellite subhalos and cores with at least 100 particles.
    Returns their residual (L2 norm) in `residualRange`."""
    if MdepPlots:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('z={}'.format(z))

        for ax,lM in zip([ax1, ax2, ax3, ax4],Mlist):
            M1, M2 = 10**lM, 10**(lM+massbinsize)
            # ax.set_title( '{} $\leq$ log(M/$h^{{-1}}M_\odot$)$\leq$ {}'.format(np.log10(M1),np.log10(M2)) )

            cc_filtered = cc_filtered_dict[lM] #generate_cores_kdtree(M1=M1, M2=M2, s1=False, disrupt=None, onlyFiltered=True)
            cores100mask = cc_filtered[m_evolved_col(A, zeta)]>=SHMLM.PARTICLECUTMASS
            if logmOverMFlag:
                normScalarVal = 1
                r = (-3, 0)
                # shmf = sh['subhalo_mass'][sh_mask][sh100mask] / sh['M'][sh_mask][sh100mask]
                shmf, nH = shmfdict[lM]
                sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=normBinsize, normCnts=normCnts, normLogCnts=normLogCntsFlag, normScalar=normScalarVal, plotFlag=plotFlag, label='subhalos', alpha=1, range=r, ax=ax)

                cmf = cc_filtered[m_evolved_col(A, zeta)][cores100mask] / cc_filtered['M'][cores100mask]
                cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=normScalarVal, normCnts=normCnts, normBinsize=normBinsize, normLogCnts=normLogCntsFlag, ax=ax)
                ax.set_xlim(-3.2, 0)
                #ax.set_ylim(-0.8, 3)
            else:
                r = (9, 15)
                # shmf = sh['subhalo_mass'][sh_mask][sh100mask] / sh['M'][sh_mask][sh100mask]
                shmf, nH = shfdict[lM]
                normScalarVal = 1
                if normScalar:
                    normScalarVal = nH
                sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=normBinsize, normCnts=normCnts, normLogCnts=normLogCntsFlag, normScalar=normScalarVal, plotFlag=plotFlag, label='subhalos', alpha=1, range=r, ax=ax)

                cmf = cc_filtered[m_evolved_col(A, zeta)][cores100mask]
                cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=normScalarVal, normCnts=normCnts, normBinsize=normBinsize, normLogCnts=normLogCntsFlag, ax=ax)
                ax.set_xlim(8.8, 15)
                ax.set_ylim(-0.8, 6)
                if vlinepart:
                    ax.axvline(x=np.log10(SHMLM.PARTICLES100MASS), c='k', lw=0.5, label='100 part.')
                    ax.axvline(x=np.log10(SHMLM.PARTICLES100MASS/100.*20), c='r', lw=0.5, label='20 part.')

            assert np.array_equal(cores_xarr, sh_xarr)
            ##ax.set_title( '[{},{}]'.format(np.log10(M1),np.log10(M2)) )
            ax.set_title( '{} $\leq$ log(M/$h^{{-1}}M_\odot$)$\leq$ {}'.format(np.log10(M1),np.log10(M2)) )
            ##ax.text(0.5, 0.7, 'sh:{}'.format(np.format_float_scientific(len(shmf))), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
            ##ax.text(0.5, 0.6, 'c:{}'.format(np.format_float_scientific(len(cmf))), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
            ##ax.text(0.5, 0.5, 'H:{}'.format(np.format_float_scientific(nH)), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
        if logmOverMFlag:
            fig.text(0.5,0.04, r'$\log(m/M)$', ha="center", va="center")
            fig.text(0.05,0.5, r'$\log \left[ \mathrm{d}N/\mathrm{d} \log(m/M) \right]$', ha="center", va="center", rotation=90)
        else:
            fig.text(0.5,0.04, r'$\log(m)$', ha="center", va="center")
            fig.text(0.05,0.5, r'$\log \left[ \mathrm{d}N/\mathrm{d} \log(m) \right]$', ha="center", va="center", rotation=90)
        ax1.legend()
        plt.subplots_adjust(hspace = 0.4)
        return fig, ((ax1, ax2), (ax3, ax4))
    else:
        # sh_mask = (sh['subhalo_tag']!=0)&(M1<=sh['M'])&(sh['M']<=M2)#&(distance_mask) #&(sh['subhalo_mass']>=SHMLM.SUBHALOMASSCUT)
        # sh100mask = sh['subhalo_mass'][sh_mask]>=SHMLM.PARTICLECUTMASS
        # cores_tree, cc_filtered, nHalo = generate_cores_kdtree(M1=M1, M2=M2, s1=False, disrupt=None)
        cores100mask = cc_filtered[m_evolved_col(A, zeta)]>=SHMLM.PARTICLECUTMASS
        if phaseMergedFlag:
            cores100mask = cores100mask&(cc_filtered['phaseSpaceMerged']!=1)
        if logmOverMFlag:
            r = (-3, 0)
            shmf = sh['subhalo_mass'][sh_mask][sh100mask] / sh['M'][sh_mask][sh100mask]
            sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=False, normCnts=normCnts, normLogCnts=normLogCntsFlag, normScalar=1, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)

            cmf = cc_filtered[m_evolved_col(A, zeta)][cores100mask] / cc_filtered['M'][cores100mask]
            cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=1, normCnts=normCnts, normBinsize=False, normLogCnts=normLogCntsFlag)
        else:
            r = (9, 15)
            shmf = sh['subhalo_mass'][sh_mask][sh100mask]
            sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=False, normCnts=normCnts, normLogCnts=normLogCntsFlag, normScalar=1, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)

            cmf = cc_filtered[m_evolved_col(A, zeta)][cores100mask]
            cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=1, normCnts=normCnts, normBinsize=False, normLogCnts=normLogCntsFlag)
        assert np.array_equal(cores_xarr, sh_xarr)
        # print sh_cnts, cores_cnts
        res_mask = (residualRange[0]<=sh_xarr)&(sh_xarr<=residualRange[1])
        return np.linalg.norm(np.true_divide(((sh_cnts-cores_cnts)[res_mask]), sh_cnts[res_mask]))

def plots100PartResidual(dir, residualRange, M1, M2, phaseMergedFlag, normCnts=True):
    R = np.empty((len(A_arr), len(zeta_arr)))
    Rlog = np.empty((len(A_arr), len(zeta_arr)))
    for i in range(len(A_arr)):
        for j in range(len(zeta_arr)):
            A, zeta = A_arr[i], zeta_arr[j]
            R[i,j] = subhalo_core_mass_plots(A, zeta, M1, M2, plotFlag=False, residualRange=residualRange, normLogCntsFlag=False, logmOverMFlag=True, phaseMergedFlag=phaseMergedFlag, normCnts=normCnts)
            Rlog[i,j] = subhalo_core_mass_plots(A, zeta, M1, M2, plotFlag=False, residualRange=residualRange, normLogCntsFlag=True, logmOverMFlag=True, phaseMergedFlag=phaseMergedFlag, normCnts=normCnts)
            print R[i,j]
    plt.figure()
    plt.pcolormesh(zeta_arr, A_arr, R, cmap=plt.cm.jet)
    plt.colorbar()
    plt.savefig(dir+'res2d_loggedcnts_{}_{}.png'.format(residualRange[0], residualRange[1]))
    plt.close()

    plt.figure()
    plt.pcolormesh(zeta_arr, A_arr, Rlog, cmap=plt.cm.jet)
    plt.colorbar()
    plt.savefig(dir+'res2d_{}_{}.png'.format(residualRange[0], residualRange[1]))
    plt.close()

def residual(A, zeta, realFlag=False, plotFlag=False):
    """Returns Euclidean norm of subhalo and core mass functions, where only subhalos and cores in 1:1 match are considered."""
    # residual plots
    print len(sh['subhalo_mass'][sh_mask][fmask])
    print len(cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]])
    # realmask = cc_filtered['fof_halo_tag'][idx[:,0][fmask]]>=0

    if realFlag:
        realmask = np.invert( cc_filtered['wasInFragment'][idx[:,0][fmask]] )
    else:
        realmask = np.full( np.sum(fmask), True, dtype=bool )

    shmf = sh['subhalo_mass'][sh_mask][fmask][realmask] / sh['M'][sh_mask][fmask][realmask]
    r = (-3, 0)
    r_res = (-3, -2)

    sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=True, normCnts=False, normLogCnts=True, normScalar=nHalo, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)
    res_mask = (r_res[0]<=sh_xarr)&(sh_xarr<=r_res[1])

    cmf = cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]][realmask] / cc_filtered['M'][idx[:,0][fmask]][realmask]
    cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=nHalo, normCnts=False, normBinsize=True, normLogCnts=True)

    return np.linalg.norm((sh_cnts-cores_cnts)[res_mask])

# def residual(A, zeta, plotFlag=False):
#     """Returns Euclidean norm of subhalo and core mass functions, where only subhalos and cores in 1:1 match are considered."""
#     # residual plots
#     print len(sh['subhalo_mass'][sh_mask][fmask])
#     print len(cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]])

#     shmf = sh['subhalo_mass'][sh_mask][fmask] / cc_filtered['M'][idx[:,0][fmask]]#sh['M'][sh_mask][fmask]
#     r = (-3, 0)
#     r_res = (-3, -2)

#     sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=True, normCnts=False, normLogCnts=True, normScalar=nHalo, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)
#     res_mask = (r_res[0]<=sh_xarr)&(sh_xarr<=r_res[1])

#     cmf = cc_filtered[m_evolved_col(A, zeta)][idx[:,0][fmask]] / cc_filtered['M'][idx[:,0][fmask]]
#     cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=nHalo, normCnts=False, normBinsize=True, normLogCnts=True)

#     return np.linalg.norm((sh_cnts-cores_cnts)[res_mask])

# """Save matched mass functions"""
# def saveMf(A, zeta):
#     plt.figure()
#     residual(A, zeta, realFlag=True, plotFlag=True)
#     plt.title( 'z={}, {} $\leq$ log(M/$h^{{-1}}M_\odot$)$\leq$ {}'.format(z, np.log10(M1),np.log10(M2)) )
#     ftitle = 'testthreadplots/A_{}_zeta_{}_notInFragment.png'.format(A, zeta)
#     plt.savefig( ftitle )
#     print ftitle
#     plt.close()

# for A, zeta in it.product(A_arr, zeta_arr):
#     plt.figure()
#     residual(A, zeta, realFlag=True, plotFlag=True)
#     plt.title( 'z={}, {} $\leq$ log(M/$h^{{-1}}M_\odot$)$\leq$ {}'.format(z, np.log10(M1),np.log10(M2)) )
#     ftitle = 'mfs_fitting2_499/A_{}_zeta_{}_notInFragment.png'.format(A, zeta)
#     plt.savefig( ftitle )
#     print ftitle
#     plt.close()


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

def mostMassiveCore(A, zeta, plotFlag):
    iarr, carr, missing = [], [], []
    # cnt = 0
    for i in tqdm(range(len(distance_upper_bound))):
        qres = cores_tree.query_ball_point(sh_arr[i], r=distance_upper_bound[i])
        # qres2 = []
        # for qidx in qres:
        #     if SHMLM.dist(sh['X'][sh_mask][i], sh['Y'][sh_mask][i], sh['Z'][sh_mask][i], cc_filtered['x'][qidx], cc_filtered['y'][qidx], cc_filtered['z'][qidx]) != 0:
        #         qres2.append(qidx)
        #     else:
        #         cnt += 1
        # qres = qres2
        if len(qres)>0:
            idxmax = qres[ np.argmax(cc_filtered[m_evolved_col(A, zeta)][qres]) ]
            iarr.append(i)
            carr.append(idxmax)
        else:
            missing.append(i)
    percentexists = len(iarr)/np.sum(sh_mask)*100
    print '{}% of masked subhalos have at least 1 core within their search radius.'.format(percentexists)
    shmf = sh['subhalo_mass'][sh_mask][iarr]
    r = (9, 15)

    sh_xarr, sh_cnts = hist(np.log10(shmf), bins=100, normed=True, normBinsize=True, normCnts=False, normLogCnts=True, normScalar=nHalo, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)

    cmf = cc_filtered[m_evolved_col(A, zeta)][carr]
    cores_xarr, cores_cnts = hist(np.log10(cmf), bins=100, normed=True, plotFlag=plotFlag, label='cores', alpha=1, range=r, normScalar=nHalo, normCnts=False, normBinsize=True, normLogCnts=True)
    # print "Cnt: "+str(cnt)

    plt.figure()
    plt.hist((sh['subhalo_mass'][sh_mask][iarr] - cc_filtered[m_evolved_col(A, zeta)][carr])/sh['subhalo_mass'][sh_mask][iarr], range=(-5,5), bins=100)#, alpha=.5)
    plt.axvline(x=0, c='k')

    plt.figure()
    shmassmask = sh['subhalo_mass'][sh_mask][iarr] >= (SHMLM.PARTICLES100MASS*10)
    plt.hist((sh['subhalo_mass'][sh_mask][iarr][shmassmask] - cc_filtered[m_evolved_col(A, zeta)][carr][shmassmask])/sh['subhalo_mass'][sh_mask][iarr][shmassmask], range=(-5,5), bins=100)#, alpha=.5)
    plt.axvline(x=0, c='k')

if __name__ == "__main__":
    plt_latex()

    sh['rvir'] = SHMLM.getRvir(sh['subhalo_mass'], z)

    idx_m21_sh = many_to_one(sh['fof_halo_tag'], mt['fof_halo_tag'])
    sh['M'] = mt['fof_halo_mass'][idx_m21_sh]

    sh['X'] = mt['fof_halo_center_x'][idx_m21_sh]
    sh['Y'] = mt['fof_halo_center_y'][idx_m21_sh]
    sh['Z'] = mt['fof_halo_center_z'][idx_m21_sh]

    # M1, M2 = 10**12., 10**15.5
    ### Which cores to match subhalos to ###
    # Only include satellite cores in matching
    # cores_tree, cc_filtered, nHalo = generate_cores_kdtree(M1=M1, M2=M2, s1=False, disrupt=None)
    # _, _, nHalo = generate_cores_kdtree(M1=M1, M2=M2, s1=False, disrupt=None)

    # Attempt to remove 'true' FOF central cores from list of all cores
    # cc_mask = np.invert( np.isin(cc['x'], sh['X'])&(np.isin(cc['y'], sh['Y']))&(np.isin(cc['z'], sh['Z'])) )
    # cores_tree, cc_filtered = spatial.cKDTree( np.vstack((cc['x'][cc_mask], cc['y'][cc_mask], cc['z'][cc_mask])).T ), {k:cc[k][cc_mask].copy() for k in cc.keys()}

    # Use ALL cores in matching
    # cores_tree, cc_filtered = spatial.cKDTree( np.vstack((cc['x'], cc['y'], cc['z'])).T ), cc

    # Use Not merged cores in matching
    # notMergedMask = (cc['mergedCoreTag'] == 0)
    # cc_filtered = {k:cc[k][notMergedMask].copy() for k in cc.keys()}
    # cores_tree = spatial.cKDTree( np.vstack((cc_filtered['x'], cc_filtered['y'], cc_filtered['z'])).T )
    ### End: Which cores to match subhalos to ###
    subhalo_mean_x = SHMLM.periodic_bcs(sh['subhalo_mean_x'], sh['X'], SHMLM.BOXSIZE)
    subhalo_mean_y = SHMLM.periodic_bcs(sh['subhalo_mean_y'], sh['Y'], SHMLM.BOXSIZE)
    subhalo_mean_z = SHMLM.periodic_bcs(sh['subhalo_mean_z'], sh['Z'], SHMLM.BOXSIZE)

    #distance_mask = SHMLM.dist(subhalo_mean_x, subhalo_mean_y, subhalo_mean_z, sh['X'], sh['Y'], sh['Z']) >= SHMLM.SUBHALOHOSTVIRIALRADIUSFACTOR*SHMLM.getRvir(sh['M'], z)
    #sh_mask = (sh['subhalo_tag']!=0)&(M1<=sh['M'])&(sh['M']<=M2)#&(distance_mask) #&(sh['subhalo_mass']>=SHMLM.SUBHALOMASSCUT)
    #print '{}% satellite subhalos in given mass range remain after mask.'.format(np.sum(sh_mask)/np.sum((sh['subhalo_tag']!=0)&(M1<=sh['M'])&(sh['M']<=M2))*100.)

    # sh_mask = (sh['subhalo_tag']!=0)&(M1<=sh['M'])&(sh['M']<=M2)#&(distance_mask) #&(sh['subhalo_mass']>=SHMLM.SUBHALOMASSCUT)
    # sh_mask = (sh['subhalo_tag']!=0)&(M1<=sh['M'])&(sh['M']<=M2)&(sh['subhalo_mass']>=SHMLM.SUBHALOMASSCUT)
    # sh100mask = sh['subhalo_mass'][sh_mask]>=SHMLM.PARTICLECUTMASS
    # cores_tree, cc_filtered, nHalo = generate_cores_kdtree(M1=M1, M2=M2, s1=False, disrupt=None)
    '''
    sh_arr = np.vstack((subhalo_mean_x[sh_mask], subhalo_mean_y[sh_mask], subhalo_mean_z[sh_mask])).T
    distance_upper_bound = SHMLM.SUBHALOVIRIALRADIUSFACTOR * sh['rvir'][sh_mask]
    # for SUBHALOVIRIALRADIUSFACTOR in [1,2,3,4]:
    #     distance_upper_bound = SUBHALOVIRIALRADIUSFACTOR * sh['rvir'][sh_mask]
    #     mostMassiveCore(A=1.1, zeta=0.01, plotFlag=False)

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
    '''

    Mlist = [11,12,13,14]
    def shmf_gen(lM, shfFlag=False):
        M1, M2 = 10**lM, 10**(lM+massbinsize)
        sh_mask = (sh['subhalo_tag']!=0)&(M1<=sh['M'])&(sh['M']<=M2)#&(distance_mask) #&(sh['subhalo_mass']>=SHMLM.SUBHALOMASSCUT)
        sh100mask = sh['subhalo_mass'][sh_mask]>=SHMLM.PARTICLECUTMASS
        if shfFlag:
            shmf = sh['subhalo_mass'][sh_mask][sh100mask]
        else:
            shmf = sh['subhalo_mass'][sh_mask][sh100mask] / sh['M'][sh_mask][sh100mask]
        nH = len(np.unique(sh['fof_halo_tag'][sh_mask]))
        return (shmf, nH)
    def cc_filtered_gen(lM):
        M1, M2 = 10**lM, 10**(lM+massbinsize)
        return generate_cores_kdtree(M1=M1, M2=M2, s1=False, disrupt=None, onlyFiltered=True, giveFragmentsFofMass=True)
    shmfdict = {k:shmf_gen(k) for k in Mlist}
    shfdict = {k:shmf_gen(k, shfFlag=True) for k in Mlist}
    cc_filtered_dict = {k:cc_filtered_gen(k) for k in Mlist}


