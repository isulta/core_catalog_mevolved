import numpy as np
import matplotlib.pyplot as plt
import h5py
import subhalo_mass_loss_model as SHMLM
from itk import hist, h5_read_dict, plt_latex, gio_read_dict, many_to_one

# TO CHANGE
step = 499

#cc/merger tree Sept. 11 2019
data_dir = "/home/isultan/data/AlphaQ/"
fname_mt = data_dir + "updated_tree_nodes/09_03_2019.AQ.{}.treenodes".format(step)
fname_subhalos = data_dir + "subhalos/m000-{}.subhaloproperties".format(step)
# END TO CHANGE

# GLOBAL VARS
sh = None
mt = None
# END GLOBAL VARS

def subhalos_read():
    """Read in subhalos and merger tree into global dicts `sh` and `mt`.""" 
    global sh, mt
    if sh and mt:
        return
    sh = gio_read_dict(fname_subhalos, ['subhalo_mass', 'subhalo_tag', 'fof_halo_tag'])
    mt = gio_read_dict(fname_mt, ['fof_halo_tag', 'fof_halo_mass'])

def SHMF(M1, M2):
    """Returns 'subhalo mass function' m/M for M in [`M1`,`M2`]."""
    subhalos_read()
    assert np.all(np.isin(sh['fof_halo_tag'], mt['fof_halo_tag'])), 'Every subhalo does not have corresponding fof_halo_tag in merger tree.'
    assert len(np.unique(mt['fof_halo_tag']))==len(mt['fof_halo_tag']), 'Duplicate fof_halo_tag(s) found in merger tree.'

    idx_m21 = many_to_one(sh['fof_halo_tag'], mt['fof_halo_tag'])
    
    # Initialize M array (corresponds with sh) to be host halo fof mass of each subhalo
    M = mt['fof_halo_mass'][idx_m21]

    # m/M with mask: M in [M1,M2] with central subhalos (subhalo tag == 0) removed
    mask = (M1<=M)&(M<=M2)&(sh['subhalo_tag']!=0)
    plot_arr = (sh['subhalo_mass']/M)[mask]
    # nH = np.sum( np.isin(mt['fof_halo_tag'], sh['fof_halo_tag'][mask]) ) #all halos with at least 1 subhalo
    nH = np.sum( (M1 <= mt['fof_halo_mass'])&(mt['fof_halo_mass']<=M2) )
    
    return plot_arr, nH

def CMF(outfile, M1, M2, s1=False, disrupt=None, returnUnevolved=False, cc=None, returnZeta0=False, A=None):
    """Generate cores array m/M for mass function plot.
    
    Arguments:
        outfile {string} -- HDF5 core catalog file with 'coredata' dir.
        M1, M2 {float} -- Host halos with M in mass range [M1, M2], where M is central's `infall_mass` from core catalog.
        cc {dict} -- If given, `outfile` is not read and `cc` is used instead.
        A {float} -- Fitting parameter used for fast mass evolution if `returnZeta0`.
        disrupt {None, int, or list} -- If given, applies core disruption criteria defined in SHMLM when filtering substructure. (default: {None})
    
    Keyword Arguments:
        s1 {bool} -- If true, consider only 1st order substructure (i.e. subhalos). (default: {False})
        returnUnevolved {bool} -- If true, use unevolved m (i.e. core `infall_mass`) (default: {False})
        returnZeta0 {bool} -- If true, does fast mass evolution for the case of zeta=0, A=`A`. (default: {False})
    
    Returns:
        np 1d array -- m/M array with appropriate mask
        integer -- number of host halos (M)
    """

    if not cc:
        # Load extended core catalog
        cc = h5_read_dict(outfile, 'coredata')

    idx_filteredsatcores, M, nHalo = SHMLM.core_mask(cc, M1, M2, s1=s1, disrupt=disrupt)
    
    # m/M array for CMF
    if returnUnevolved:
        plot_arr = cc['infall_mass'][idx_filteredsatcores]/M
    elif returnZeta0:
        plot_arr = SHMLM.fast_m_evolved( cc['infall_mass'][idx_filteredsatcores]/M, cc['infall_step'][idx_filteredsatcores], A)
    else:
        plot_arr = cc['m_evolved'][idx_filteredsatcores]/M
    
    return plot_arr, nHalo

def plotCMF(outfile, M1, M2, s1, returnUnevolved, label='', r=None, plotFlag=True, cc=None, normLogCnts=True, returnZeta0=False, A=None, disrupt=None):
    """Plot log/log plot of cores mass function (evolved or unevolved, given by `returnUnevolved`) with 100 bins on log(m/M) range `r` and M range [`M1`, `M2`]."""
    parr, nH = CMF(outfile, M1, M2, s1, disrupt, returnUnevolved, cc, returnZeta0, A)
    return hist(np.log10(parr), bins=100, normed=True, plotFlag=plotFlag, label=label, alpha=1, range=r, normScalar=nH, normCnts=False, normBinsize=True, normLogCnts=normLogCnts)

def plotSHMF(M1, M2, r=None, label='subhalos', plotFlag=True, normLogCnts=True):
    """Plot log/log plot of subhalo mass function with 100 bins on log(m/M) range `r` and M range [`M1`, `M2`]."""
    shmf, nH = SHMF(M1, M2)
    return hist(np.log10(shmf), bins=100, normed=True, normBinsize=True, normCnts=False, normLogCnts=normLogCnts, normScalar=nH, plotFlag=plotFlag, label='subhalos', alpha=1, range=r)

def SHMF_plot(outfile, M1, M2, bins, step):
	# TODO plot histogram
    pass
	# plt_latex()
	# plt.figure(dpi=120)
	# plt.hist(np.log10(plot_arr), bins=bins, histtype='bar', density=False)
	# plt.yscale('log')

	# plt.xlabel(r'$\log(m/M)$')
	# plt.ylabel(r'$\mathrm{d}n/\mathrm{d} \log(m/M)$')
	# plt.title( 'z=' + str(round(SHMLM.step2z[step],3)) + r', {} $\leq$ log(M/$h^{{-1}}M_\odot$)$\leq$ {}'.format(np.log10(M1),np.log10(M2)) )
	# plt.show()

if __name__ == '__main__':
    pass
	# SHMF_plot(outfile='output/09_03_2019.AQ.499.corepropertiesextend.hdf5', M1=10**14.0, M2=10**15.0, bins=500, step=499)
