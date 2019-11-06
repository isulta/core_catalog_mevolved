import numpy as np
import matplotlib.pyplot as plt
import h5py
import subhalo_mass_loss_model as SHMLM
from itk import hist, h5_read_dict, plt_latex, gio_read_dict

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

    # Match subhalos with mt nodes
    _, _, idx2 = np.intersect1d( sh['fof_halo_tag'], mt['fof_halo_tag'], assume_unique=False, return_indices=True)

    # Unique subhalos array with inverse indices
    _, idx_inv = np.unique(sh['fof_halo_tag'], return_inverse=True)
    
    # Initialize M array (corresponds with sh) to be host halo fof mass of each subhalo
    M = mt['fof_halo_mass'][idx2][idx_inv]

    # m/M with mask: M in [M1,M2] with central subhalos (subhalo tag == 0) removed
    mask = (M1<=M)&(M<=M2)&(sh['subhalo_tag']!=0)
    plot_arr = (sh['subhalo_mass']/M)[mask]
    # nH = np.sum( np.isin(mt['fof_halo_tag'], sh['fof_halo_tag'][mask]) ) #all halos with at least 1 subhalo
    nH = np.sum( (M1 <= mt['fof_halo_mass'])&(mt['fof_halo_mass']<=M2) )
    
    return plot_arr, nH

def CMF(outfile, M1, M2, s1=False, returnUnevolved=False, cc=None, returnZeta0=False, A=None):
    """Generate cores array m/M for mass function plot.
    
    Arguments:
        outfile {string} -- HDF5 core catalog file with 'coredata' dir.
        M1, M2 {float} -- Host halos with M in mass range [M1, M2], where M is central's `infall_mass` from core catalog.
        cc {dict} -- If given, `outfile` is not read and `cc` is used instead.
    
    Keyword Arguments:
        s1 {bool} -- If true, consider only 1st order substructure (i.e. subhalos). (default: {False})
        returnUnevolved {bool} -- If true, use unevolved m (i.e. core `infall_mass`) (default: {False})
        returnZeta0 {bool} -- If true, does fast mass evolution for the case of zeta=0, A=`A`.
    
    Returns:
        np 1d array -- m/M array with appropriate mask
        integer -- number of host halos (M)
    """

    if not cc:
        # Load extended core catalog
        cc = h5_read_dict(outfile, 'coredata')

    satellites_mask = cc['central'] == 0
    centrals_mask = cc['central'] == 1

    # Match satellites tni with centrals tni.
    vals2, idx3, idx4 = np.intersect1d( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask], return_indices=True)

    # Unique satellites tni array with inverse indices
    vals3, idx_inv = np.unique(cc['tree_node_index'][satellites_mask], return_inverse=True)

    # Some checks
    assert np.array_equal(vals2, vals3), "All satellites don't have a central match."
    assert np.array_equal(vals3[idx_inv], cc['tree_node_index'][satellites_mask]), 'np.unique inverse indices: original array recreation failure'
    assert np.array_equal(cc['tree_node_index'][centrals_mask][idx4], np.sort(cc['tree_node_index'][centrals_mask][idx4])), 'Centrals with satellites: array sorting failure'

    # Create M array (corresponds with cc[satellites_mask]) to be host tree node mass of each satellite
    M = cc['infall_mass'][centrals_mask][idx4][idx_inv]
    Coretag = cc['core_tag'][centrals_mask][idx4][idx_inv]
    
    mask = np.flatnonzero( (M1 <= M) & (M <= M2) & (cc['infall_mass'][satellites_mask] >= SHMLM.PARTICLES100MASS) )
    if s1:
        mask = np.intersect1d( mask, np.flatnonzero(cc['host_core'][satellites_mask]==Coretag) )
    
    # m/M array for CMF
    if returnUnevolved:
        plot_arr = (cc['infall_mass'][satellites_mask]/M)[mask]
    elif returnZeta0:
        plot_arr = (cc['infall_mass'][satellites_mask]/M)[mask]
    else:
        plot_arr = (cc['m_evolved'][satellites_mask]/M)[mask]
    
    nHalo = len(np.unique(cc['tree_node_index'][satellites_mask][mask]))
    
    return plot_arr, nHalo

def plotCMF(outfile, M1, M2, s1, returnUnevolved, label='', r=None, plotFlag=True, cc=None, normLogCnts=True):
    """Plot log/log plot of cores mass function (evolved or unevolved, given by `returnUnevolved`) with 100 bins on log(m/M) range `r` and M range [`M1`, `M2`]."""
    parr, nH = CMF(outfile, M1, M2, s1, returnUnevolved, cc)
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
