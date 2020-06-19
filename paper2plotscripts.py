def load_data(step):
    sh_vars = ['fof_halo_tag','subhalo_mean_x','subhalo_mean_y','subhalo_mean_z','subhalo_mean_vx', 'subhalo_mean_vy', 'subhalo_mean_vz', 'subhalo_count', 'subhalo_tag', 'subhalo_mass']
    hp_vars = ['fof_halo_tag', 'fof_halo_mass', 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z']

    cc_HM = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_ALCC_localhost_dtfactor_0.5_fitting3/{}.corepropertiesextend.hdf5'.format(step), 'coredata')
    #cc_HM['infall_fof_halo_mass'] = gio.gio_read('/home/isultan/data/ALCC/CoreCatalog/{}.coreproperties'.format(step), 'infall_fof_halo_mass')[0]

    sh_HM = gio_read_dict('/home/isultan/data/ALCC/subhalos/STEP{0}/m000p-{0}.subhaloproperties'.format(step), sh_vars)
    hp_HM = gio_read_dict('/home/isultan/data/ALCC/subhalos/STEP{0}/m000p-{0}.haloproperties'.format(step), hp_vars)

    cc_SV = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_LJDS_localhost_dtfactor_0.5_fitting2/m000p-{}.corepropertiesextend.hdf5'.format(step), 'coredata')
    sh_SV = gio_read_dict('/home/isultan/data/LJDS/subhalos/m000p-{}.subhaloproperties'.format(step), sh_vars)
    hp_SV = gio_read_dict('/home/isultan/data/LJDS/subhalos/m000p-{}.haloproperties'.format(step), hp_vars)                      

    cc_AQ = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_newdata_localhost_dtfactor_0.5_rev4070_v3/{}.corepropertiesextend.hdf5'.format(step), 'coredata')
    sh_AQ = gio_read_dict('/home/isultan/data/AlphaQ/100P/m000-{}.subhaloproperties'.format(step), sh_vars)
    hp_AQ = gio_read_dict('/home/isultan/data/AlphaQ/100P/m000-{}.haloproperties'.format(step), hp_vars)                  

    print len(np.unique(hp_HM['fof_halo_tag']))/len(hp_HM['fof_halo_tag'])*100

    for sh, hp, assert_x0_unique in [ (sh_HM, hp_HM, False), (sh_SV, hp_SV, True), (sh_AQ, hp_AQ, True) ]:
        idx_m21_sh = many_to_one(sh['fof_halo_tag'], hp['fof_halo_tag'], assert_x0_unique=assert_x0_unique)
        sh['M'] = hp['fof_halo_mass'][idx_m21_sh]
        sh['X'] = hp['fof_halo_center_x'][idx_m21_sh]
        sh['Y'] = hp['fof_halo_center_y'][idx_m21_sh]
        sh['Z'] = hp['fof_halo_center_z'][idx_m21_sh]

    centrals_mask_HM = cc_HM['central'] == 1
    centrals_mask_SV = cc_SV['central'] == 1
    centrals_mask_AQ = cc_AQ['central'] == 1

    for cc, centrals_mask in zip([cc_HM, cc_SV, cc_AQ], [centrals_mask_HM, centrals_mask_SV, centrals_mask_AQ]):
        idx_m21_cc = many_to_one(cc['tree_node_index'], cc['tree_node_index'][centrals_mask])
        for hk, k in [ ('M', 'infall_fof_halo_mass'), ('X', 'x'), ('Y', 'y'), ('Z', 'z'), ('CORETAG', 'core_tag') ]: #('CDELTA', 'infall_sod_halo_cdelta'), 
            cc[hk] = cc[k][centrals_mask][idx_m21_cc]
    
    return (cc_HM, sh_HM, centrals_mask_HM), (cc_SV, sh_SV, centrals_mask_SV),(cc_AQ, sh_AQ, centrals_mask_AQ)

def m_evolved_col(A, zeta, next=False):
    if next:
        return 'next_m_evolved_{}_{}'.format(A, zeta)
    else:
        return 'm_evolved_{}_{}'.format(A, zeta)

def cores_plot(cc, centrals_mask, M1, M2, label, bins, r, A=None, zeta=None, mlim=0, returnMask=False, verbose=True, mplot=False, bin_mask=None, nH_cores=None): 
    if A is None:
        A, zeta = AFID, ZETAFID
    if nH_cores is None:
        fht_fof = (cc['fof_halo_tag']<0)*np.bitwise_and(cc['fof_halo_tag']*-1, 0xffffffffffff) + (cc['fof_halo_tag']>=0)*cc['fof_halo_tag']
        nH_cores = len(np.unique( fht_fof[centrals_mask&(M1 <= cc['M'])&(cc['M'] <= M2)] ))      
    if bin_mask is None:
        bin_mask = (~centrals_mask)&(M1 <= cc['M'])&(cc['M'] <= M2)&(cc[m_evolved_col(A, zeta)]>mlim)
    else:
        assert mlim==0.
    #print label, "non fragment satellite cores", np.sum((cc['fof_halo_tag'][~centrals_mask]>=0))/np.sum(~centrals_mask)
    #print label, "non fragments count in mass bin", np.sum((~centrals_mask)&(M1 <= cc['M'])&(cc['M'] <= M2)&(cc[m_evolved_col(A, zeta)]>mlim)&(cc['fof_halo_tag']>=0))
    
    if verbose:
        print label, nH_cores
    parr = cc[m_evolved_col(A, zeta)][bin_mask] / (1.0 if mplot else cc['M'][bin_mask])
    x, y, yerr, yerr_log = hist(np.log10(parr), bins=bins, normed=True, plotFlag=False, range=r, normScalar=nH_cores, normBinsize=True, normLogCnts=True, retEbars=True)
    if returnMask:
        return x, y, yerr, yerr_log, nH_cores, bin_mask
    else:
        return x, y, yerr, yerr_log, nH_cores

def subhalo_plot(sh, M1, M2, label, bins, r, mlim=0, returnMask=False, mplot=False):
    bin_mask_sh = (sh['subhalo_tag']!=0)&(M1 <= sh['M'])&(sh['M'] <= M2)&(sh['subhalo_mass']>mlim)
    nH_sh = len(np.unique( sh['fof_halo_tag'][(M1 <= sh['M'])&(sh['M'] <= M2)] ))
    print label, nH_sh
    parr_sh = sh['subhalo_mass'][bin_mask_sh] / (1.0 if mplot else sh['M'][bin_mask_sh])
    x_sh, y_sh, yerr_sh, yerr_log_sh = hist(np.log10(parr_sh), bins=bins, normed=True, plotFlag=False, range=r, normScalar=nH_sh, normBinsize=True, normLogCnts=True, retEbars=True)
    if returnMask:
        return x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh, bin_mask_sh
    else:
        return x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh

def errorbar(ax, x, y, yerr, label='', marker='o', alpha=0.8, c=None, zerocut=False):
    if zerocut:
        cutmask = y!=0
        x, y, yerr = x[cutmask], y[cutmask], yerr[cutmask]
    ax.errorbar(x, y, yerr, label=label, marker=marker, ls='', mec='k', alpha=alpha, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=c )
    #ax.errorbar(x, y, yerr, label=label, marker=None, ls='-', mec='k', alpha=alpha, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=c )

def resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot=False, A=None, zeta=None):
    r = (9,13) if mplot else (-5,-0.5)
    bins = 40
    alpha = .8

    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='all', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [2, 1]}, figsize=[4.8*3,4.8*1.5], dpi=150)
    for logM1, ax, axr in zip((12, 13, 14), (ax1, ax2, ax3), (ax4, ax5, ax6)):
        yfid = 'HM'
        yerrfid = None
        M1, M2 = 10**logM1, 10**(logM1+0.5)
        ax.set_title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(logM1+0.0, logM1+0.5), y=0.9)

        for cc, sh, centrals_mask, label, marker, c in zip([cc_HM, cc_SV], [sh_HM, sh_SV], [centrals_mask_HM, centrals_mask_SV], ['HM', 'SV'], ['s', 'o'], ['#0d49fb', '#26eb47']):
            x, y, yerr, yerr_log, nH_cores, bin_mask = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=SUBHALOMINMASS[label], returnMask=True, mplot=mplot, A=A, zeta=zeta)
            errorbar(ax, x, y, yerr=yerr_log, label='Cores '+label, marker=marker)

            x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh, bin_mask_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=SUBHALOMINMASS[label], returnMask=True, mplot=mplot)
            errorbar(ax, x_sh, y_sh, yerr=yerr_log_sh, label='Subhalos '+label, marker=marker)

            print ''
            assert nH_cores == nH_sh

            if yfid == label:
                yfid = y_sh
                yerrfid = yerr_sh

            errorbar(axr, x, 10**(y-yfid), yerr=nratioerr(10**y, yerr, 10**yfid, yerrfid), marker=marker, c=c, zerocut=True)
            axr.axhline(1, c='k',ls='--', lw=1, zorder=-1)

            #ax.axvline( np.log10(SUBHALOMINMASS[label]/np.max(sh['M'][bin_mask_sh])), label=r'$\log \mathrm{m_{sh,min}/M_{max}}$ '+label, c='k' )
            if mplot:
                ax.axvline(  np.log10(100*PARTICLEMASS[label]), ymax=1., ls='--', c=c )
                axr.axvline( np.log10(100*PARTICLEMASS[label]), ls='--', label=r'$\log \left(\mathrm{100m_{p,'+label+ r'}}\right)$', c=c )
            else:
                ax.axvline(  np.log10(100*PARTICLEMASS[label]/np.mean(sh['M'][bin_mask_sh])), ymax=1., ls='--', c=c )
                axr.axvline( np.log10(100*PARTICLEMASS[label]/np.mean(sh['M'][bin_mask_sh])), ls='--', label=r'$\log \left(\mathrm{100m_{p,'+label+ r'}/\langle M_{sh} \rangle}\right)$', c=c )
        if mplot:
            ax.set_ylim(-3,3.6) #CHANGE
            axr.set_ylim(-1,3.95) #CHANGE
            ax.set_xlim(9,12.95)
        else:
            ax.set_ylim(-2,4) #CHANGE
            axr.set_ylim(-0.5,3.95) #CHANGE
            ax.set_xlim(-5,-0.5)
        
    ax1.legend(loc=3)
    ax4.legend(loc=2)
    if mplot:
        ax5.set_xlabel(r'$\log(m)$')
        ax1.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m) \right]$')
    else:
        ax5.set_xlabel(r'$\log(m/M)$')
        ax1.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m/M) \right]$')
    ax4.set_ylabel(r'ratio')

def paramscan_resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot, outfile):
    for i, A in enumerate(tqdm(A_arr)):
        for j, zeta in enumerate(zeta_arr):
            resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot, A, zeta)
            plt.savefig(outfile + '_{}_{}.png'.format(A, zeta)) #'/home/isultan/projects/halomassloss/core_catalog_mevolved/Paper2Figs/paramexploration_mplot/z1_{}_{}.png'.format(A, zeta)
            plt.close()

def cosmology_tests(cc_SV, centrals_mask_SV, cc_AQ, centrals_mask_AQ, r=(-4.0,0.), bins=20, A=None, zeta=None):
    alpha = 1#.7

    fig, (ax, ax1, ax2, ax3) = plt.subplots(4, sharex='all', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [4, 1, 1, 1]}, figsize=[4.8*1,4.8*7/4], dpi=150)
    for logM1, axr, marker in zip((12, 13, 14), (ax1,ax2,ax3), ('o', 'D', 's')):
        yfid = 'SV'
        yerrfid = None
        M1, M2 = 10**logM1, 10**(logM1+0.5)

        for cc, centrals_mask, label, markerc in zip([cc_SV, cc_AQ], [centrals_mask_SV, centrals_mask_AQ], ['SV', 'AlphaQ'], ['r', 'b']):
            x, y, yerr, yerr_log, nH_cores = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, A, zeta)
            errorbar(ax, x, y, yerr=yerr_log, label='Cores {} {}'.format(logM1, label), marker=marker, alpha=alpha)#, c=markerc)
                    
            print ''

            if yfid == label:
                yfid = y
                yerrfid = yerr
            else:
                errorbar(axr, x, 10**(y-yfid), yerr=nratioerr(10**y, yerr, 10**yfid, yerrfid), marker=marker, alpha=1)
            axr.axhline(1, c='k',ls='--', lw=0.7, zorder=-1)
        print axr.get_ylim()
        axr.set_ylim(0.5,1.49)

    ax3.set_xlabel(r'$\log(m/M)$')
    ax.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m/M) \right]$')
    ax2.set_ylabel(r'$\mathrm{CMF_{AlphaQ}/CMF_{SV}}$')
    ax.legend()

    # handles = [ ax.plot([], [], marker, label='[{}, {}]'.format(logM1+0.0, logM1+0.5)) for logM1, marker in zip((12, 13, 14), ('o', 'D', 's')) ]
    # handles = [ '[{}, {}]'.format(logM1+0.0, logM1+0.5) for logM1 in (12, 13, 14) ]
    # ax.legend(handles=handles, title=r'$\log \left[ \langle M \rangle / \left(h^{{-1}}M_\odot \right) \right]=$')

### CHI-SQUARE FITTING ###
def mask_ReducedChi2_gen(ReducedChi2):
    return np.abs(ReducedChi2 - ReducedChi2.min())<=1

def ReducedChi2dict_gen(cc, sh, centrals_mask, label, rdict, M1dict, M2dict, bins=20, mlim=0, mplot=False, dlog=False):
    ReducedChi2dict = {}
    for Mlabel in sorted(rdict.keys()):
        r = rdict[Mlabel]
        M1, M2 = M1dict[Mlabel], M2dict[Mlabel]
        _, _, _, _, nH_cores = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, mplot=mplot)
        x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=mlim, mplot=mplot)
        assert nH_cores == nH_sh

        Chi2 = np.empty((len(A_arr), len(zeta_arr)))
        for i, A in enumerate(A_arr):
            for j, zeta in enumerate(zeta_arr):
                x, y, yerr, yerr_log, _ = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, A=A, zeta=zeta, verbose=False, mplot=mplot, nH_cores=nH_cores)
                Chi2[i,j] = np.sum( (y_sh-y)**2/(yerr_log**2 + yerr_log_sh**2) ) if dlog else np.sum( (10**y_sh-10**y)**2/(yerr**2 + yerr_sh**2) )
        ReducedChi2 = np.true_divide(Chi2 , bins-2)
        assert np.isfinite(ReducedChi2).all()
        ReducedChi2dict[Mlabel] = ReducedChi2
        print ''
    return ReducedChi2dict

def pcolorplot_gen(ReducedChi2, Mlabel, M1, M2, outfile):
    plt.figure()
    plt.pcolormesh(zeta_arr, A_arr, ReducedChi2, cmap=plt.cm.jet)#, vmin=0.9240790129278156, vmax=542.9304287245984)
    cb = plt.colorbar()
    cb.set_label(r'$\chi^2_{\mathrm{dof}}$')

    Abfi, zetabfi = np.unravel_index(ReducedChi2.argmin(), ReducedChi2.shape)
    Abf, zetabf =  A_arr[Abfi], zeta_arr[zetabfi]
    print 'A', Abf, 'zeta', zetabf

    mask_ReducedChi2 = mask_ReducedChi2_gen(ReducedChi2)

    plt.plot(zetabf, Abf, 'x', ms=10, zorder=5, c='w')

    mask_ReducedChi2_marr = np.ma.masked_equal(mask_ReducedChi2, False)
    plt.pcolormesh(zeta_arr, A_arr, mask_ReducedChi2_marr, cmap='binary', alpha=.3)

    plt.xlabel('$\zeta$')
    plt.ylabel('$\mathcal{A}$')
    if Mlabel == 'ALL':
        plt.title(r'$\log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \geq$ {}'.format(np.log10(M1)), y=0.9, color='w')
    elif Mlabel == 12:
        plt.title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(np.log10(M1), np.log10(M2)), y=0.9, color='w')
    else:
        plt.title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(np.log10(M1), np.log10(M2)), y=0.9, color='w')
    print np.log10(M1), np.log10(M2)
    print ReducedChi2.min(), ReducedChi2.max()
    print ''

    if outfile:
        plt.savefig(outfile)
    
def pcolorplots(ReducedChi2dict, M1dict, M2dict, outfile=None):
    for Mlabel in sorted(M1dict.keys()):
        ReducedChi2 = ReducedChi2dict[Mlabel]
        M1, M2 = M1dict[Mlabel], M2dict[Mlabel]
        pcolorplot_gen(ReducedChi2, Mlabel, M1, M2, outfile=None)#'Paper2Figs/paramexploration_{}.pdf'.format(Mlabel)

### SUBHALO-TO-CORE MATCHING ###