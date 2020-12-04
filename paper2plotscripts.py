CC_HOST_VARS = [ ('M', 'infall_fof_halo_mass'), ('X', 'x'), ('Y', 'y'), ('Z', 'z'), ('CORETAG', 'core_tag'), ('CDELTA', 'infall_sod_halo_cdelta'), ('Mtn', 'infall_tree_node_mass') ]

def load_data(step):
    sh_vars = ['fof_halo_tag','subhalo_mean_x','subhalo_mean_y','subhalo_mean_z','subhalo_mean_vx', 'subhalo_mean_vy', 'subhalo_mean_vz', 'subhalo_count', 'subhalo_tag', 'subhalo_mass', 'fof_halo_count']
    hp_vars = ['fof_halo_tag', 'fof_halo_mass', 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z']

    cc_HM = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_ALCC_localhost_dtfactor_0.5_fitting3/{}.corepropertiesextend.hdf5'.format(step), 'coredata')
    sh_HM = h5_read_dict(f'/home/isultan/data/SHfindertests/HM_final/subhalos_HM_{step}.hdf5', 'subhalos')

    cc_SV = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_LJDS_localhost_dtfactor_0.5_fitting2/m000p-{}.corepropertiesextend.hdf5'.format(step), 'coredata')
    cc_SV['infall_sod_halo_cdelta'] = gio.gio_read('/home/isultan/data/LJDS/CoreCatalog/m000p-{}.coreproperties'.format(step), 'infall_sod_halo_cdelta')[0]
    sh_SV = gio_read_dict('/home/isultan/data/LJDS/subhalos/m000p-{}.subhaloproperties'.format(step), sh_vars)
    hp_SV = gio_read_dict('/home/isultan/data/LJDS/subhalos/m000p-{}.haloproperties'.format(step), hp_vars)

    cc_AQ = h5_read_dict('/home/isultan/projects/halomassloss/core_catalog_mevolved/output_newdata_localhost_dtfactor_0.5_rev4070_v3/{}.corepropertiesextend.hdf5'.format(step), 'coredata')
    sh_AQ = gio_read_dict('/home/isultan/data/AlphaQ/100P/m000-{}.subhaloproperties'.format(step), sh_vars)
    
    for sh, label in [ (sh_HM, 'HM'), (sh_SV, 'SV'), (sh_AQ, 'AQ') ]:
        sh['M'] = sh['fof_halo_count']*PARTICLEMASS[label]

    for sh, hp, assert_x0_unique in [ (sh_SV, hp_SV, True) ]:
        idx_m21_sh = many_to_one(sh['fof_halo_tag'], hp['fof_halo_tag'], assert_x0_unique=assert_x0_unique)
        sh['X'] = hp['fof_halo_center_x'][idx_m21_sh]
        sh['Y'] = hp['fof_halo_center_y'][idx_m21_sh]
        sh['Z'] = hp['fof_halo_center_z'][idx_m21_sh]

    centrals_mask_HM = cc_HM['central'] == 1
    centrals_mask_SV = cc_SV['central'] == 1
    centrals_mask_AQ = cc_AQ['central'] == 1

    for cc, centrals_mask in zip([cc_HM, cc_SV, cc_AQ], [centrals_mask_HM, centrals_mask_SV, centrals_mask_AQ]):
        idx_m21_cc = many_to_one(cc['tree_node_index'], cc['tree_node_index'][centrals_mask])
        for hk, k in CC_HOST_VARS:
            cc[hk] = cc[k][centrals_mask][idx_m21_cc]

    return (cc_HM, sh_HM, centrals_mask_HM), (cc_SV, sh_SV, centrals_mask_SV),(cc_AQ, sh_AQ, centrals_mask_AQ)

def load_cc(step, dir):
    cc = h5_read_dict(dir+'{}.corepropertiesextend.hdf5'.format(step), 'coredata')
    centrals_mask = cc['central'] == 1
    idx_m21_cc = many_to_one(cc['tree_node_index'], cc['tree_node_index'][centrals_mask])
    for hk, k in CC_HOST_VARS:
        cc[hk] = cc[k][centrals_mask][idx_m21_cc]
    return cc, centrals_mask

def load_cc_LastJourney(step, MfofScaleFactor=None):
    cc = {
        **h5_read_files(f'/home/isultan/data/LastJourney/CoreCatalog_004_Reduced/m000p-{step}'),
        **h5_read_files(f'/home/isultan/data/LastJourney/CoreCatalog_004_Reduced_SHMF/m000p-{step}'),
        **h5_read_dict(f'/home/isultan/projects/halomassloss/core_catalog_mevolved/output_LastJourney_localhost_dtfactor_0.5/m000p-{step}.corepropertiesextend.hdf5', 'coredata')
    }
    centrals_mask = cc['central'] == 1
    if MfofScaleFactor is not None:
        cc['infall_fof_halo_mass'] *= MfofScaleFactor
    idx_m21_cc = many_to_one(cc['tree_node_index'][~centrals_mask], cc['tree_node_index'][centrals_mask], verbose=True, assert_x0_unique=True, assert_x1_in_x0=False)
    for hk, k in CC_HOST_VARS[:1]:
        cc[hk] = np.zeros_like(cc[k])
        cc[hk][~centrals_mask] = cc[k][centrals_mask][idx_m21_cc]
        cc[hk][centrals_mask] = cc[k][centrals_mask]
    return cc, centrals_mask

def m_evolved_col(A, zeta, next=False):
    if next:
        return 'next_m_evolved_{}_{}'.format(A, zeta)
    else:
        return 'm_evolved_{}_{}'.format(A, zeta)

def cores_plot(cc, centrals_mask, M1, M2, label, bins, r, A=None, zeta=None, mlim=0, returnMask=False, verbose=True, mplot=False, bin_mask=None, nH_cores=None, Mvar='M', returnMavg=False):
    if A is None:
        A, zeta = AFID, ZETAFID
    if nH_cores is None:
        if Mvar == 'Mtn':
            nH_cores = np.sum( (M1<=cc['Mtn'][centrals_mask])&(cc['Mtn'][centrals_mask]<=M2) )
        else:
            fht_fof = real_fof_tags(cc['fof_halo_tag'])
            nH_cores = len(np.unique( fht_fof[centrals_mask&(M1 <= cc[Mvar])&(cc[Mvar] <= M2)] ))
    if bin_mask is None:
        bin_mask = (~centrals_mask)&(M1 <= cc[Mvar])&(cc[Mvar] <= M2)&(cc[m_evolved_col(A, zeta)]>mlim)
    else:
        assert mlim==0.
    #print( label, "non fragment satellite cores", np.sum((cc['fof_halo_tag'][~centrals_mask]>=0))/np.sum(~centrals_mask) )
    #print( label, "non fragments count in mass bin", np.sum((~centrals_mask)&(M1 <= cc[Mvar])&(cc[Mvar] <= M2)&(cc[m_evolved_col(A, zeta)]>mlim)&(cc['fof_halo_tag']>=0)) )

    if verbose:
        print(label, 'nH cores', nH_cores)
    parr = cc[m_evolved_col(A, zeta)][bin_mask] / (1.0 if mplot else cc[Mvar][bin_mask])
    x, y, yerr, yerr_log = hist(np.log10(parr), bins=bins, normed=True, plotFlag=False, range=r, normScalar=nH_cores, normBinsize=True, normLogCnts=True, retEbars=True)
    if returnMask:
        return x, y, yerr, yerr_log, nH_cores, bin_mask
    elif returnMavg:
        fht_fof = real_fof_tags(cc['fof_halo_tag'])
        hosts_mask = centrals_mask&(M1 <= cc[Mvar])&(cc[Mvar] <= M2)
        Mavg = np.mean( cc[Mvar][hosts_mask][np.unique( fht_fof[hosts_mask], return_index=True )[1]] )
        print(f'Mavg (cores centrals): {np.format_float_scientific(Mavg)}')
        return x, y, yerr, yerr_log, nH_cores, Mavg
    else:
        return x, y, yerr, yerr_log, nH_cores

def subhalo_plot(sh, M1, M2, label, bins, r, mlim=0, returnMask=False, mplot=False, Mvar='M', returnMavg=False):
    bin_mask_sh = (sh['subhalo_tag']!=0)&(M1 <= sh[Mvar])&(sh[Mvar] <= M2)&(sh['subhalo_mass']>mlim)
    nH_sh = len(np.unique( sh['fof_halo_tag'][(M1 <= sh[Mvar])&(sh[Mvar] <= M2)] ))
    print(label, 'nH_sh', nH_sh)
    parr_sh = sh['subhalo_mass'][bin_mask_sh] / (1.0 if mplot else sh[Mvar][bin_mask_sh])
    x_sh, y_sh, yerr_sh, yerr_log_sh = hist(np.log10(parr_sh), bins=bins, normed=True, plotFlag=False, range=r, normScalar=nH_sh, normBinsize=True, normLogCnts=True, retEbars=True)
    print(label, 'number of subhalos in bin', len(parr_sh))
    if returnMask:
        return x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh, bin_mask_sh
    elif returnMavg:
        assert len(np.unique(sh['fof_halo_tag'][sh['subhalo_tag']==0]))==np.sum(sh['subhalo_tag']==0), 'Duplicate fof halo tags in sh centrals'
        Mavg = np.mean( sh[Mvar][(sh['subhalo_tag']==0)&(M1 <= sh[Mvar])&(sh[Mvar] <= M2)] )
        print(f'Mavg (sh centrals): {np.format_float_scientific(Mavg)}')
        return x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh, Mavg
    else:
        return x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh

def errorbar(ax, x, y, yerr, label='', marker='o', alpha=1, c=None, zerocut=False):
    if zerocut:
        cutmask = y!=0
        x, y, yerr = x[cutmask], y[cutmask], yerr[cutmask]
    ax.errorbar(x, y, yerr, label=label, marker=marker, ls='', mec='k', alpha=alpha, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=c )
    #ax.errorbar(x, y, yerr, label=label, marker=None, ls='-', mec='k', alpha=alpha, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=c )

def resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot=False, A=None, zeta=None, ccMvar='M', fixedAxis=True, zlabel=None, smallRatioYaxis=False, assert_nH=True):
    r = (9,13) if mplot else (-5,-0.5)
    bins = 40

    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='all', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [2, 1]}, figsize=[4.8*3,4.8*1.5], dpi=150)
    for logM1, ax, axr in zip((12, 13, 14), (ax1, ax2, ax3), (ax4, ax5, ax6)):
        yfid = 'HM'
        yerrfid = None
        M1, M2 = 10**logM1, 10**(logM1+0.5)
        ax.set_title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(logM1+0.0, logM1+0.5), y=0.9)

        for cc, sh, centrals_mask, label, marker, c in zip([cc_HM, cc_SV], [sh_HM, sh_SV], [centrals_mask_HM, centrals_mask_SV], ['HM', 'SV'], ['s', 'o'], ['#2402ba', '#98c1d9']):
            x, y, yerr, yerr_log, nH_cores, Mavg = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=SUBHALOMINMASS[label], returnMavg=True, mplot=mplot, A=A, zeta=zeta, Mvar=ccMvar)
            errorbar(ax, x, y, yerr=yerr_log, label='Cores '+label, marker=marker)

            x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh, Mavg_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=SUBHALOMINMASS[label], returnMavg=True, mplot=mplot)
            errorbar(ax, x_sh, y_sh, yerr=yerr_log_sh, label='Subhalos '+label, marker=marker)

            print(reldif(Mavg, Mavg_sh)*100., 'percent relative difference between Mavg_cores and Mavh_sh')

            print('')
            if (ccMvar!='Mtn') and assert_nH:
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
            elif label=='SV':
                # old Mavg_sh = np.mean(sh['M'][bin_mask_sh])
                print('SV lim:', np.log10(100*PARTICLEMASS[label]/Mavg_sh))
                ax.axvline(  np.log10(100*PARTICLEMASS[label]/Mavg_sh), ymax=1., ls='--', c=c )
                axr.axvline( np.log10(100*PARTICLEMASS[label]/Mavg_sh), ls='--', label=r'$\log \left(\mathrm{100m_{p,'+label+ r'}/\langle M \rangle}\right)$', c=c )
        if mplot and fixedAxis:
            ax.set_ylim(-3,3.6) #CHANGE
            ax.set_xlim(9,12.95)
        elif fixedAxis: #finalized for res0,1 plots
            ax.set_ylim(-2.3,3.8)
            ax.set_xlim(-5,-0.5)

        if smallRatioYaxis:
            # axr.axhline(0.0)
            # axr.axhline(1.95)
            axr.set_ylim(0.0,1.95)
    print('ax1 YLIM', ax1.get_ylim())
    ax1.legend(loc=3)
    ax4.legend(loc=2)
    if mplot:
        ax5.set_xlabel(r'$\log(m)$')
        ax1.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m) \right]$')
    else:
        ax5.set_xlabel(r'$\log(m/M)$')
        ax1.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m/M) \right]$')
    ax4.set_ylabel(r'ratio')

    if zlabel:
        # ax1.text(-4.75,2.5,zlabel, fontsize=12)
        ax1.text(r[0]+0.25,2.5,zlabel, fontsize=12)

def paramscan_resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot, outfile, ccMvar='M', ):
    for i, A in enumerate(tqdm(A_arr)):
        for j, zeta in enumerate(zeta_arr):
            resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot, A, zeta, ccMvar)
            plt.savefig(outfile + '_{}_{}.png'.format(A, zeta)) #'/home/isultan/projects/halomassloss/core_catalog_mevolved/Paper2Figs/paramexploration_mplot/z1_{}_{}.png'.format(A, zeta)
            plt.close()

def cosmology_tests(cc_SV, centrals_mask_SV, cc_AQ, centrals_mask_AQ, r=(-4.0,0.), bins=20, A=None, zeta=None, mlim=0, mplot=False, newlegend=False):
    alpha = 1
    fig, (ax, ax1, ax2, ax3) = plt.subplots(4, sharex='all', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [4, 1, 1, 1]}, figsize=[4.8*1,4.8*7/4], dpi=150)

    ### new legend ###
    if newlegend:
        handles = []
        handles.append(ax.errorbar([], [], [], label=' ', marker='o', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[0])) #SV12
        handles.append(ax.errorbar([], [], [], label=' ', marker='o', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[1])) #AQ12
        handles.append(ax.errorbar([], [], [], label=' ', marker='D', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[2])) #SV13
        handles.append(ax.errorbar([], [], [], label=' ', marker='D', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[3])) #AQ13
        handles.append(ax.errorbar([], [], [], label='SV', marker='s', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[4])) #SV14
        handles.append(ax.errorbar([], [], [], label='AlphaQ', marker='s', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[5])) #AQ14
        legend1 = ax.legend(handles=handles, ncol=3, columnspacing=-1.2)

        handles2 = []
        handle2c = 'k'
        handles2.append(ax.errorbar([], [], [], label='[12.0, 12.5]', marker='o', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=handle2c))
        handles2.append(ax.errorbar([], [], [], label='[13.0, 13.5]', marker='D', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=handle2c))
        handles2.append(ax.errorbar([], [], [], label='[14.0, 14.5]', marker='s', ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=handle2c))
        ax.legend(handles=handles2, loc=4, title=r'$\log \left[M / \left(h^{{-1}}M_\odot \right) \right]\in$')#, prop={'size': 6})
        ax.add_artist(legend1)
    ### end new legend ###

    for logM1, axr, marker in zip((12, 13, 14), (ax1,ax2,ax3), ('o', 'D', 's')):
        yfid = 'SV'
        yerrfid = None
        M1, M2 = 10**logM1, 10**(logM1+0.5)

        for cc, centrals_mask, label in zip([cc_SV, cc_AQ], [centrals_mask_SV, centrals_mask_AQ], ['SV', 'AlphaQ']):
            x, y, yerr, yerr_log, nH_cores = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, A, zeta, mlim=mlim, mplot=mplot)
            errorbar(ax, x, y, yerr=yerr_log, label='{} {}'.format(logM1, label), marker=marker, alpha=alpha)

            print('')

            if yfid == label:
                yfid = y
                yerrfid = yerr
            else:
                errorbar(axr, x, 10**(y-yfid), yerr=nratioerr(10**y, yerr, 10**yfid, yerrfid), marker=marker, alpha=1, c='k')
            axr.axhline(1, c='k',ls='--', lw=0.7, zorder=-1)
        print(axr.get_ylim())
        axr.set_ylim(0.5,1.49)
    print(ax.get_xlim())
    ax3.set_xlabel(r'$\log(m)$' if mplot else r'$\log(m/M)$')
    ax.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m) \right]$' if mplot else r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m/M) \right]$')
    ax2.set_ylabel(r'$\mathrm{CMF_{AlphaQ}/CMF_{SV}}$')
    if not newlegend:
        ax.legend()

    # if not mplot: #similar lims as all m m/M plot
    #     ax.set_ylim(-2.4,3.2) #CHANGE
    #     ax.set_xlim(-4.1, -0.2) #CHANGE

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
        print('')
    return ReducedChi2dict

def pcolorplot_gen(ReducedChi2, Mlabel, M1, M2, outfile, avgchi2):
    plt.figure()
    plt.pcolormesh(zeta_arr, A_arr, ReducedChi2, cmap=plt.cm.jet)#, vmin=0.9240790129278156, vmax=542.9304287245984)
    cb = plt.colorbar()
    cb.set_label(r'$\langle\chi^2_{\mathrm{dof}}\rangle$' if avgchi2 else r'$\chi^2_{\mathrm{dof}}$')

    Abfi, zetabfi = np.unravel_index(ReducedChi2.argmin(), ReducedChi2.shape)
    Abf, zetabf =  A_arr[Abfi], zeta_arr[zetabfi]
    print('A', Abf, 'zeta', zetabf)

    mask_ReducedChi2 = mask_ReducedChi2_gen(ReducedChi2)

    plt.plot(zetabf, Abf, 'x', ms=10, zorder=5, c='w')

    mask_ReducedChi2_marr = np.ma.masked_equal(mask_ReducedChi2, False)
    plt.pcolormesh(zeta_arr, A_arr, mask_ReducedChi2_marr, cmap='binary', alpha=.3)

    plt.xlabel('$\zeta$')
    plt.ylabel('$\mathcal{A}$')
    if Mlabel == 'ALL' or Mlabel == 100:
        plt.title(r'$\log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \geq$ {}'.format(np.log10(M1)), y=0.9, color='w')
    elif Mlabel == 12:
        plt.title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(np.log10(M1), np.log10(M2)), y=0.9, color='w')
    else:
        plt.title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(np.log10(M1), np.log10(M2)), y=0.9, color='w')
    print(np.log10(M1), np.log10(M2))
    print(ReducedChi2.min(), ReducedChi2.max())
    print('')

    if outfile:
        plt.savefig(outfile)

def pcolorplots(ReducedChi2dict, M1dict, M2dict, outfile=None, avgchi2=False):
    for Mlabel in sorted(M1dict.keys()):
        ReducedChi2 = ReducedChi2dict[Mlabel]
        M1, M2 = M1dict[Mlabel], M2dict[Mlabel]
        pcolorplot_gen(ReducedChi2, Mlabel, M1, M2, outfile=(f'{outfile}_{Mlabel}.pdf' if outfile else None), avgchi2=avgchi2)

def sigma1plots(cc, sh, centrals_mask, label, rdict, M1dict, M2dict, ReducedChi2dict, bins=20, mlim=0, mplot=False, outfile=None, avgchi2=False, zlabel=None, fixedAxis=False, legendFlag=True, bfparamslabelFlag=False):
    alpha = 1.0
    fixedylim = {12:(-1.6, 0.0), 13:(-0.64, 0.83), 14:(-0.65, 1.8)} #verified min/max for z=0 and z=1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=False, gridspec_kw={'hspace': 0, 'wspace': .15}, figsize=[4.8*3,4.8*1], dpi=150)
    for Mlabel, ax in zip( (12, 13, 14), (ax1, ax2, ax3) ):
        ReducedChi2 = ReducedChi2dict[Mlabel]
        r, M1, M2 = rdict[Mlabel], M1dict[Mlabel], M2dict[Mlabel]

        x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=mlim, mplot=mplot)

        Avals, zvals = np.unravel_index( np.flatnonzero(mask_ReducedChi2_gen(ReducedChi2)), ReducedChi2.shape )
        best_ys = np.zeros((len(Avals), bins), dtype=np.float64)
        print('len(Avals)', len(Avals))
        for i, (A, zeta) in enumerate(zip(A_arr[Avals], zeta_arr[zvals])):
            # print(A, zeta)
            _, y, _, _, _ = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, mplot=mplot, A=A, zeta=zeta, verbose=False)
            best_ys[i] = y

        Abfi, zetabfi = np.unravel_index(ReducedChi2.argmin(), ReducedChi2.shape)
        Abf, zetabf =  A_arr[Abfi], zeta_arr[zetabfi]
        print('Abf: ', Abf, 'zetabf: ', zetabf)
        x, y, yerr, yerr_log, nH_cores = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, mplot=mplot, A=Abf, zeta=zetabf, verbose=True)
        assert nH_cores==nH_sh

        ax.fill_between(x, np.amin(best_ys, axis=0), np.amax(best_ys, axis=0), label=(r'$\Delta \langle\chi^2_{\mathrm{dof}}\rangle \le 1$' if avgchi2 else r'$\Delta \chi^2_{\mathrm{dof}} \le 1$'), alpha=0.5, fc='b')
        errorbar(ax, x_sh, y_sh, yerr=yerr_log_sh, label='Subhalos', c='r', alpha=alpha, marker='s')
        errorbar(ax, x, y, yerr=yerr_log, label=r'Cores ($\min\left(\langle\chi_{\mathrm{dof}}^2\rangle\right)$ parameters)', c='k', alpha=alpha )
        
        if fixedAxis:
            ax.set_ylim(fixedylim[Mlabel])
        print('xlim: ', ax.get_xlim(), '\nylim: ', ax.get_ylim())
        print()
        ax.set_title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(np.log10(M1), np.log10(M2)), y=0.05, x=0.5)
        
        if bfparamslabelFlag:
            ax.text(0.97, 0.97, r'$(\mathcal{A},\zeta)_{\min\left(\langle\chi_{\mathrm{dof}}^2\rangle\right)}=$'+f' ({Abf}, {zetabf})', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=11)

    if legendFlag:
        ax3.legend(loc=1)
    ax2.set_xlabel(r'$\log(m)$' if mplot else r'$\log(m/M)$')
    ax1.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m) \right]$' if mplot else r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m/M) \right]$')

    if zlabel:
        ax1.text(11.15,-1.0,zlabel, fontsize=12)

    if outfile:
        plt.savefig(outfile) #'Paper2Figs/1sigma_z0.pdf'

### SUBHALO-TO-CORE MATCHING ###
def subhalo_core_match(M1, M2, cc, centrals_mask, sh, step, mlim=OBJECTMASSCUT['SV'], SHMLM=SHMLM_SV, A=AFID, zeta=ZETAFID, s1=False, mostMassiveCore=False, assert_nH=True, infall_mass_var='infall_fof_halo_mass'):
    sh['rvir'] = SHMLM.getRvir(sh['subhalo_mass'], SHMLM.step2z[step])
    subhalo_mean_x = periodic_bcs(sh['subhalo_mean_x'], sh['X'], SHMLM.BOXSIZE)
    subhalo_mean_y = periodic_bcs(sh['subhalo_mean_y'], sh['Y'], SHMLM.BOXSIZE)
    subhalo_mean_z = periodic_bcs(sh['subhalo_mean_z'], sh['Z'], SHMLM.BOXSIZE)

    cores_x = periodic_bcs(cc['x'], cc['X'], SHMLM.BOXSIZE)
    cores_y = periodic_bcs(cc['y'], cc['Y'], SHMLM.BOXSIZE)
    cores_z = periodic_bcs(cc['z'], cc['Z'], SHMLM.BOXSIZE)

    bin_mask_sh = (sh['subhalo_tag']!=0)&(M1 <= sh['M'])&(sh['M'] <= M2)&(sh['subhalo_mass']>mlim)
    nH_sh = len(np.unique( sh['fof_halo_tag'][(M1 <= sh['M'])&(sh['M'] <= M2)] ))

    sh_arr = np.vstack((subhalo_mean_x[bin_mask_sh], subhalo_mean_y[bin_mask_sh], subhalo_mean_z[bin_mask_sh])).T
    distance_upper_bound = 2. * sh['rvir'][bin_mask_sh]

    bin_mask_cores = (~centrals_mask)&(M1 <= cc['M'])&(cc['M'] <= M2)&(cc[m_evolved_col(A, zeta)]>mlim)
    if s1:
        bin_mask_cores = bin_mask_cores&(cc['host_core']==cc['CORETAG'])
    fht_fof = (cc['fof_halo_tag']<0)*np.bitwise_and(cc['fof_halo_tag']*-1, 0xffffffffffff) + (cc['fof_halo_tag']>=0)*cc['fof_halo_tag']
    nH_cores = len(np.unique( fht_fof[centrals_mask&(M1 <= cc['M'])&(cc['M'] <= M2)] ))
    print(f'nH_cores: {nH_cores}, nH_sh: {nH_sh}')
    if assert_nH:
        assert nH_cores == nH_sh

    cores_tree = spatial.cKDTree( np.vstack((cores_x[bin_mask_cores], cores_y[bin_mask_cores], cores_z[bin_mask_cores])).T )

    if mostMassiveCore:
        iarr, carr = [], []
        for i in range(len(distance_upper_bound)):
            qres = cores_tree.query_ball_point(sh_arr[i], r=distance_upper_bound[i])
            if len(qres)>0:
                idxmax = qres[ np.argmax(cc[m_evolved_col(A, zeta)][bin_mask_cores][qres]) ]
                iarr.append(i)
                carr.append(idxmax)
        percentexists = len(iarr)/np.sum(bin_mask_sh)*100
        print('{}% of masked subhalos have at least 1 core within their search radius.'.format(percentexists))
        matched_mask_cores = np.flatnonzero(bin_mask_cores)[carr]
        matched_mask_sh = np.flatnonzero(bin_mask_sh)[iarr]
    else:
        dist, idx = [], []
        for i in range(len(distance_upper_bound)):
            dv, iv = cores_tree.query(sh_arr[i], k=2, distance_upper_bound=distance_upper_bound[i])
            dist.append(dv)
            idx.append(iv)
        dist = np.array(dist)
        idx = np.array(idx)

        f1, f2 = (dist != np.inf).T
        fmask = f1^f2

        percentmatch = np.sum(fmask)/np.sum(bin_mask_sh)*100
        f1i = f1[np.invert(fmask)]
        percentmany = np.sum(f1i)/len(f1i)*100
        percentnone = np.sum(np.invert(f1i))/len(f1i)*100
        print('{}% of masked subhalos have 1:1 core match. Of the unmatched subhalos, {}% have mutliple cores and {}% have no core.'.format(percentmatch, percentmany, percentnone))

        matched_mask_cores = np.flatnonzero(bin_mask_cores)[idx[:,0][fmask]]
        matched_mask_sh = np.flatnonzero(bin_mask_sh)[fmask]

    matched_m_cores = cc[m_evolved_col(A, zeta)][matched_mask_cores]
    matched_M_cores = cc['M'][matched_mask_cores]
    matched_ifhm_cores = cc[infall_mass_var][matched_mask_cores]

    matched_m_sh = sh['subhalo_mass'][matched_mask_sh]
    matched_M_sh = sh['M'][matched_mask_sh]

#     assert np.array_equal(matched_M_cores, matched_M_sh)

    return (matched_m_cores, matched_M_cores, nH_cores, matched_ifhm_cores), (matched_m_sh, matched_M_sh, nH_sh), matched_mask_cores, matched_mask_sh

def contour_matched_m(matched_m_cores, matched_m_sh, M1, M2, bins=20, density=True):
    r = ( np.min([np.log10(matched_m_cores), np.log10(matched_m_sh)]), np.max([np.log10(matched_m_cores), np.log10(matched_m_sh)]) )
    H, xedges, yedges = np.histogram2d(np.log10(matched_m_cores), np.log10(matched_m_sh), range=(r,r), bins=bins, density=density)
    assert np.array_equal(xedges,yedges)
    edges = xedges[:-1] + (xedges[1:]-xedges[:-1])/2

    fig = plt.figure(figsize=(4.8, 3.9))
    plt.contourf(edges, edges, H, cmap=plt.cm.inferno)
    cb = plt.colorbar()
    cb.set_label('density' if density else 'counts')

    plt.plot(edges, edges, 'w--', label=r'$m_c=m_{sh}$')
    # plt.legend(loc=4)
    plt.ylabel('$\log m_c$')
    plt.xlabel('$\log m_{sh}$')
    plt.title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(np.log10(M1), np.log10(M2)), y=0.9, color='w')

    fig.axes[0].tick_params(which='both', color='w')
    fig.axes[0].set_aspect('equal')
    fig.axes[0].set_yticks(fig.axes[0].get_xticks()[1:-1])

def histogram_matched_m(matched_ifhm_cores, matched_m_sh, matched_m_cores, M1, M2, bins=100, r=(0.0, 2.1), alpha=0.3, density=True, legend=False, ylim=(0.,3.7)):
    plt.figure()
    plt.hist(reldif(matched_ifhm_cores, matched_m_sh), range=r, bins=bins, alpha=alpha, label=r'$m_i=m_{\mathrm{infall}}$', density=density, color='b')
    plt.hist(reldif(matched_m_cores, matched_m_sh), range=r, bins=bins, alpha=alpha, label=r'$m_i=m_{\mathrm{evolved}}$', density=density, color='r')

    plt.xlabel(r'$\left|m_i-m_{sh}\right|/\left[\left(m_i+m_{sh}\right)/2\right]$')
    plt.ylabel('density' if density else 'counts')
    if legend:
        plt.legend(loc=7)
    print('ylim: ', plt.ylim())
    print('xlim: ', plt.xlim())
    plt.ylim(ylim)
    plt.title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(np.log10(M1), np.log10(M2)), y=0.9)

### Last Journey full sim ###
def hostbins(cc, centrals_mask, M1_th, M2_th, verbose=False, s1=False, Nhalos=10**4, minHalosFlag=False):
    satellites_mask = ~centrals_mask
    M = cc['M'][satellites_mask]
    massmask = (cc['M'][centrals_mask] >= M1_th)&(cc['M'][centrals_mask] < M2_th)
    asort = np.argsort(cc['M'][centrals_mask][massmask])
    if verbose:
        print('Total host halos within mass threshold', np.sum(massmask))

    cim = cc['M'][centrals_mask][massmask][asort]
    ctni = cc['tree_node_index'][centrals_mask][massmask][asort]

    if minHalosFlag:
        Nhalos = np.sum(cim <= cim[Nhalos-1])
    if verbose:
        print('Number of host halos in bin', np.format_float_scientific(Nhalos))

    bs_str = np.format_float_scientific(cim[Nhalos-1] - cim[0])
    if verbose:
        print("Host halo mass bin size:", bs_str)
    M1, M2 = cim[0], cim[Nhalos-1]

    assert len(np.unique(ctni[:Nhalos]))==Nhalos
    isin = np.isin(cc['tree_node_index'][satellites_mask][(M1<=M)&(M<=M2)], ctni[:Nhalos])
    if s1:
        isin = isin&(Coretag[(M1<=M)&(M<=M2)]==cc['host_core'][satellites_mask][(M1<=M)&(M<=M2)])
    bin_mask = np.flatnonzero(satellites_mask)[(M1<=M)&(M<=M2)][isin]
    Nih = np.sum(isin)
    if verbose:
        print("Number of infall halos:", Nih)

    ninfallhosts = len(np.unique(cc['tree_node_index'][bin_mask]))
    if verbose:
        print("Hosts with no infall halos: {} ({}%)".format(Nhalos-ninfallhosts, 100*(1-ninfallhosts/Nhalos)))
        print()

    valun, idxun = np.unique( real_fof_tags(cc['fof_halo_tag'][centrals_mask][massmask][asort][:Nhalos]), return_index=True )
    Mavg = np.mean(cim[:Nhalos][idxun])
    Nhalos = len(valun)
    return bin_mask, Mavg, Nhalos #TODO 10^4 cutoff doesn't work for fragments z=1

# Wide host bin tests for Last Journey full sim
def hostbins_wide(cc, centrals_mask, M1_th, M2_th, verbose=False, s1=False, Nhalos=10**4, minHalosFlag=False):
    satellites_mask = ~centrals_mask
    M = cc['M'][satellites_mask]
    massmask = (cc['M'][centrals_mask] >= M1_th)&(cc['M'][centrals_mask] < M2_th)
    asort = np.argsort(cc['M'][centrals_mask][massmask])
    if verbose:
        print('Total host halos within mass threshold', np.sum(massmask))

    cim = cc['M'][centrals_mask][massmask][asort]
    ctni = cc['tree_node_index'][centrals_mask][massmask][asort]
    if Nhalos == -1:
        Nhalos = np.sum(massmask)
    if minHalosFlag:
        Nhalos = np.sum(cim <= cim[Nhalos-1])
    if verbose:
        print('Number of host halos in bin', np.format_float_scientific(Nhalos))

    bs_str = np.format_float_scientific(cim[Nhalos-1] - cim[0])
    if verbose:
        print("Host halo mass bin size:", bs_str)
    M1, M2 = cim[0], cim[Nhalos-1]

    assert len(np.unique(ctni[:Nhalos]))==Nhalos
    isin = np.isin(cc['tree_node_index'][satellites_mask][(M1<=M)&(M<=M2)], ctni[:Nhalos])
    if s1:
        isin = isin&(Coretag[(M1<=M)&(M<=M2)]==cc['host_core'][satellites_mask][(M1<=M)&(M<=M2)])
    bin_mask = np.flatnonzero(satellites_mask)[(M1<=M)&(M<=M2)][isin]
    Nih = np.sum(isin)
    if verbose:
        print("Number of infall halos:", Nih)

    ninfallhosts = len(np.unique(cc['tree_node_index'][bin_mask]))
    if verbose:
        print("Hosts with no infall halos: {} ({}%)".format(Nhalos-ninfallhosts, 100*(1-ninfallhosts/Nhalos)))
        print()

    valun, idxun = np.unique( real_fof_tags(cc['fof_halo_tag'][centrals_mask][massmask][asort][:Nhalos]), return_index=True )
    Mavg = np.mean(cim[:Nhalos][idxun])
    Nhalos = len(valun)
    return bin_mask, Mavg, Nhalos #TODO 10^4 cutoff doesn't work for fragments z=1

### Host halo concentration effects ###
def concentration_plot(cc, centrals_mask, sh, h1, h2, Med_cdelta, A=AFID, zeta=ZETAFID, label='SV', mlim=OBJECTMASSCUT['SV'], mplot=True, bins=25):
    r = (np.log10(mlim),13) if mplot else (-3,0)

    fig, ax = plt.subplots(1, sharex='all', sharey='row', figsize=[4.8*1,4.8*1], dpi=150)
    for h, marker, plabel in zip( (h1, h2), ('o', 's'), ('($c_{200c}\le'+str(round(Med_cdelta,2))+'$)', '($c_{200c}>'+str(round(Med_cdelta,2))+'$)') ):
        bin_mask = (~centrals_mask)&np.isin(cc['fof_halo_tag'], h)&(cc[m_evolved_col(A, zeta)]>mlim)
        nH_cores_h = len(h)
        
        bin_mask_sh = (sh['subhalo_tag']!=0)&np.isin(sh['fof_halo_tag'], h)&(sh['subhalo_mass']>mlim)
        nH_sh_h = len(np.unique( sh['fof_halo_tag'][np.isin(sh['fof_halo_tag'], h)] ))
        
        assert nH_cores_h==nH_sh_h, 'Different number of hosts for subhalos and cores for concentration bin'
        print('nH_h, sh\t', nH_cores_h)

        parr = cc[m_evolved_col(A, zeta)][bin_mask] / (1.0 if mplot else cc[Mvar][bin_mask])
        x, y, yerr, yerr_log = hist(np.log10(parr), bins=bins, normed=True, plotFlag=False, range=r, normScalar=nH_cores_h, normBinsize=True, normLogCnts=True, retEbars=True)
        errorbar(ax, x, y, yerr=yerr_log, label='Cores '+plabel, marker=marker)
        
        parr_sh = sh['subhalo_mass'][bin_mask_sh] / (1.0 if mplot else sh['M'][bin_mask_sh])
        x_sh, y_sh, yerr_sh, yerr_log_sh = hist(np.log10(parr_sh), bins=bins, normed=True, plotFlag=False, range=r, normScalar=nH_sh_h, normBinsize=True, normLogCnts=True, retEbars=True)
        errorbar(ax, x_sh, y_sh, yerr=yerr_log_sh, label='Subhalos '+plabel, marker=marker)

    ax.set_xlabel(r'$\log(m)$' if mplot else r'$\log(m/M)$')
    ax.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m) \right]$' if mplot else r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m/M) \right]$')
    # ax.set_title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(np.log10(M1), np.log10(M2)), y=0.9)
    ax.legend(loc=3)
    print('xlim', ax.get_xlim())