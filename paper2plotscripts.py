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

def resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot=False, A=None, zeta=None, ccMvar='M', fixedAxis=True, zlabel=None, smallRatioYaxis=False, assert_nH=True, r=None):
    if r is None:
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
        ax1.text(r[0]+0.25,2.5,zlabel, fontsize=12)

def paramscan_resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot, outfile, ccMvar='M', fixedAxis=True, zlabel=None, smallRatioYaxis=False):
    for i, A in enumerate(tqdm(A_arr)):
        for j, zeta in enumerate(zeta_arr):
            resolution_tests(cc_HM, sh_HM, centrals_mask_HM, cc_SV, sh_SV, centrals_mask_SV, mplot, A, zeta, ccMvar, fixedAxis, zlabel, smallRatioYaxis)
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
def mask_ReducedChi2_gen(ReducedChi2, maxDelta):
    return np.abs(ReducedChi2 - ReducedChi2.min()) <= maxDelta

def ReducedChi2dict_gen(cc, sh, centrals_mask, label, rdict, M1dict, M2dict, bins=20, mlim=0, mplot=False, dlog=False):
    ReducedChi2dict = {}
    for Mlabel in sorted(rdict.keys()):
        r = rdict[Mlabel]
        M1, M2 = M1dict[Mlabel], M2dict[Mlabel]
        _, _, _, _, nH_cores = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, mplot=mplot)
        x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=mlim, mplot=mplot)
        assert nH_cores == nH_sh, 'nH_cores != nH_sh'

        Chi2 = np.empty((len(A_arr), len(zeta_arr)))
        for i, A in enumerate(A_arr):
            for j, zeta in enumerate(zeta_arr):
                x, y, yerr, yerr_log, _ = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, A=A, zeta=zeta, verbose=False, mplot=mplot, nH_cores=nH_cores)
                Chi2[i,j] = np.sum( (y_sh-y)**2/(yerr_log**2 + yerr_log_sh**2) ) if dlog else np.sum( (10**y_sh-10**y)**2/(yerr**2 + yerr_sh**2) )
        ReducedChi2 = np.true_divide(Chi2 , bins-2)
        assert np.isfinite(ReducedChi2).all(), 'ReducedChi2 not finite'
        ReducedChi2dict[Mlabel] = ReducedChi2
        print('')
    return ReducedChi2dict

def pcolorplot_gen(ReducedChi2, Mlabel, M1, M2, outfile, avgchi2, markfiducialparams, ReducedChi2_outline0, ReducedChi2_outline1, maxDelta):
    plt.figure()
    ax = plt.gca()
    plt.pcolormesh(zeta_arr, A_arr, ReducedChi2, cmap=plt.cm.jet, shading='nearest')
    cb = plt.colorbar()
    cb.set_label(r'$\langle\chi^2_{\mathrm{dof}}\rangle$' if avgchi2 else r'$\chi^2_{\mathrm{dof}}$')

    Abfi, zetabfi = np.unravel_index(ReducedChi2.argmin(), ReducedChi2.shape)
    Abf, zetabf =  A_arr[Abfi], zeta_arr[zetabfi]
    print('A', Abf, 'zeta', zetabf)

    mask_ReducedChi2 = mask_ReducedChi2_gen(ReducedChi2, maxDelta)

    plt.plot(zetabf, Abf, 'x', ms=10, zorder=5, c='w')
    if markfiducialparams:
        plt.plot(ZETAFID, AFID, '*', ms=10, zorder=5, c='w')

    mask_ReducedChi2_marr = np.ma.masked_equal(mask_ReducedChi2, False)
    plt.pcolormesh(zeta_arr, A_arr, mask_ReducedChi2_marr, cmap='binary', alpha=.3, shading='nearest')

    for ReducedChi2_outline, c_outln, hs, ls in zip((ReducedChi2_outline0, ReducedChi2_outline1), (COLOR_SCHEME[3],COLOR_SCHEME[-1]), ('///','\\\\\\'), ('-','--')):
        if ReducedChi2_outline is not None:
            mask_ReducedChi2_outline = mask_ReducedChi2_gen(ReducedChi2_outline, maxDelta)
            ax.add_collection(LineCollection(binaryarray_outline(mask_ReducedChi2_outline, zeta_arr, A_arr), linewidths=1.5, colors=c_outln, linestyle=ls))
            # Hatch plot
            # plt.rcParams['hatch.color'] = '#808080'
            # mask_ReducedChi2_marr_outline = np.ma.masked_equal(mask_ReducedChi2_outline, False)
            # ax.pcolor(zeta_arr, A_arr, mask_ReducedChi2_marr_outline, shading='nearest', cmap='binary', hatch=hs, alpha=0)
    
    # optional tick marks that are at the parameter search grid points
    '''plt.xticks(zeta_arr,rotation=60, fontsize=10)
    plt.yticks(A_arr)
    ax.tick_params(which='minor', length=0)
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0]'''
    ax.tick_params(axis='y', which='both', color='w')

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

def pcolorplots(ReducedChi2dict, M1dict, M2dict, outfile=None, avgchi2=False, markfiducialparams=False, ReducedChi2dict_outline0=None, ReducedChi2dict_outline1=None, maxDelta=DELTACHIDOF2MAX):
    for Mlabel in sorted(M1dict.keys()):
        ReducedChi2 = ReducedChi2dict[Mlabel]
        M1, M2 = M1dict[Mlabel], M2dict[Mlabel]
        ReducedChi2_outline0 = ReducedChi2dict_outline0[Mlabel] if ReducedChi2dict_outline0 else None
        ReducedChi2_outline1 = ReducedChi2dict_outline1[Mlabel] if ReducedChi2dict_outline1 else None
        pcolorplot_gen(ReducedChi2, Mlabel, M1, M2, (f'{outfile}_{Mlabel}.pdf' if outfile else None), avgchi2, markfiducialparams, ReducedChi2_outline0, ReducedChi2_outline1, maxDelta)

def sigma1plots(cc, sh, centrals_mask, label, rdict, M1dict, M2dict, ReducedChi2dict, bins=20, mlim=0, mplot=False, outfile=None, avgchi2=False, zlabel=None, fixedAxis=False, legendFlag=True, bfparamslabelFlag=False, maxDelta=DELTACHIDOF2MAX):
    alpha = 1.0
    # fixedylim = {12:(-1.6, 0.0), 13:(-0.64, 0.83), 14:(-0.65, 1.8)} #verified min/max for SV z=0 and z=1
    fixedylim = {12:(-1.56, 0.0), 13:(-0.85, 0.84), 14:(-0.85, 1.71)} #verified min/max for HM z=0 and z=1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=False, gridspec_kw={'hspace': 0, 'wspace': .15}, figsize=[4.8*3,4.8*1], dpi=150)
    for Mlabel, ax in zip( (12, 13, 14), (ax1, ax2, ax3) ):
        ReducedChi2 = ReducedChi2dict[Mlabel]
        r, M1, M2 = rdict[Mlabel], M1dict[Mlabel], M2dict[Mlabel]

        x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=mlim, mplot=mplot)

        Avals, zvals = np.unravel_index( np.flatnonzero(mask_ReducedChi2_gen(ReducedChi2, maxDelta)), ReducedChi2.shape )
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

        ax.fill_between(x, np.amin(best_ys, axis=0), np.amax(best_ys, axis=0), label=(r'$\Delta \langle\chi^2_{\mathrm{dof}}\rangle' if avgchi2 else r'$\Delta \chi^2_{\mathrm{dof}}')+f' \le {maxDelta}$', alpha=0.5, fc='b')
        errorbar(ax, x_sh, y_sh, yerr=yerr_log_sh, label='Subhalos', c='r', alpha=alpha, marker='s')
        errorbar(ax, x, y, yerr=yerr_log, label=r'Cores ($\min\left(\langle\chi_{\mathrm{dof}}^2\rangle\right)$ parameters)', c='k', alpha=alpha )
        print('y_sh', y_sh)
        print('y', y)
        if fixedAxis:
            ax.set_ylim(fixedylim[Mlabel])
        print('xlim: ', ax.get_xlim(), '\nylim: ', ax.get_ylim())
        print()
        if not bfparamslabelFlag:
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
def host_concentration_bins_gen(sod, M1, M2):
    '''Given an `sod` halo catalog, splits the halos with `fof_halo_mass` in [`M1`, `M2`] and c>0 into two `sod_halo_cdelta` (i.e. c) bins.
    `Med_cdelta` is the median c of the positive concentration halos in the mass bin.
    `h1` is array of `fof_halo_tag`s of positive concentration halos in the mass bin with c <= `Med_cdelta`.
    `h2` is array of `fof_halo_tag`s of positive concentration halos in the mass bin with c > `Med_cdelta`.
    '''
    assert not np.any(sod['sod_halo_cdelta']==0), 'cdelta==0 exists'

    mbin_mask = inrange(sod['fof_halo_mass'], (M1, M2))
    nH_mbin = np.sum(mbin_mask)                           # Number of halos in mass bin

    mbin_posc_mask = mbin_mask&(sod['sod_halo_cdelta']>0) # Halos in mass bin with cdelta>0
    nH_mbin_posc = np.sum(mbin_posc_mask)                 # Number of such halos

    print(f"In the {np.log10((M1,M2))} mass bin, there are {nH_mbin_posc:,} halos with cdelta>0 out of {nH_mbin:,} halos ({nH_mbin_posc/nH_mbin*100}%).")

    Med_cdelta = np.median( sod['sod_halo_cdelta'][mbin_posc_mask] )
    print(f'Median cdelta (in cdelta>0 mass bin): {Med_cdelta}')
    print('range(cdelta)', np.min(sod['sod_halo_cdelta'][mbin_posc_mask]), np.max(sod['sod_halo_cdelta'][mbin_posc_mask]))

    half1_mask = sod['sod_halo_cdelta'][mbin_posc_mask] <= Med_cdelta
    h1 = sod['fof_halo_tag'][mbin_posc_mask][half1_mask]
    h2 = sod['fof_halo_tag'][mbin_posc_mask][~half1_mask]

    print(f'Concentration bin 1 (cdelta <= Med_cdelta) contains {len(h1):,} host halos.')
    print(f'Concentration bin 2 (cdelta > Med_cdelta) contains {len(h2):,} host halos.')
    
    return h1, h2, Med_cdelta

def concentration_plot(cc, centrals_mask, sh, h1, h2, Med_cdelta, A=AFID, zeta=ZETAFID, mlim=OBJECTMASSCUT['SV'], mplot=True, bins=25):
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

### Fig1 ###
def hosts_in_sh(sod, sh, SHMLM, z, ra_list=[(10**12.0, 10**12.5), (10**13.0, 10**13.5), (10**14.0, 10**14.5)], printNoMissing=False):
    '''Finds host halos in the sod catalog with c>0 in the defined mass bins that are missing from the subhalo catalog.'''
    assert (sod['sod_halo_cdelta']!=0).all(), 'c==0 exists!'
    assert len(np.unique(sod['fof_halo_tag']))==len(sod['fof_halo_tag']), 'sod halos are not unique!'
    sod_pc = {k:sod[k][sod['sod_halo_cdelta']>0].copy() for k in sod}

    sod_pc['M200_to_Mvir'] = M200_to_Mvir(sod_pc['sod_halo_mass'], sod_pc['sod_halo_cdelta'], SHMLM, z)
    
    for ra in ra_list:
        ra_inrange = inrange(sod_pc['M200_to_Mvir'], ra)
        nH_ra = np.sum(ra_inrange)
        missing_ra_mask = ~np.isin(sod_pc['fof_halo_tag'][ra_inrange], sh['fof_halo_tag'])
        N_missing_ra = np.sum(missing_ra_mask)
        if printNoMissing or (N_missing_ra>0):
            print(f'For bin {np.log10(ra)}, there are {N_missing_ra} missing host halos.', end=' ')
            print(f'This is {N_missing_ra/nH_ra*100}% of the {nH_ra} host halos in this bin (for which c>0).')
            print(f"Missing halos (fof_halo_tag): {sod_pc['fof_halo_tag'][ra_inrange][missing_ra_mask]}")
            print(f"Missing halos (log10 fof_halo_mass): {np.log10(sod_pc['fof_halo_mass'][ra_inrange][missing_ra_mask])}")
            print(f"Missing halos (log10 sod_halo_mass): {np.log10(sod_pc['sod_halo_mass'][ra_inrange][missing_ra_mask])}")
            print(f"Missing halos (log10 M200_to_Mvir): {np.log10(sod_pc['M200_to_Mvir'][ra_inrange][missing_ra_mask])}\n")

def sh_Mfof_convert(sh, fof_halo_tag, sod_halo_cdelta, sod_halo_mass, SHMLM, assert_x0_unique=True, z=0, step=499, b=0.168, Delta=200):
    isin_clookup = np.isin(sh['fof_halo_tag'], fof_halo_tag)
    assert np.all(isin_clookup)
    print(f"Subhalos with host in cdelta lookup table: {isin_clookup.sum()/len(sh['fof_halo_tag'])*100}% of all subhalos")
    # print(f"Corresponding halos with tag in cdelta lookup table: {len(np.unique(sh['fof_halo_tag'][isin_clookup]))/len(np.unique(sh['fof_halo_tag']))*100}% of unique halos")
    # missingM = sh['M'][np.unique(sh['fof_halo_tag'][~isin_clookup], return_index=True)[1]]
    assert (sod_halo_cdelta!=0).all(), 'c=0 exists'
    m21_clookup = many_to_one(sh['fof_halo_tag'][isin_clookup], fof_halo_tag, assert_x0_unique=assert_x0_unique)
    CDELTA = sod_halo_cdelta[m21_clookup]
    print(f'subhalos (including centrals) found in sod catalog that have negative c: {(CDELTA<0).sum()} out of {len(CDELTA)} ({(CDELTA<0).sum()/len(CDELTA)*100} %)')
    sh_res = {k:sh[k][isin_clookup][CDELTA>0].copy() for k in sh.keys()}
    sh_res['CDELTA'] = CDELTA[CDELTA>0]
    
    sh_res['Mfof_to_M200'] = Mfof_to_Mso(SHMLM, z, sh_res['CDELTA'], sh_res['M'], b, Delta)
    sh_res['Mfof_to_M200_to_Mvir'] = SHMLM.m_vir(sh_res['Mfof_to_M200'], step)
    sh_res['M200'] = sod_halo_mass[m21_clookup][CDELTA>0]
    print(sh_res['CDELTA'].min(), sh_res['CDELTA'].max())
    sh_res['M200_to_Mvir'] = M200_to_Mvir(sh_res['M200'], sh_res['CDELTA'], SHMLM, z)
    print()
    return sh_res

def A_c(c):
    return np.log(1+c) - c/(1+c)

def Mfof_to_Mso(SHMLM, z, c, Mfof, b, Delta):
    Delta_ISO = 3/(2*np.pi*b**3)*SHMLM.Omega(z)
    Mfof_Mdelta_factor = 1/A_c(c) * ( np.log(c) + 1/3*np.log(Delta/(3*A_c(c)*Delta_ISO)) + 1/c*(Delta/(3*A_c(c)*Delta_ISO))**(-1/3) - 1 )
    return Mfof/Mfof_Mdelta_factor

def f_xvir(xvir, c200, Delta):
    return 1/A_c(c200)*(np.log(1+c200*xvir) - c200*xvir/(1+c200*xvir)) - Delta/200*(xvir**3)

def M200_to_Mvir(M200, c200, SHMLM, z):
    xf = np.linspace(2,20, 50)
    rootres_int = root(f_xvir, args=(xf, SHMLM.delta_vir(z)), x0=np.ones_like(xf))
    assert rootres_int.success, 'root finder failed'
    yf = rootres_int.x
    fin = interp1d(xf, yf, fill_value='extrapolate')
    
    valun, invun = np.unique(c200, return_inverse=True)
    assert (valun!=0).all()
    nullmask = (valun<0)
    
    xvir = np.zeros_like(valun, dtype=type(M200[0]))
    xvir[~nullmask] = fin(valun[~nullmask])
    xvir[nullmask] = 0
    xvir = xvir[invun]
    return M200 * xvir**3 * SHMLM.delta_vir(z)/200

def sh_finder_comparison(
    sh_HM, sh_SV, sh_AQ, Mvar, setoriglims=True, computeRatioFromSHMF=False, addFiducialModels=False, addvlinelim=False, vlinepartlim=50., binwidth=0.5, point_colors=COLOR_SCHEME[:3]):
    r = (-3,0)
    bins = 30
    alpha = 1.0

    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='all', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [2, 1]}, figsize=[4.8*3,4.8*1.5], dpi=150)
    fig5_plot(None, ax3, legendFlag=True)
    ax3.legend(loc=3)
    for logM1, ax, axr in zip((12, 13, 14), (ax1, ax2, ax3), (ax4, ax5, ax6)):
        M1, M2 = 10**logM1, 10**(logM1+binwidth)
        ax.set_title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(logM1+0.0, logM1+binwidth), y=0.9)
        if addFiducialModels:
            if ax == ax1:
                ax.plot(np.linspace(*r, 100), fitting_model(np.linspace(*r, 100), logM1), '--', color='k', lw=1, label="Fiducial function (Equation 1)")
                ax.legend(loc=3)
            else: 
                ax.plot(np.linspace(*r, 100), fitting_model(np.linspace(*r, 100), logM1), '--', color='k', lw=1)
        for sh, label, marker, c in zip([sh_HM, sh_SV, sh_AQ], ['HM', 'SV', 'AQ'], ['s', 'o', 'D'], point_colors):
            x, y, yerr, yerr_log, nH_sh, Mavg_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=SUBHALOMINMASS[label], Mvar=Mvar, returnMavg=True)

            errorbar(ax, x, y, yerr=yerr_log, label=f'Subhalos {label}', marker=marker, alpha=alpha, c=c)
            errorbar(axr, x, 10**(y-fitting_model(x, logM1)), yerr=yerr/10**fitting_model(x, logM1), marker=marker, alpha=alpha, zerocut=True, c=c)

            axr.axhline(1, c='k',ls='--', lw='1', zorder=-2)
            if label=='SV' and addvlinelim and (ax!=ax3):
                vlinelim = np.log10(vlinepartlim*PARTICLEMASS[label]/Mavg_sh) 
                print('SV lim:', vlinelim)
                ax.axvline(  vlinelim, ymax=0.9, ls=':', c=c, alpha=.7 )
                axr.axvline( vlinelim, ls=':', label=r'$\log \left(\mathrm{'+str(int(vlinepartlim))+r'm_{p,'+label+ r'}/\langle M_{'+label+r'} \rangle}\right)$', c=c, alpha=.7 )
                if axr==ax4:
                    axr.legend(loc=2)

        if ax == ax2:
            ax.legend(loc=3)
        fig5_plot(logM1, ax)
        fig5_plot(logM1, axr, ratioFlag=True, computeRatioFromSHMF=computeRatioFromSHMF)
    ax5.set_xlabel(r'$\log(m/M)$')
    ax1.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m/M) \right]$')
    ax4.set_ylabel(r'ratio')
    if setoriglims:
        ax1.set_ylim(-3.4, 1.7)
        ax4.set_ylim(-0.2, 3.4)
        ax1.set_xlim(-3.2,0)
    else:
        ax1.set_ylim(-3.4, 1.71)
        ax4.set_ylim(-0.2, 2.2)
    print('ax1 ylim', ax1.get_ylim())
    print('ax4 ylim', ax4.get_ylim())
    print('ax1 xlim', ax1.get_xlim())

### high-z ###
def ReducedChi2dict_gen_highz(cc, sh, centrals_mask, label, rdict, M1dict, M2dict, bins=20, mlim=0, mplot=False, dlog=False, zeta=ZETAFID):
    ReducedChi2dict = {}
    for Mlabel in sorted(rdict.keys()):
        r = rdict[Mlabel]
        M1, M2 = M1dict[Mlabel], M2dict[Mlabel]
        _, _, _, _, nH_cores = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, mplot=mplot)
        x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=mlim, mplot=mplot)
        # assert nH_cores == nH_sh, 'nH_cores != nH_sh'
        print(f'RELDIF: {reldif(nH_cores, nH_sh)*100.}% between nH_cores and nH_sh for Mlabel:{Mlabel}.')

        Chi2 = np.empty(len(A_arr))
        for i, A in enumerate(A_arr):
            x, y, yerr, yerr_log, _ = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, A=A, zeta=zeta, verbose=False, mplot=mplot, nH_cores=nH_cores)
            Chi2[i] = np.sum( (y_sh-y)**2/(yerr_log**2 + yerr_log_sh**2) ) if dlog else np.sum( (10**y_sh-10**y)**2/(yerr**2 + yerr_sh**2) )
        ReducedChi2 = np.true_divide(Chi2 , bins-1)
        assert np.isfinite(ReducedChi2).all(), 'ReducedChi2 not finite'
        ReducedChi2dict[Mlabel] = ReducedChi2
        print('')
    return ReducedChi2dict

def bestA_highz_gen(ReducedChi2_SV_highz):
    bestA_arr = []
    A_min_arr = []
    A_max_arr = []
    for i, ReducedChi2 in enumerate(ReducedChi2_SV_highz):
        #loop over z=0 to z=len(ReducedChi2_SV_highz)-1
        print(f'z={i}')
        print(f'ReducedChi2.min: {ReducedChi2.min()}')
        
        bestA = A_arr[ReducedChi2.argmin()]
        bestA_arr.append(bestA)
        print(f'best A: {bestA}')

        mask_ReducedChi2 = mask_ReducedChi2_gen(ReducedChi2, DELTACHIDOF2MAX)
        idx_mask_ReducedChi2 = np.flatnonzero(mask_ReducedChi2)
        assert np.array_equal(idx_mask_ReducedChi2[:-1]+1, idx_mask_ReducedChi2[1:]), 'All good A values are not contiguous in A_arr.'
        
        print(A_arr[mask_ReducedChi2])
        A_min_arr.append(A_arr[mask_ReducedChi2].min())
        A_max_arr.append(A_arr[mask_ReducedChi2].max())
        
        print()
    
    bestA_arr = np.array(bestA_arr)
    A_min_arr = np.array(A_min_arr)
    A_max_arr = np.array(A_max_arr)

    A_min_arr = bestA_arr - A_min_arr
    A_max_arr -= bestA_arr

    redshift_arr = np.arange(len(ReducedChi2_SV_highz))

    return redshift_arr, bestA_arr, np.array([A_min_arr, A_max_arr])

def resolution_tests_highz(cc_SV, sh_SV, centrals_mask_SV,
mplot=False, A=None, zeta=None, ccMvar='M', fixedAxis=True, zlabel=None, smallRatioYaxis=False, assert_nH=True, r=None, dlM=0.5, logMlist=(12, 13, 14), mlim=SUBHALOMINMASS['SV'], 
bins=40, ax_xlim=None, ax_ylim=None, draw_vline=True):
    if r is None:
        r = (9,13) if mplot else (-5,-0.5)

    fig, (axtop,axlow) = plt.subplots(2, len(logMlist), sharex='all', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [2, 1]}, figsize=[4.8*len(logMlist),4.8*1.5], dpi=150)
    for logM1, ax, axr in zip(logMlist, axtop, axlow):
        yfid = 'SV'
        yerrfid = None
        M1, M2 = 10**logM1, 10**(logM1+dlM)
        ax.set_title(r'{} $\le \log \left[ M / \left(h^{{-1}}M_\odot \right) \right] \le$ {}'.format(logM1+0.0, logM1+dlM), y=0.9)

        for cc, sh, centrals_mask, label, marker, c in zip([cc_SV], [sh_SV], [centrals_mask_SV], ['SV'], ['o'], ['k']):
            x, y, yerr, yerr_log, nH_cores, Mavg = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, returnMavg=True, mplot=mplot, A=A, zeta=zeta, Mvar=ccMvar)
            errorbar(ax, x, y, yerr=yerr_log, label='Cores '+label, marker=marker)

            x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh, Mavg_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=mlim, returnMavg=True, mplot=mplot)
            errorbar(ax, x_sh, y_sh, yerr=yerr_log_sh, label='Subhalos '+label, marker=marker)

            print(reldif(Mavg, Mavg_sh)*100., 'percent relative difference between Mavg_cores and Mavh_sh')

            print('')
            print(f'RELDIF: {reldif(nH_cores, nH_sh)*100.}%')
            if (ccMvar!='Mtn') and assert_nH:
                assert nH_cores == nH_sh

            if yfid == label:
                yfid = y_sh
                yerrfid = yerr_sh

            errorbar(axr, x, 10**(y-yfid), yerr=nratioerr(10**y, yerr, 10**yfid, yerrfid), marker=marker, c=c, zerocut=True)
            axr.axhline(1, c='k',ls='--', lw=1, zorder=-1)

            if mplot and draw_vline:
                ax.axvline(  np.log10(100*PARTICLEMASS[label]), ymax=1., ls='--', c=c )
                axr.axvline( np.log10(100*PARTICLEMASS[label]), ls='--', label=r'$\log \left(\mathrm{100m_{p,'+label+ r'}}\right)$', c=c )
            elif label=='SV' and draw_vline:
                print('SV lim:', np.log10(100*PARTICLEMASS[label]/Mavg_sh))
                ax.axvline(  np.log10(100*PARTICLEMASS[label]/Mavg_sh), ymax=1., ls='--', c=c )
                axr.axvline( np.log10(100*PARTICLEMASS[label]/Mavg_sh), ls='--', label=r'$\log \left(\mathrm{100m_{p,'+label+ r'}/\langle M \rangle}\right)$', c=c )
        if ax_xlim:
            ax.set_xlim(ax_xlim)
        if ax_ylim:
            ax.set_ylim(ax_ylim)
        
        if mplot and fixedAxis:
            ax.set_ylim(-3,3.6) #CHANGE
            ax.set_xlim(9,12.95)
        elif fixedAxis: #finalized for res0,1 plots
            ax.set_ylim(-2.3,3.8)
            ax.set_xlim(-5,-0.5)

        if smallRatioYaxis:
            axr.set_ylim(0.0,1.95)
    print('ax1 YLIM', axtop[0].get_ylim())
    print('ax1 XLIM', axtop[0].get_xlim())
    axtop[0].legend(loc=3)
    axlow[0].legend(loc=2)
    if mplot:
        axlow[len(axlow)//2].set_xlabel(r'$\log(m)$')
        if len(axlow)%2==0:
            axlow[len(axlow)//2-1].set_xlabel(r'$\log(m)$')
        axtop[0].set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m) \right]$')
    else:
        axlow[len(axlow)//2].set_xlabel(r'$\log(m/M)$')
        if len(axlow)%2==0:
            axlow[len(axlow)//2-1].set_xlabel(r'$\log(m/M)$')
        axtop[0].set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m/M) \right]$')
    axlow[0].set_ylabel(r'CMF/SHMF')

    if zlabel:
        axtop[0].text(r[0]+0.1,0.5,zlabel, fontsize=11)

def highz_plot(cc, sh, centrals_mask, A=None, zeta=None, zlabel=None, r=(np.log10(OBJECTMASSCUT['SV']), 13.12), dlM=0.5, logMlist=(12, 13), mlim=OBJECTMASSCUT['SV'], bins=20, newlegend=True, ylabels=True, markers=('o', 'D')):
    fig, (ax, ax1, ax2) = plt.subplots(3, sharex='all', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0, 'height_ratios': [4, 1, 1]}, figsize=[4.8*1,4.8*6/4], dpi=150)
    label = 'SV'
    ### new legend ###
    if newlegend:
        handles = []
        handles.append(ax.errorbar([0], [0], [0], label=' ', marker=markers[0], ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[0])) #cores12
        handles.append(ax.errorbar([0], [0], [0], label=' ', marker=markers[0], ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[1])) #sh12
        handles.append(ax.errorbar([0], [0], [0], label='Cores', marker=markers[1], ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[2])) #cores13
        handles.append(ax.errorbar([0], [0], [0], label='Subhalos', marker=markers[1], ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=COLOR_SCHEME[3])) #sh13
        legend1 = ax.legend(handles=handles, ncol=2, columnspacing=-1.2)

        handles2 = []
        handle2c = 'k'
        handles2.append(ax.errorbar([0], [0], [0], label='[12.0, 12.5]', marker=markers[0], ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=handle2c))
        handles2.append(ax.errorbar([0], [0], [0], label='[13.0, 13.5]', marker=markers[1], ls='', mec='k', alpha=1, mew=0.5, ms=20**0.5, capsize=4, elinewidth=1.5, c=handle2c))
        ax.legend(handles=handles2, loc=4, title=r'$\log \left[M / \left(h^{{-1}}M_\odot \right) \right]\in$')#, prop={'size': 6})
        ax.add_artist(legend1)
    ### end new legend ###
    for logM1, axr, marker, c_cores, c_sh in zip(logMlist, (ax1, ax2), (markers[0], markers[1]), (COLOR_SCHEME[0], COLOR_SCHEME[2]), (COLOR_SCHEME[1], COLOR_SCHEME[3])):
        M1, M2 = 10**logM1, 10**(logM1+dlM)

        x, y, yerr, yerr_log, nH_cores, Mavg = cores_plot(cc, centrals_mask, M1, M2, label, bins, r, mlim=mlim, returnMavg=True, mplot=True, A=A, zeta=zeta)
        errorbar(ax, x, y, yerr=yerr_log, label=f'Cores {label} {logM1}', marker=marker, c=c_cores)

        x_sh, y_sh, yerr_sh, yerr_log_sh, nH_sh, Mavg_sh = subhalo_plot(sh, M1, M2, label, bins, r, mlim=mlim, returnMavg=True, mplot=True)
        errorbar(ax, x_sh, y_sh, yerr=yerr_log_sh, label=f'Subhalos {label} {logM1}', marker=marker, c=c_sh)

        print(reldif(Mavg, Mavg_sh)*100., 'percent relative difference between Mavg_cores and Mavh_sh')
        print(f'RELDIF: {reldif(nH_cores, nH_sh)*100.}% between nH_cores and nH_sh for logM1:{logM1}.')

            
        errorbar(axr, x, 10**(y-y_sh), yerr=nratioerr(10**y, yerr, 10**y_sh, yerr_sh), marker=marker, c='k', zerocut=True)
        axr.axhline(1, c='k',ls='--', lw=0.7, zorder=-1)
        axr.set_ylim(0.5,1.49)
        
    ax.set_xlim((np.log10(OBJECTMASSCUT['SV']),13.12))
    ax.set_ylim((-3.84,1.3011))
    
    # ax.legend(loc=3)

    ax2.set_xlabel(r'$\log(m)$')

    if ylabels:
        ax.set_ylabel(r'$\log \left[ \mathrm{d}n/\mathrm{d} \log(m) \right]$')
        ax2.set_ylabel(r'CMF/SHMF')

    zlabel += '\n\n$\mathcal{A}=$ '+str(A) + '\n$\zeta=$ '+str(zeta)
    ax.text(r[0]+0.1,-3.4,zlabel, fontsize=11)
    