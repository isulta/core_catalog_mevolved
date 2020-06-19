PARTICLES100MASS = 1.148276e11 #h^-1 Msun
BOXSIZE = 256
OMEGA_M = 0.2647926998611387
OMEGA_L = 0.7352073001388613
THUBBLE = 9.78 #h^-1 Gyr
LITTLEH = 0.71
OMEGA_0 = OMEGA_M #1.0
COREVIRIALRADIUSFACTOR = 1
SUBHALOVIRIALRADIUSFACTOR = 1
SUBHALOHOSTVIRIALRADIUSFACTOR = 1
CORERADIUSCUT = 20e-3 #20 kpc/h
COREDISTANCEPERCENT = 1#0.05 #5% virial radius of host halo
PARTICLECUTMASS = PARTICLES100MASS*0
SUBHALOMASSCUT = PARTICLES100MASS
DELTATFACTOR = 0.5
FIDUCIALPARTICLECUTMASS = PARTICLES100MASS*1
FIDUCIAL_A = 0.9
FIDUCIAL_ZETA = 0.005
cc_data_dir = '/home/isultan/data/AlphaQ/core_catalog_merg/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting_localhost_dtfactor_0.5_fiducial_A_0.9_zeta_0.005_FIDUCIALPARTICLECUTMASS_PARTICLES100MASS_spline_match_zbin_1.5_2/'

import numpy as np
import os
from itk import periodic_bcs, many_to_one

# step to z/lookback time cache
from cosmo import StepZ
stepz_converter = StepZ(sim_name='AlphaQ')

from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM
cosmoFLCDM = FlatLambdaCDM(H0=71, Om0=0.265, Tcmb0=0, Neff=3.04, m_nu=None, Ob0=0.0448)

steps = sorted([int(i.split('.')[2]) for i in os.listdir(cc_data_dir) if '#' not in i])
step2z = {}
step2lookback = {} #in h^-1 Gyr
for step in steps:
    z = stepz_converter.get_z(step)
    if step == 499:
        z = 0.0
    step2z[step] = z
    step2lookback[step] = cosmoFLCDM.lookback_time(z).value * LITTLEH

def E(z):
    """E(z) = H(z)/H0"""
    return (OMEGA_M*((1+z)**3) + OMEGA_L)**0.5

def Omega(z):
    return OMEGA_0 * (1+z)**3 / E(z)**2

def x(z):
    return Omega(z) - 1

def delta_vir(z):
    return 18*np.pi**2 + 82*x(z) - 39*x(z)**2

def tau(z, A):
    """returns in units h^-1 Gyr"""
    return 1.628/A * ( delta_vir(z)/delta_vir(0) )**(-0.5) * E(z)**(-1)

def delta_t_quad(z):
    """
    Alternate method to compute lookback time.
    Returns in units h^-1 Gyr.
    """
    from scipy.integrate import quad
    return quad( lambda z_: THUBBLE/(E(z_)*(1+z_)), 0, z )[0]

def ratio_R200c_Rvir(z):
    """Returns R200c/Rvir at redshift z."""
    return 0.746*(delta_vir(z)/delta_vir(0))**0.395

def m_vir(m200c, step):
    """Convert m200c (c is crticial density at z) to virial mass."""
    z = step2z[step]
    return m200c * (delta_vir(z)/200.) * ratio_R200c_Rvir(z)**-3

def convertA(A):
    """A conversion to fix delta_vir(0) != 178 discrepancy."""
    return A * (delta_vir(0)/178.)**0.5

def m_evolved_helper(m0, M0, step, step_prev, A, zeta, dtFactorFlag):
    """Sets Giocoli et al. 2008 fitting parameters if none given."""
    if A is None:
        A = 1.628/(2*LITTLEH)
        zeta = 0.06
        print "A not defined!"
    # A = convertA(A)

    z = step2z[step]
    delta_t = step2lookback[step_prev] - step2lookback[step]
    if dtFactorFlag:
        delta_t *= DELTATFACTOR

    if zeta == 0:
        return m0 * np.exp( -delta_t/tau(z,A) )
    else:
        return m0 * ( 1 + zeta * (m0/M0)**zeta * delta_t/tau(z,A) )**(-1/zeta)

def m_evolved_belowcut(m0, step, step_prev, dtFactorFlag, gfunc=[]):
    delta_t = step2lookback[step_prev] - step2lookback[step]
    if dtFactorFlag:
        delta_t *= DELTATFACTOR
    if gfunc == []:
        xarr_poly = np.linspace(0, 100, 1000)
        from scipy.interpolate import interp1d
        #spline
        #f = np.array([4.75487332e-10, -2.20820202e-07,  4.24564520e-05, -4.33037246e-03, 2.47714926e-01, -7.58520065e+00,  9.88160267e+01])

        #spline_zbin_1.5_2
        #f = np.array([-5.94873513e-10,  4.04120234e-07, -9.30366167e-05,  9.90221344e-03, -5.11633271e-01,  1.07159654e+01])

        #spline_match_zbin_1.5_2
        f = np.array([-1.78131330e-11,  8.56347941e-09, -1.73469252e-06,  1.91808802e-04, -1.24982012e-02,  4.80027802e-01, -1.00855673e+01,  9.03464481e+01])

        g = interp1d(np.polyval(f, xarr_poly), xarr_poly, bounds_error=False, fill_value=(np.nan, 0.001))
        gfunc.append(g)
    else:
        g = gfunc[0]
    return g(delta_t + np.polyval(f, m0/(PARTICLES100MASS/100.))) * PARTICLES100MASS/100.

def m_evolved(m0, M0, step, step_prev, A=None, zeta=None, dtFactorFlag=False, useFiducialFlag=False, belowCutSplineFlag=False):
    res = m_evolved_helper(m0, M0, step, step_prev, A, zeta, dtFactorFlag)
    if useFiducialFlag:
        mFidMask = m0>=FIDUCIALPARTICLECUTMASS
        res[mFidMask] = m_evolved_helper(m0[mFidMask], M0[mFidMask], step, step_prev, FIDUCIAL_A, FIDUCIAL_ZETA, dtFactorFlag)
    if belowCutSplineFlag:
        res[np.invert(mFidMask)] = m_evolved_belowcut(m0[np.invert(mFidMask)], step, step_prev, dtFactorFlag)
    return res

def fexp(infall_step, A):
    """Returns array of exponential factors exp(-sum(Delta t_i/tau_i)) for each satellite core with `infall_step`."""
    flipsteps = np.flip(steps)
    fexp_map = np.zeros_like(flipsteps, dtype=np.float32)

    for i in range(2, len(flipsteps)):
        si, si1, si2 = flipsteps[i], flipsteps[i-1], flipsteps[i-2]
        zi2 = step2z[si2]
        delta_t = step2lookback[si1] - step2lookback[si2]
        fexp_map[i] = delta_t/tau(zi2, A)

    # fexp_map is exponential factor for each step in steps
    fexp_map = np.flip( np.exp( -1 * np.cumsum(fexp_map) ) )

    vals, inv_idx = np.unique(infall_step, return_inverse=True)

    return fexp_map[np.isin(steps, vals)][inv_idx]

def fast_m_evolved(psi, infall_step, A):
    """Quickly computes m_evolved for zeta=0 case, given z=0 satellite cores.

    Arguments:
        psi {np 1d array} -- m_infall/M for satellites at z=0
        infall_step {np 1d array} -- infall_step of cores in `psi`
        A {float} -- fitting parameter

    Returns:
        np 1d array -- m_evolved of given cores
    """
    A = convertA(A)
    return psi * fexp(infall_step, A)

def getR200(Mfof):
    """Computes R200 at z=0: radius of SO halo of mass `Mfof` with density 200*critical density(z=0).
    Mfof must have units Msun/h.
    Returned R200 has units Mpc/h.
    """
    return np.power(4.302e-15*Mfof, 1./3)

def getRvir_0(Mfof):
    """Computes Rvir at z=0: radius of SO halo of mass `Mfof` with density Delta_virial(0)*critical density(0).
    Mfof must have units Msun/h.
    Returned Rvir_0 has units Mpc/h.
    """
    return np.power(4.302e-15*200/delta_vir(0)*Mfof, 1./3)

def getRvir(Mfof, z):
    """Computes Rvir at z: radius of SO halo of mass `Mfof` with density Delta_virial(z)*critical density(z).
    Mfof must have units Msun/h.
    Returned Rvir has units Mpc/h.
    """
    pc = (cosmoFLCDM.critical_density(z).to('Msun* Mpc**(-3)')/LITTLEH**2).value
    return np.power(Mfof*3/(4*np.pi*delta_vir(z)*pc), 1./3)

def dist(x,y,z,x0,y0,z0):
    """Find distance between two points (x,y,z) and (x0, y0, z0)."""
    return np.sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )

def disruption_mask(cc_satellites, criteria, M, X, Y, Z, z):
    """Returns np bool array of shape `cc_satellites` with disruption criteria (None, int, or list) implemented on satellite cores."""
    if criteria is None:
        criteria = []
    elif type(criteria) is int:
        criteria = [criteria]

    # criterion: (1) core radius is less than 20 kpc/h
    radius_mask = cc_satellites['radius'] < CORERADIUSCUT

    # criterion: (2) remove merged cores
    merged_mask = cc_satellites['merged'] != 1

    # criterion: (3) cores are distance r >= COREDISTANCEPERCENT*Rvir(z) from central core
    Rvir = getRvir(M, z)
    x, y, z = periodic_bcs(cc_satellites['x'], X, BOXSIZE), periodic_bcs(cc_satellites['y'], Y, BOXSIZE), periodic_bcs(cc_satellites['z'], Z, BOXSIZE)
    distance_mask = dist(x, y, z, X, Y, Z) >= (COREDISTANCEPERCENT*Rvir)

    masks_dict = { 1:radius_mask, 2:merged_mask, 3:distance_mask}#, 4:mergedCoreTag_mask }

    # criterion: (4) cores with a nonzero `mergedCoreTag` are removed
    if 4 in criteria:
        mergedCoreTag_mask = cc_satellites['mergedCoreTag']==0
        masks_dict[4] = mergedCoreTag_mask


    mask = np.full_like(cc_satellites['x'], True, bool)
    for i in criteria:
        mask = mask&masks_dict[i]

    return mask

def core_mask(cc, M1, M2, s1=False, disrupt=None, z=0, idx_m21=None):
    """Given a core catalog and filtering criteria, returns indices in cc of filtered satellite cores.

    Arguments:
        cc {dict} -- core catalog
        M1, M2 {float} -- Host halos with M in mass range [M1, M2], where M is central's `infall_mass` from core catalog.
        disrupt {None, int, or list} -- If given, applies core filtering criteria defined in SHMLM when filtering substructure. (default: {None})
        z {int} -- Redshift (default: {0})

    Keyword Arguments:
        s1 {bool} -- If true, consider only 1st order substructure (i.e. subhalos). (default: {False})

    Returns:
        np 1d array -- indices of satellite filtered cores in `cc`
        np 1d array -- M array (corresponds with satellite filtered cores)
        integer -- number of host halos (M)
    """
    satellites_mask = cc['central'] == 0
    centrals_mask = cc['central'] == 1

    # Create M array (corresponds with cc[satellites_mask]) to be host tree node mass of each satellite
    if idx_m21 is None:
        idx_m21 = many_to_one( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask] )
    M = cc['infall_mass'][centrals_mask][idx_m21]
    if 'x' in cc.keys():
        X, Y, Z = cc['x'][centrals_mask][idx_m21], cc['y'][centrals_mask][idx_m21], cc['z'][centrals_mask][idx_m21]

    mask = np.flatnonzero( (M1 <= M) & (M <= M2) )#& (cc['infall_mass'][satellites_mask] >= PARTICLES100MASS) )
    if s1:
        Coretag = cc['core_tag'][centrals_mask][idx_m21]
        mask = np.intersect1d( mask, np.flatnonzero(cc['host_core'][satellites_mask]==Coretag) )
    if disrupt is not None:
        cc_satellites = { k:cc[k][satellites_mask] for k in cc.keys() }
        mask = np.intersect1d( mask, np.flatnonzero(disruption_mask(cc_satellites, disrupt, M, X, Y, Z, z)) )

    nHalo = np.sum( (M1<=cc['infall_mass'][centrals_mask])&(cc['infall_mass'][centrals_mask]<=M2) )

    if 'x' in cc.keys():
        return np.flatnonzero(satellites_mask)[mask], M[mask], X[mask], Y[mask], Z[mask], nHalo
    else:
        return np.flatnonzero(satellites_mask)[mask], M[mask], nHalo
