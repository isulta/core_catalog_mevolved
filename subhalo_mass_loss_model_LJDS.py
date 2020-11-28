'''
PARTICLES100MASS = 1.252e11 #h^-1 Msun
BOXSIZE = 250
OMEGA_M = 0.310
OMEGA_L = 0.690
THUBBLE = 9.78 #h^-1 Gyr
LITTLEH = 0.6766
OMEGA_0 = OMEGA_M #1.0
COREVIRIALRADIUSFACTOR = 1
SUBHALOVIRIALRADIUSFACTOR = 1
SUBHALOHOSTVIRIALRADIUSFACTOR = 1
CORERADIUSCUT = 20e-3 #20 kpc/h
COREDISTANCEPERCENT = 1#0.05 #5% virial radius of host halo
PARTICLECUTMASS = PARTICLES100MASS*1
SUBHALOMASSCUT = PARTICLES100MASS
DELTATFACTOR = 0.5
FIDUCIALPARTICLECUTMASS = PARTICLES100MASS*1
'''
SIMNAME = 'SV'
import itk
PARTICLES100MASS = itk.SIMPARAMS[SIMNAME]['PARTICLEMASS']*100.
BOXSIZE = itk.SIMPARAMS[SIMNAME]['Vi']
OMEGA_M = itk.SIMPARAMS[SIMNAME]['OMEGA_M']
OMEGA_L = itk.SIMPARAMS[SIMNAME]['OMEGA_L']
LITTLEH = itk.SIMPARAMS[SIMNAME]['h']
OMEGA_0 = OMEGA_M
DELTATFACTOR = 0.5

cc_data_dir = '/home/isultan/data/LJDS/CoreCatalog/'
# cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_LJDS_localhost_dtfactor_0.5_fitting2/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_SV_localhost_dtfactor_0.5_sod/'

import numpy as np
import os
from itk import periodic_bcs, many_to_one

# step to z/lookback time cache
#from cosmo import StepZ
#stepz_converter = StepZ(sim_name='AlphaQ')

from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM
cosmoFLCDM = FlatLambdaCDM(H0=LITTLEH*100, Om0=OMEGA_M,  Tcmb0=0, Neff=3.04, m_nu=None, Ob0=None)

import re
steps = sorted([int(re.split('\-|\.', i)[1]) for i in os.listdir(cc_data_dir) if '#' not in i])
zarr = [10.04, 9.81, 9.56, 9.36, 9.15, 8.76, 8.57, 8.39, 8.05, 7.89,7.74, 7.45, 7.31, 7.04, 6.91, 6.67, 6.56, 6.34, 6.13, 6.03, 5.84,5.66, 5.48, 5.32, 5.24, 5.09, 4.95, 4.74, 4.61, 4.49, 4.37, 4.26,4.10, 4.00, 3.86, 3.76, 3.63,  3.55, 3.43, 3.31, 3.21, 3.10,3.04,2.94, 2.85, 2.74, 2.65, 2.58, 2.48, 2.41, 2.32, 2.25, 2.17, 2.09,2.02, 1.95, 1.88, 1.80, 1.74, 1.68, 1.61, 1.54, 1.49, 1.43, 1.38,1.32, 1.26, 1.21, 1.15, 1.11, 1.06, 1.01, 0.96, 0.91, 0.86, 0.82,0.78, 0.74, 0.69, 0.66, 0.62, 0.58, 0.54, 0.50, 0.47, 0.43, 0.40,0.36, 0.33, 0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.13, 0.10, 0.07,0.05,0.02, 0.00]
step2z = {}
step2lookback = {} #in h^-1 Gyr
for i,step in enumerate(steps):
    z = zarr[i]
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

def m_evolved(m0, M0, step, step_prev, A=None, zeta=None, dtFactorFlag=False):
    """Sets Giocoli et al. 2008 fitting parameters if none given."""
    if A is None:
        A = 1.628/(2*LITTLEH)
        zeta = 0.06
        print("A not defined!")
    # A = convertA(A)

    z = step2z[step]
    delta_t = step2lookback[step_prev] - step2lookback[step]
    if dtFactorFlag:
        delta_t *= DELTATFACTOR

    if zeta == 0:
        return m0 * np.exp( -delta_t/tau(z,A) )
    else:
        return m0 * ( 1 + zeta * (m0/M0)**zeta * delta_t/tau(z,A) )**(-1/zeta)

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

def core_mask(cc, M1, M2, s1=False, disrupt=None, z=0):
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
    idx_m21 = many_to_one( cc['tree_node_index'][satellites_mask], cc['tree_node_index'][centrals_mask] )
    M = cc['infall_mass'][centrals_mask][idx_m21]
    X, Y, Z = cc['x'][centrals_mask][idx_m21], cc['y'][centrals_mask][idx_m21], cc['z'][centrals_mask][idx_m21]
    CDELTA = cc['infall_sod_halo_cdelta'][centrals_mask][idx_m21]

    mask = np.flatnonzero( (M1 <= M) & (M <= M2) )#& (cc['infall_mass'][satellites_mask] >= PARTICLES100MASS) )
    if s1:
        Coretag = cc['core_tag'][centrals_mask][idx_m21]
        mask = np.intersect1d( mask, np.flatnonzero(cc['host_core'][satellites_mask]==Coretag) )
    if disrupt is not None:
        cc_satellites = { k:cc[k][satellites_mask] for k in cc.keys() }
        mask = np.intersect1d( mask, np.flatnonzero(disruption_mask(cc_satellites, disrupt, M, X, Y, Z, z)) )

    nHalo = np.sum( (M1<=cc['infall_mass'][centrals_mask])&(cc['infall_mass'][centrals_mask]<=M2) )

    return np.flatnonzero(satellites_mask)[mask], M[mask], X[mask], Y[mask], Z[mask], CDELTA[mask], nHalo
