PARTICLES100MASS = 1.148276e11 #h^-1 Msun
BOXSIZE = 256
OMEGA_M = 0.2647926998611387
OMEGA_L = 0.7352073001388613
THUBBLE = 9.78 #h^-1 Gyr
LITTLEH = 0.71
OMEGA_0 = OMEGA_M #1.0
cc_data_dir = '/home/isultan/data/AlphaQ/core_catalog_merg/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_Hiroshima/'

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
    return 1.628/A * ( delta_vir(z)/178 )**(-0.5) * E(z)**(-1)

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

def m_evolved(m0, M0, step, step_prev, A=None, zeta=None):
    """Sets 2016a fitting parameters if none given."""
    # if A is None:
    #     A = 0.86
    #     zeta = 0.07
    # A = convertA(A)

    z = step2z[step]
    delta_t = step2lookback[step_prev] - step2lookback[step]

    logM0 = np.log10(M0/LITTLEH)
    A = 10**((-0.0003*logM0+0.02)*z + 0.011*logM0 - 0.354)
    zeta = (0.00012*logM0-0.0033)*z-0.0011*logM0+0.026

    return (zeta==0)*(m0 * np.exp( -delta_t/tau(z,A) )) + (zeta!=0)*(m0 * ( 1 + zeta * (m0/M0)**zeta * delta_t/tau(z,A) )**(-1/zeta))

    # if zeta == 0:
    #     return m0 * np.exp( -delta_t/tau(z,A) )
    # else:
    #     return m0 * ( 1 + zeta * (m0/M0)**zeta * delta_t/tau(z,A) )**(-1/zeta)

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

def dist(x,y,z,x0,y0,z0):
    """Find distance between two points (x,y,z) and (x0, y0, z0)."""
    return np.sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )

def disruption_mask(cc_satellites, criteria, M, X, Y, Z):
    """Returns np bool array of shape `cc_satellites` with disruption criteria (None, int, or list) implemented on satellite cores."""
    if criteria is None:
        criteria = []
    elif type(criteria) is int:
        criteria = [criteria]

    # criterion: (1) core radius is less than 20 kpc/h
    radius_mask = cc_satellites['radius'] < 20e-3

    # criterion: (2) remove merged cores
    merged_mask = cc_satellites['merged'] != 1

    # criterion: (3) cores are distance r >= 0.05*Rvir_0 from central core
    Rvir_0 = getRvir_0(M)
    x, y, z = periodic_bcs(cc_satellites['x'], X, BOXSIZE), periodic_bcs(cc_satellites['y'], Y, BOXSIZE), periodic_bcs(cc_satellites['z'], Z, BOXSIZE)
    distance_mask = dist(x, y, z, X, Y, Z) >= (0.05*Rvir_0)

    masks_dict = { 1:radius_mask, 2:merged_mask, 3:distance_mask }

    mask = np.full_like(cc_satellites['x'], True, bool)
    for i in criteria:
        mask = mask&masks_dict[i]
    
    return mask

def core_mask(cc, M1, M2, s1=False, disrupt=None):
    """Given a core catalog and filtering criteria, returns indices in cc of filtered satellite cores.
    
    Arguments:
        cc {dict} -- core catalog
        M1, M2 {float} -- Host halos with M in mass range [M1, M2], where M is central's `infall_mass` from core catalog.
        disrupt {None, int, or list} -- If given, applies core filtering criteria defined in SHMLM when filtering substructure. (default: {None})
    
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
    
    mask = np.flatnonzero( (M1 <= M) & (M <= M2) )#& (cc['infall_mass'][satellites_mask] >= PARTICLES100MASS) )
    if s1:
        Coretag = cc['core_tag'][centrals_mask][idx_m21]
        mask = np.intersect1d( mask, np.flatnonzero(cc['host_core'][satellites_mask]==Coretag) )
    if disrupt is not None:
        cc_satellites = { k:cc[k][satellites_mask] for k in cc.keys() }
        X, Y, Z = cc['x'][centrals_mask][idx_m21], cc['y'][centrals_mask][idx_m21], cc['z'][centrals_mask][idx_m21]
        mask = np.intersect1d( mask, np.flatnonzero(disruption_mask(cc_satellites, disrupt, M, X, Y, Z)) )
    
    nHalo = np.sum( (M1<=cc['infall_mass'][centrals_mask])&(cc['infall_mass'][centrals_mask]<=M2) )
    
    return np.flatnonzero(satellites_mask)[mask], M[mask], nHalo