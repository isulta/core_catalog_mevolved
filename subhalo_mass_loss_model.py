PARTICLES100MASS = 1.148276e11 #h^-1 Msun
BOXSIZE = 256
OMEGA_M = 0.2647926998611387
OMEGA_L = 0.7352073001388613
THUBBLE = 9.78 #h^-1 Gyr
LITTLEH = 0.71
OMEGA_0 = OMEGA_M #1.0
cc_data_dir = '/home/isultan/data/AlphaQ/core_catalog_merg/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof/'

import numpy as np
import os

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

def m_vir(m200c, step):
    """Convert m200c (c is crticial density at z) to virial mass."""
    z = step2z[step]
    return m200c * (delta_vir(z)/200.) * (0.746*(delta_vir(z)/delta_vir(0))**0.395)**-3

def m_evolved(m0, M0, step, step_prev, A=0.86, zeta=0.07):
    # A conversion to fix delta_vir(0) != 178 discrepancy
    A = A * (delta_vir(0)/178.)**0.5

    z = step2z[step]
    delta_t = step2lookback[step_prev] - step2lookback[step]

    if zeta == 0:
        return m0 * np.exp( -delta_t/tau(z,A) )
    else:
        return m0 * ( 1 + zeta * (m0/M0)**zeta * delta_t/tau(z,A) )**(-1/zeta)