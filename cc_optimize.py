from scipy.optimize import least_squares, minimize_scalar
import cc_generate
import plot_subhalo_mass_fn
from itk import gio_read_dict
import numpy as np

normLogCnts = False

# initial fitting parameters: 2016a
# x0 = [0.86, 0.07]
x0 = [1, 0]
M1, M2 = 10**14.0, 10**14.5
r = (-6, 0)

fitr = (-3, -1.5)

x_shmf, y_shmf = plot_subhalo_mass_fn.plotSHMF(M1, M2, r, plotFlag=False, normLogCnts=normLogCnts)
mask = (fitr[0]<=x_shmf)&(x_shmf<=fitr[1])


def f(x):
    A, zeta = x[0], x[1]
    print "A={}, zeta={}".format(A, zeta)

    if A==0.86 and zeta==0.07:
        outfile = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof/09_03_2019.AQ.499.corepropertiesextend.hdf5'
        _, y_cmf = plot_subhalo_mass_fn.plotCMF(outfile, M1, M2, s1=True, returnUnevolved=False, r=r, plotFlag=False, normLogCnts=normLogCnts)
    else:
        cc = cc_generate.create_core_catalog_mevolved(virialFlag=False, A=A, zeta=zeta, writeOutputFlag=False)
        _, y_cmf = plot_subhalo_mass_fn.plotCMF('', M1, M2, s1=True, returnUnevolved=False, r=r, plotFlag=False, cc=cc, normLogCnts=normLogCnts)

    return (y_cmf-y_shmf)[mask]

vars_cc = [
'core_tag', 
'tree_node_index',
'infall_mass', 
'infall_step', 
'central', 
'host_core'
]
cc = gio_read_dict('/home/isultan/data/AlphaQ/core_catalog_merg/09_03_2019.AQ.499.coreproperties', vars_cc)

def f_zeta0(x):
    A = x
    print "A={}".format(A)
    _, y_cmf = plot_subhalo_mass_fn.plotCMF('', M1, M2, s1=True, returnUnevolved=False, r=r, plotFlag=False, cc=cc, normLogCnts=normLogCnts, returnZeta0=True, A=A)

    ans = np.linalg.norm( (y_cmf-y_shmf)[mask] )
    print ans
    return ans

if __name__ == "__main__":
    # least_squares(f, x0, verbose=2)
    minimize_scalar(f_zeta0, bounds=(.1, 10), method='bounded')