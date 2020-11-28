output_merg_virial/
cc_data_dir = '/home/isultan/data/AlphaQ/core_catalog_merg/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_virial/'
A=0.86, zeta=0.07
A scale: A = A * (delta_vir(0)/178.)**0.5
virialFlag = True: infall mass output is M200c->virial conversion applied to FOF mass

output_merg_fof/
cc_data_dir = '/home/isultan/data/AlphaQ/core_catalog_merg/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof/'
A=0.86, zeta=0.07
A scale: A = A * (delta_vir(0)/178.)**0.5
virialFlag = False: FOF mass used

output_merg_fof_Hiroshima/
cc_data_dir = '/home/isultan/data/AlphaQ/core_catalog_merg/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_Hiroshima/'
A,zeta according to Hiroshima et al. 2018: A,zeta depend on M, z
A scale: none
tau_dyn: according to 2016a Eq. 2
virialFlag = False: FOF mass used

output_merg_fof_fitting/
cc_data_dir = '/home/isultan/data/AlphaQ/core_catalog_merg/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_merg_fof_fitting/'
A_arr = [0.25,0.5,1.0,1.2,1.4,1.5,1.6,2.0,4.0,10]
zeta_arr = [0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.2]

output_LJDS_localhost_dtfactor_0.5_fitting/
cc_data_dir = '/home/isultan/data/LJDS/CoreCatalog/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_LJDS_localhost_dtfactor_0.5_fitting/'
A_arr = [0.4, 0.5, 0.6 , 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
zeta_arr = [0.001, 0.005, 0.01, 0.02, 0.04, 0.07, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2]

output_LJDS_localhost_dtfactor_0.5/
cc_data_dir = '/home/isultan/data/LJDS/CoreCatalog/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_LJDS_localhost_dtfactor_0.5/'
A_arr, zeta_arr = [0.9], [0.005]

output_ALCC_localhost_dtfactor_0.5/
cc_data_dir = '/home/isultan/data/ALCC/CoreCatalog/'
cc_output_dir = '/home/isultan/projects/halomassloss/core_catalog_mevolved/output_ALCC_localhost_dtfactor_0.5/'
A_arr, zeta_arr = [0.9], [0.005]
