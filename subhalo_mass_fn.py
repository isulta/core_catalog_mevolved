import numpy as np
import matplotlib.pyplot as plt
import h5py
import subhalo_mass_loss_model as SHMLM

def read_dict_from_disk(outfile):
	hf = h5py.File(outfile, 'r')
	cc = {}
	for k in hf['coredata'].keys():
		cc[k] = np.array( hf['coredata'][k] )
	hf.close()

	return cc

def plt_latex():
	# Latex for plt
	from matplotlib import rc
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

def SHMF_plot(outfile, M1, M2, bins, step):

	# Load z=0 extended core catalog
	cc = read_dict_from_disk(outfile)

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

	# m/M array for SHMF
	plot_arr = cc['m_evolved'][satellites_mask]/M

	# M1 <= M <= M2 mask and infall mass >= 100 Particles mask
	plot_arr = plot_arr[ (M1 <= M) & (M <= M2) & (cc['infall_mass'][satellites_mask] >= SHMLM.PARTICLES100MASS) ]

	# plot histogram
	plt_latex()
	plt.figure(dpi=120)
	plt.hist(np.log10(plot_arr), bins=bins, histtype='bar', density=False)
	plt.yscale('log')

	plt.xlabel(r'$\log(m/M)$')
	plt.ylabel(r'$\mathrm{d}n/\mathrm{d} \log(m/M)$')
	plt.title( 'z=' + str(round(SHMLM.step2z[step],3)) + r', {} $\leq$ log(M/$h^{{-1}}M_\odot$)$\leq$ {}'.format(np.log10(M1),np.log10(M2)) )
	plt.show()

if __name__ == '__main__':
	SHMF_plot(outfile='output/09_03_2019.AQ.499.corepropertiesextend.hdf5', M1=10**14.0, M2=10**14.5, bins=500, step=499)