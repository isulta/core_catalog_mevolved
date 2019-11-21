# core_catalog_mevolved
Appends m_evolved column to given core catalog and saves output core catalog in HDF5 format. m_evolved is initialized as 0 and tracks halo mass loss after core infall according to given subhalo mass loss model. **Currently configured for AlphaQ core catalog.**

## Usage

### Generate core catalog with m_evolved
1. Change cosmology parameters, `cc_data_dir`, and `cc_output_dir` in `subhalo_mass_loss_model.py`
2. Create empty directory `cc_output_dir`

`python cc_generate.py`

### Plot mass functions
See `Notebooks/` for useful Jupyter notebooks. Helper functions for plotting subhalo, core evolved mass, and core unevolved mass functions are provided in `plot_subhalo_mass_fn.py`.

## Subhalo mass loss model
Currently using [Jiang and van den Bosch 2016](https://academic.oup.com/mnras/article/458/3/2848/2589187) model. A different model can be specifed by modifying `subhalo_mass_loss_model.py`.

## Dependencies
- dtk: [cosmo.py](https://github.com/dkorytov/dtk/blob/master/cosmo.py) (included with repo)
- Astropy
- H5Py
- [GenericIO](https://trac.alcf.anl.gov/projects/genericio)
- [itk](https://github.com/isulta/itk)
- [tqdm](https://github.com/tqdm/tqdm) (for progress bar)
- Numpy
- Matplotlib
