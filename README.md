# core_catalog_mevolved
Appends m_evolved column to given core catalog and saves output core catalog in HDF5 format. m_evolved is initialized as 0 and tracks halo mass loss after core infall according to given subhalo mass loss model. **Currently configured for AlphaQ core catalog.**

## Usage

### Generate core catalog with m_evolved
1. Change cosmology parameters, `cc_data_dir`, and `cc_output_dir` in `subhalo_mass_loss_model.py`
2. Create empty directory `cc_output_dir`

`python cc_generate.py`

### Plot subhalo mass function using m_evolved
`python plot_subhalo_mass_fn.py`

## Subhalo mass loss model
Currently using [Jiang and van den Bosch 2016](https://academic.oup.com/mnras/article/458/3/2848/2589187) model. A different model can be specifed by modifying `subhalo_mass_loss_model.py`.

## Dependencies
- [cosmo.py](https://github.com/dkorytov/dtk/blob/master/cosmo.py) (included with repo)
- Astropy
- H5Py
- [tqdm](https://github.com/tqdm/tqdm) (for progress bar)
- Numpy
- Matplotlib