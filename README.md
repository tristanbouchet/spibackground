# spibackground

Python modules to create background for INTEGRAL/SPI

Required libraries: numpy, scipy, astropy, tqdm. Optional: matplotlib

## Create background for an observation

obs_background.py creates background spectra for an obs using DB.

it can be called directly with "python obs_background.py" after main_dir has been changed, or imported as a module.

## Create the initial DB

background_db creates the initial DB from parameters stored in .sav files

