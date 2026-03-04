'''
Module to create the background Data Base
it converts the line+continuum parameters stored in the .sav file into counts spectra stored in FITS file
this takes more memory space (~1 MB per rev), but saves time by avoiding re-calculation of the background for each run
'''

from astropy.io import fits
import numpy as np
from scipy.io import readsav
from scipy.special import erfc, log_ndtr
from tqdm import tqdm
# default libraries
import os
from glob import glob
import functools
from time import time
from datetime import datetime


GAUSSCONST = np.sqrt(np.pi/2)
LOGGAUSSCONST = np.log(GAUSSCONST) 

def timer(func):
    '''add @timer before function call to print computation time'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        elapsed = time() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

def order_path_list(spec_params_path_list):
    '''order the list of path by energy, using the file name'''
    spec_params_file_list = [f.split('/')[-1] for f in spec_params_path_list]
    # use the upper energy range written in file name to find the sorted index
    file_order_idx = np.argsort([float(f.split('_')[4]) for f in spec_params_file_list])
    return [spec_params_path_list[i] for i in file_order_idx]

class EnergyConversion:
    '''Static methods for energy-index conversions'''
    @staticmethod
    def idx_to_energy_SE(x):
        return 18.25 + x * 0.5
    
    @staticmethod
    def idx_to_energy_HE(x):
        return 18.25 + x

def log_erfc(x):
    '''logarithm of the complementary error function
    relies on the logarithm of gaussian cumulative distribution (log_ndtr) from scipy
    this handles very large/small values well
    faster alternative: np.log(erfcx(x)) - x**2 -> but not numerically stable
    '''
    return np.log(2.0) + log_ndtr(-x * np.sqrt(2.0))

def distorted_gauss(E, A, E0, sig, tau):
    '''convolve line shape (gaussian with exponential)
    A is in counts/bin
    the log of the convolution is computed first to avoid numerical issue with multiplication
    of very large and very small numbers
    sometimes A is negative, so we use: x = sign(x) * exp(ln|x|)
    '''
    if A==0.: #  or sig<0 or tau<0
        return np.zeros_like(E)
    # if sig<=0 or tau<=0:
    #     print(f'sig={sig} tau={tau}')
    log_line = LOGGAUSSCONST + np.log(np.abs(A) * sig / tau) + (2*tau*(E-E0) + sig**2) / (2*tau**2)\
            + log_erfc((tau * (E - E0) + sig**2) / (np.sqrt(2) * sig * tau))
    return np.sign(A) * np.exp(log_line)

def power_law(E, Em, C0, alpha):
    '''C0 in counts/bin'''
    return C0 * (E/Em)**alpha

#################### Background models ####################

class BkgModel:
    '''superclass background model
    defined for a single specified energy band
    '''
    def __init__(self, Em):
        self.Em = Em
    
    def init_params(self, params):
        self.params = params
        self.n_par = len(self.params)

    def calc(self, E):
        raise NotImplementedError
    
    def __call__(self, E):
        return self.calc(E)

class ClsPLModel(BkgModel):
    '''
    convolve line shape (gaussian with exponential) + power-law continuum
    use a different (A,E0,sig,tau) for each line
    result is in counts/bin
    '''
    def calc(self, E):
        if (self.n_par - 2)%4 != 0:
            raise IndexError
        n_lines = (self.n_par - 2)//4
        cont = power_law(E, self.Em, *self.params[:2])
        all_lines = np.array([distorted_gauss(E, *self.params[2+4*l: 2+4*(l+1)]) for l in range(n_lines)])
        return {'cont':cont, 'lines':all_lines}
    
class Cls2PLModel(BkgModel):
    '''
    convolve line shape (gaussian with exponential) + power-law continuum
    use a different (A,E0) for each line, but the same (sig,tau) for all lines
    result is in counts/bin
    '''
    def calc(self, E):
        if (self.n_par - 3)%2 != 0:
            raise IndexError
        n_lines = (self.n_par - 3)//2
        cont = power_law(E, self.Em, *self.params[:2])
        all_lines = np.array([distorted_gauss(E, self.params[2+2*l], self.params[3+2*l], self.params[4+2*l], self.params[-1])\
                               for l in range(n_lines)])
        return {'cont':cont, 'lines':all_lines}


BKG_MODELS = {
    'cls_plaw_function': ClsPLModel,
    'cls_plaw_function2': Cls2PLModel,
}
"""Dictionary mapping function name in .sav files with the class name"""

#################### Make background files from parameters ####################

class BkgEband:
    '''
    background defined for a specific energy band
    '''
    def __init__(self, evt_type, spec_param_dir, spec_param_file):
        self.evt_type = evt_type
        # load the .sav IDL file
        self.spec_param_file = spec_param_file
        self.param_sav = readsav(f"{spec_param_dir}{spec_param_file}")

        # fetch useful info
        self.orbits = self.param_sav['orbits']
        self.fct_name = self.param_sav['fit_func'].decode("utf-8")
        self.Em = self.param_sav['xc']
        self.idx_range = self.param_sav['x_idx_range']
        # Make energy range (in keV)
        if evt_type=='SE' or 'PSD':
            self.E_range = EnergyConversion.idx_to_energy_SE(self.idx_range)
            self.bin_size = .5
        elif evt_type=='HE':
            self.E_range = EnergyConversion.idx_to_energy_HE(self.idx_range)
            self.bin_size = 1.
        else: raise ValueError(f'Event type {evt_type} not implemented.')

        # params 4D table (value/error, param, detector, revolution)
        self.params_table = self.param_sav['spec_params_det']
        # initialize the background model with energy band center (Em) and parameters
        if self.fct_name not in BKG_MODELS:
            raise ValueError(f"Unknown background model: {self.fct_name}. Available models: {list(BKG_MODELS.keys())}")
        self.model = BKG_MODELS[self.fct_name](self.Em)

    def calc_spec_rev_det_eband(self,  nrev, ndet, E=None, plot=False):
        '''
        calculate spectrum for a specific revolution and detector
        '''
        # convert revolution number into the right index
        idx_rev_list = np.where(self.orbits==nrev)[0]
        if len(idx_rev_list)==0:
            # print(f'rev {nrev} not in file {self.spec_param_file}!')
            return False
        else:
            idx_rev = idx_rev_list[0]
        if E is None:
            E = self.E_range
        # select parameter list values for specific det and rev
        params_list = self.params_table[0, :, ndet, idx_rev]
        self.model.init_params(params_list)
        # Calculate the background components in counts/bins
        spec_per_bin = self.model.calc(E)
        # Convert from counts/bin to counts/keV using bin size (depends on evt type)
        self.spec_dico = {comp:spec_per_bin[comp]/self.bin_size for comp in spec_per_bin.keys()}
        # add the lines and continuum, for checking result
        self.lines_sum = self.spec_dico['lines'].sum(axis=0)
        self.total_spec = self.spec_dico['cont'] + self.lines_sum
        if plot: self.plot(E)
        return True
    
    def plot(self, E):
        import matplotlib.pyplot as plt
        fig, ax= plt.subplots(1,1,figsize=(8,6))
        ax.plot(E, self.spec_dico['cont'], label='continuum', color='grey', linestyle='--')
        for l in self.spec_dico['lines']:
            ax.plot(E, l+self.spec_dico['cont'])
        ax.plot(E, self.total_spec, color='k', label='total')
        ax.legend()
        return ax


class BkgList:
    '''
    contains a list of backgrounds for all the energy range in counts/keV
    handles overlapping energy bands by averaging in overlap regions
    resulting backgrounds are saved inside 1 FITS file per rev, containing 1 detector per extension 
    '''
    def __init__(self, spec_params_path_list, evt_type='SE'):
        self.evt_type=evt_type
        if evt_type=='SE' or evt_type=='PSD':
            self.bin_size = .5
            self.n_detectors = 19
            self.idx_range = np.arange(3964)
            self.E_range = EnergyConversion.idx_to_energy_SE(self.idx_range)
        elif evt_type=='HE':
            self.bin_size = 1.
            self.n_detectors = 19
            self.idx_range = np.arange(7982)
            self.E_range = EnergyConversion.idx_to_energy_HE(self.idx_range)
        else:
            raise NotImplementedError(evt_type)
        # load background for every energy band
        self.bkg_range_list = [BkgEband(self.evt_type, '', f) for f in spec_params_path_list]
        self.fct_list = [bkg.fct_name for bkg in self.bkg_range_list]
        self.Em_array = np.array([bkg.Em for bkg in self.bkg_range_list])
        self._build_overlap_array()
    
    def _build_overlap_array(self):
        '''
        Count how many bands contribute to each global energy index
        this takes into account the overlaps and empty energy bins
        '''
        self.n_contributors = np.zeros(len(self.idx_range), dtype=int)
        for bkg in self.bkg_range_list:
            mask = np.isin(self.idx_range, bkg.idx_range)
            self.n_contributors[mask] += 1
    
    def calc_spec_rev_det(self, nrev, ndet, plot=False):
        '''calculate background spec components (continuum and lines) for specific rev/detector, handling overlaps'''
        self.nrev, self.ndet = nrev, ndet
        
        # Accumulate continuum and lines separately
        cont_spec = np.zeros(len(self.idx_range))
        sumlines_spec = np.zeros(len(self.idx_range))
        
        n_eband = 0
        # Goes over all the energy band in the list for 1 rev/1 det
        for bkg in self.bkg_range_list:
            # compute background for 1 energy band
            is_calc = bkg.calc_spec_rev_det_eband(nrev=nrev, ndet=ndet, E=None, plot=False)

            # if energy band not defined, skip iteration
            if not is_calc: continue
            else: n_eband+=1
            
            # Extract components
            continuum = bkg.spec_dico['cont']
            sumlines = bkg.spec_dico['lines'].sum(axis=0)
            
            # Map to global indices and accumulate
            mask = np.isin(self.idx_range, bkg.idx_range)
            global_indices = np.where(mask)[0]
            cont_spec[global_indices] += continuum
            sumlines_spec[global_indices] += sumlines
        
        if n_eband==0:
            return None
        # Average in overlapping regions
        cont_spec = cont_spec / np.maximum(self.n_contributors, 1)
        sumlines_spec = sumlines_spec / np.maximum(self.n_contributors, 1)
        
        # Keep only bins with contributors
        valid_mask = self.n_contributors > 0
        self.E_range_merged = self.E_range[valid_mask]
        self.cont_spec = cont_spec[valid_mask]
        self.sumlines_spec = sumlines_spec[valid_mask]
        self.total_spec = self.cont_spec + self.sumlines_spec
        
        if plot: self.plot()
        return {'cont': self.cont_spec, 'sumlines': self.sumlines_spec}
    
    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.E_range_merged, self.total_spec, color='k', label='total')
        ax.set_xlabel('E (keV)')
        ax.set_ylabel('rate (ct/kev)')
        ax.set_title(f'rev {self.nrev} det {self.ndet}')
        ax.legend()
        ax.loglog()
        return ax
    
    def get_available_revolutions(self):
        '''get list of unique revolution numbers available in the data'''
        orbits_set = set(self.bkg_range_list[0].orbits)
        for bkg in self.bkg_range_list[1:]:
            orbits_set = orbits_set.intersection(set(bkg.orbits))
        return np.array(sorted(orbits_set))
    
    def write_fits_files(self, output_dir='./', revolutions=None, compress=False):
        '''write FITS files for each revolution with background spectra for each detector'''
        os.makedirs(output_dir, exist_ok=True)
        # if no revolution given, take all the revolutions available from .sav param files
        if revolutions is None:
            revolutions = self.get_available_revolutions()
        
        file_ext = '.fits.gz' if compress else '.fits'
        valid_mask = self.n_contributors > 0
        E_merged = self.E_range[valid_mask]

        # Energy extension
        # THIS IS WASTING SOME SPACE (if same energy for all revs)
        energy_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name='E', format='D', unit='keV', array=E_merged),
            fits.Column(name='E_LO', format='D', unit='keV', array=E_merged - self.bin_size/2),
            fits.Column(name='E_HI', format='D', unit='keV', array=E_merged + self.bin_size/2)
        ])
        energy_hdu.header['EXTNAME'] = 'ENERGY'
        
        for nrev in tqdm(revolutions):
            cont_array = np.zeros((self.n_detectors, E_merged.shape[0]))
            lines_array = np.zeros((self.n_detectors, E_merged.shape[0]))
            
            for ndet in range(self.n_detectors):
                spec_dict = self.calc_spec_rev_det(nrev, ndet, plot=False)
                if spec_dict is None:
                    print(f'No background for rev {nrev}.')
                    break
                cont_array[ndet, :] = spec_dict['cont']
                lines_array[ndet, :] = spec_dict['sumlines']
            
            # Create FITS file
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['SATEL'] = 'INTEGRAL'
            primary_hdu.header['INST'] = 'SPI'
            primary_hdu.header['TYPE'] = self.evt_type
            primary_hdu.header['REV'] = (nrev, 'Revolution number')
            primary_hdu.header['NDET'] = (self.n_detectors, 'Number of detectors')
            primary_hdu.header['NEBIN'] = (len(E_merged), 'Number of energy bins')
            primary_hdu.header['EMIN'] = (E_merged[0], 'Minimum energy (keV)')
            primary_hdu.header['EMAX'] = (E_merged[-1], 'Maximum energy (keV)')
            primary_hdu.header['DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            primary_hdu.header['AUTHOR'] = 'tbouchet'
            
            # Continuum extension
            cols_cont = [fits.Column(name=f'DET{ndet}', format='D', unit='ct/keV', array=cont_array[ndet, :]) 
                        for ndet in range(self.n_detectors)]
            cont_hdu = fits.BinTableHDU.from_columns(cols_cont)
            cont_hdu.header['EXTNAME'] = 'CONTINUUM'
            
            # Lines extension
            cols_lines = [fits.Column(name=f'DET{ndet}', format='D', unit='ct/keV', array=lines_array[ndet, :]) 
                        for ndet in range(self.n_detectors)]
            lines_hdu = fits.BinTableHDU.from_columns(cols_lines)
            lines_hdu.header['EXTNAME'] = 'LINES'
            
            # write to file
            hdul = fits.HDUList([primary_hdu, energy_hdu, cont_hdu, lines_hdu])
            filename = f'{output_dir}/bkg_rate_rev_{nrev:04d}_{self.evt_type}{file_ext}'
            hdul.writeto(filename, overwrite=True)
        
        # Create metadata FITS file
        meta_primary_hdu = fits.PrimaryHDU()
        meta_primary_hdu.header['SATEL'] = 'INTEGRAL'
        meta_primary_hdu.header['INST'] = 'SPI'
        meta_primary_hdu.header['TYPE'] = self.evt_type
        meta_primary_hdu.header['DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        meta_primary_hdu.header['AUTHOR'] = 'tbouchet'
        
        # Create valid revolutions extension
        valid_rev_list = np.zeros(3000, dtype=int)
        valid_rev_list[revolutions - 1] = 1  # -1 because revolutions are 1-indexed
        rev_list = np.arange(1, 3001)
        
        cols_valid = [
            fits.Column(name='REV', format='J', array=rev_list),
            fits.Column(name='VALID', format='J', array=valid_rev_list)
        ]
        valid_hdu = fits.BinTableHDU.from_columns(cols_valid)
        valid_hdu.header['EXTNAME'] = 'VALID_REV'
        
        # Write metadata file
        hdul_meta = fits.HDUList([meta_primary_hdu, valid_hdu, energy_hdu])
        meta_filename = f'{output_dir}/info_rev_bkg_{self.evt_type}{file_ext}'
        hdul_meta.writeto(meta_filename, overwrite=True)



def make_det_livetime_fits(sav_file, fits_file):
    '''
    Create FITS file from spi_det_hi data with detector live times
    '''

    # Load SAV file
    spi_det_hi = readsav(sav_file)
    rdx_det_time = spi_det_hi['rdx']
    
    # Create primary HDU with header info
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['AUTHOR'] = 'tbouchet'
    primary_hdu.header['DATE'] = datetime.now().strftime('%Y-%m-%d')
    
    # Create table HDU for rdx (revolution index)
    col_rdx = fits.Column(name='RDX', format='J', array=rdx_det_time)
    rdx_hdu = fits.BinTableHDU.from_columns([col_rdx], name='RDX')
    
    # Create array for number of live detectors as a function of revolution
    n_revs = len(rdx_det_time)
    live_det_array = np.zeros(n_revs, dtype=int)
    for rev in range(n_revs):
        if rdx_det_time[rev] == -1: live_det_array[rev] = -1
        elif rev <= 139: live_det_array[rev] = 19
        elif rev <= 213: live_det_array[rev] = 18
        elif rev <= 774: live_det_array[rev] = 17
        elif rev <= 929: live_det_array[rev] = 16
        else: live_det_array[rev] = 15
    
    # Create table HDU for live detector count
    col_live_det = fits.Column(name='LIVE_DET', format='J', array=live_det_array)
    live_det_hdu = fits.BinTableHDU.from_columns([col_live_det], name='LIVE_DET')
    
    # Create table HDU for single event detector live time
    se_det_time = spi_det_hi['det_time'][:,:19]
    se_cols = [fits.Column(name=f'DET{i}', format='D', array=se_det_time[:, i], unit='s') 
               for i in range(se_det_time.shape[1])]
    se_hdu = fits.BinTableHDU.from_columns(se_cols, name='SE_DET_TIME')
    
    # Create table HDU for double event detector live time
    de_det_time = spi_det_hi['det_time'][:,19:61]
    de_cols = [fits.Column(name=f'DET{i}', format='D', array=de_det_time[:, i], unit='s') 
               for i in range(de_det_time.shape[1])]
    de_hdu = fits.BinTableHDU.from_columns(de_cols, name='DE_DET_TIME')
    
    # Create table HDU for triple event detector live time
    te_det_time = spi_det_hi['det_time'][:,61:]
    te_cols = [fits.Column(name=f'DET{i}', format='D', array=te_det_time[:, i], unit='s') 
               for i in range(te_det_time.shape[1])]
    te_hdu = fits.BinTableHDU.from_columns(te_cols, name='TE_DET_TIME')
    
    # Create HDU list and write to FITS file
    hdul = fits.HDUList([primary_hdu, rdx_hdu, live_det_hdu, se_hdu, de_hdu, te_hdu])
    hdul.writeto(fits_file, overwrite=True)
    print(f"FITS file created: {fits_file}")

bkg_sav_path = {
        'SE':'/data1/ipp_afs_mirror/integral/data/databases/spi_line_db/data',
        'PSD':'/data1/ipp_afs_mirror/integral/data/databases/spi_line_db/data/psd/links',
        'HE':'/home/nbauer/cookbook/SPI_cookbook/examples/Crab/cookbook_dataset_02_0514-2000keV_PSD_new/spi/bg'
}
''''Dictionary with paths to the .sav folder'''

if __name__=='__main__':

    evt_type=input('event type?')
    spec_param_dir = bkg_sav_path[evt_type]

    bkg_db_dir = '/home/tbouchet/BKG_DB'
    # rev_start, rev_stop = 0, 3000
    rev_start, rev_stop = 40, 50
    
    revolutions=np.arange(rev_start, rev_stop, dtype='int64')

    # import .sav files with params
    spec_params_path_list = glob(f'{spec_param_dir}/com_spec_params_e*_revidx_*.sav')
    spec_params_path_list = order_path_list(spec_params_path_list)

    # compute background for all rev and write FITS files
    bkg_full = BkgList(spec_params_path_list, evt_type=evt_type)
    bkg_full.write_fits_files(output_dir=f'{bkg_db_dir}/{evt_type}', revolutions=revolutions, compress=True)

