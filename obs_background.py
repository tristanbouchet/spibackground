'''
Create re-binned background spectra for each scw and detectors
The base model (per revolution and detector) are computed beforehand and stored in a data base folder
'''

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, LogFormatter
import numpy as np
# from tqdm import tqdm
# default libraries
import functools
from glob import glob
from time import time
from datetime import datetime

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

class LiveTimeRev:
    def __init__(self, livetime_path, evt_type):
        self.evt_type = evt_type
        self.livetime_path = livetime_path
        hdul_live = fits.open(livetime_path)
        self.det_live_rdx = hdul_live['RDX'].data['RDX']
        self.num_det_live = hdul_live['LIVE_DET'].data['LIVE_DET']
        self.det_time = hdul_live[f'{self.evt_type}_DET_TIME'].data
    
    def find_live_rev(self, rev: str):
        '''returns the array containing the live time of each detector for a rev'''
        rev_idx = self.det_live_rdx[int(rev)]
        if rev_idx==-1:
            print(f'rev {rev} not in index of {self.livetime_path}')
            return None
        # convert into numpy array
        return np.array(self.det_time[rev_idx][0])


class RevBkgDB:
    '''
    background components from 1 revolution in ct/kev, imported from the background data base
    possible evt_type: SE (single event), PSD, HE
    defined on 0.5 kev energy bins for SE
    Automatically discovers available background types from FITS extensions
    '''
    def __init__(self, rev, evt_type: str, bkg_db_dir):
        '''rev treated as string because of leading 0s'''
        self.rev = str(rev).zfill(4) # convert into 4 characters, whichever type rev is
        self.evt_type = evt_type
        self.path = f'{bkg_db_dir}/{evt_type}/bkg_rate_rev_{self.rev}_{evt_type}.fits.gz'
        self.hdul = fits.open(self.path)
        
        # build useful energy arrays
        E_table = self.hdul['ENERGY'].data
        self.E_grid = E_table['E']
        self.E_bds = np.column_stack([E_table['E_LO'], E_table['E_HI']])
        self.E_delta = np.diff(self.E_bds)
        
        # Automatically discover background types from FITS extensions (exclude PRIMARY and ENERGY)
        self.bkg_types = [hdu.name for hdu in self.hdul[1:] if hdu.name != 'ENERGY']
        
        # Load each background component and store in dictionary
        self.bkg_data = {}
        for bkg_type in self.bkg_types:
            bkg_table = self.hdul[bkg_type].data
            # convert record object to numpy array 
            bkg_array = np.column_stack([bkg_table[col] for col in bkg_table.names])
            self.bkg_data[bkg_type] = {
                'array': bkg_array,
                'spec': None,
                'rate': None
            }
    
    def counts_to_rate(self, livetime_rev: LiveTimeRev):
        '''
        convert to spectrum (ct/s/kev) by dividing by live time of each det
        then converts to rate (ct/s) with energy bin size
        '''
        live_time_array = livetime_rev.find_live_rev(self.rev)
        if live_time_array is None:
            print(f'no live time found for rev {self.rev}')
            return
        
        # Process all background types
        for bkg_type in self.bkg_types:
            self.bkg_data[bkg_type]['spec'] = self.bkg_data[bkg_type]['array'] / live_time_array # ct/s/kev
            self.bkg_data[bkg_type]['rate'] = self.bkg_data[bkg_type]['spec'] * self.E_delta # ct/s
    
    def make_rbn_mat(self, E_bds):
        '''makes a bool matrix = True if a grid energy bin is inside larger energy bands
        this assumes that data energy bounds are always larger than DB model bkg
        '''
        self.rbn_mat = np.array([(self.E_grid >= erbn[0]) & (self.E_grid <= erbn[1]) for erbn in E_bds]).T
        return self.rbn_mat
    
    def plot_rbn_mat(self):
        fig, ax=plt.subplots(1,1)
        ax.imshow(self.rbn_mat, aspect='auto', origin='lower', interpolation='none',
                  extent=(0, self.rbn_mat.shape[1], 0, self.rbn_mat.shape[0])) # make pixel start at integer coordinate
        ax.set_xlabel('data channel'); ax.set_ylabel('model channel')

    def plot(self, det: int, bkg_type='CONTINUUM', plot_rate=True):
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        if plot_rate and self.bkg_data[bkg_type]['rate'] is not None:
            ax.plot(self.E_grid, self.bkg_data[bkg_type]['rate'][:,det], 'k--', label='Continuum')
            ax.set_ylabel('Rate (ct/s/keV)')
        else:
            bkg_table = self.hdul[bkg_type].data
            ax.plot(self.E_grid, bkg_table[f'DET{det}'],'k--')
            ax.set_ylabel('Counts/keV)')
        ax.set_xlabel('E (keV)')
        ax.loglog()
        ax.set_title(f'{bkg_type} spectrum (rev {self.rev} det {det})')
        ax.legend()
        return ax

class ObsBkg:
    '''
    list of all scw contained in an observation
    contains method to build the final output backgrounds used by spimodfit
    '''
    def __init__(self, main_dir, evt_type, tracer='GeSatTot'):
        self.main_dir = main_dir
        self.evt_type = evt_type
        self.load_scw(tracer)
        self.load_energies()
        self.load_dead_time()
    
    ##### File loading #####

    def load_scw(self, tracer):
        print('loading scw info')
        hdul_scw = fits.open(f'{self.main_dir}/scw.fits.gz')
        scw_data = hdul_scw[1].data
        self.scw_list = scw_data['ScwID']
        self.rev_list = scw_data['Revolution']
        self.rev_unique = np.unique(self.rev_list)
        # Map revolution numbers to indices in rebin arrays
        self.rev_to_idx = {rev: i for i, rev in enumerate(self.rev_unique)}
        self.rev_indices = np.array([self.rev_to_idx[rev] for rev in self.rev_list]) # size=Nscw
        self.tracer = scw_data[tracer]

    def load_energies(self):
        print('load energy bounds')
        hdul_ebds = fits.open(f'{self.main_dir}/spi/energy_boundaries.fits.gz')
        ebds_data = hdul_ebds[1].data
        self.chan = ebds_data['CHANNEL']
        self.ebin_num = len(self.chan)
        E_min, E_max = ebds_data['E_MIN'], ebds_data['E_MAX']
        self.E_bds = np.column_stack([E_min, E_max])
        self.E_center = (E_min + E_max) / 2
        self.E_width = E_max - E_min

    def load_dead_time(self):
        print('loading observation live times')
        hdul_dead = fits.open(f'{self.main_dir}/spi/dead_time.fits.gz')
        self.det_num = hdul_dead[1].header['DET_NUM']
        self.pt_num = hdul_dead[1].header['PT_NUM']
        self.livetime = hdul_dead[1].data['LIVETIME'] # in s, size=Ndet*Npointings
        # self.scw_list = [ScwBkg(scw_name) for scw_name in scw_file_list]

    ##### Init obs constants (independent of scw) #####

    def init_rev_bkg_list(self, livetime_rev: LiveTimeRev, bkg_db_dir='BKG_DB'):
        print('Initialize data base meta')
        hdul_meta = fits.open(f'{bkg_db_dir}/{self.evt_type}/info_rev_bkg_{self.evt_type}.fits.gz')
        self.valid_rev_list = hdul_meta['VALID_REV'].data['VALID']
        # self.E_bds = hdul_meta['ENERGY'].data['E']
        print('Initialize revolution backgrounds from data base')
        self.bkg_rev_list = []
        for rev in self.rev_unique:
            # TO DO: take latest previous rev when orbit index is -1 
            rev_bkg = RevBkgDB(rev, self.evt_type, bkg_db_dir)
            rev_bkg.counts_to_rate(livetime_rev)
            rev_bkg.make_rbn_mat(self.E_bds)
            self.bkg_rev_list.append(rev_bkg)
    
    def normalize_tracer(self, livetime_rev: LiveTimeRev):
        '''Normalize tracer by number of live detectors, then by average tracer per revolution'''
        # Normalize by number of live detectors
        live_rev_idx = livetime_rev.det_live_rdx[self.rev_list]
        n_det_live = livetime_rev.num_det_live[live_rev_idx]
        tracer_avg = self.tracer / n_det_live
        
        # Compute average tracer per revolution
        tracer_avg_per_rev = np.zeros(len(self.rev_unique))
        for i, rev in enumerate(self.rev_unique):
            mask = self.rev_list == rev
            tracer_avg_per_rev[i] = np.mean(tracer_avg[mask])
        
        # Map each scw to its revolution's average and normalize
        rev_indices_for_avg = np.array([self.rev_to_idx[rev] for rev in self.rev_list])
        tracer_avg_per_rev_per_scw = tracer_avg_per_rev[rev_indices_for_avg]
        self.tracer_norm = tracer_avg / tracer_avg_per_rev_per_scw
    
    def calc_bkg(self, bkg_types=None):
        '''
        Calculate background spectrum for all scw, detectors, and background types.
        Formula: spectrum [ct] = (rate[ct/s] @ B) * tracer_norm * livetime[s]
        where rate = spec[ct/s/keV] * E_delta[keV] is the rev background
        tracer_norm depends on scw, but not det
        livetime depends on scw and det

        Returns:
        --------
        bkg_results : dict
            Dictionary with background types as keys and spectra as values.
            Each spectrum has shape (Ndet*Nscw, Nchan), indexed as [det + Ndet*scw, chan]
        '''
        # Use all available background types if not specified
        if bkg_types is None:
            bkg_types = self.bkg_rev_list[0].bkg_types
        
        # Get rebin spectra for all revolutions and all background types
        rebin_bkg_all_rev = {}
        for bkg_type in bkg_types:
            rebin_bkg_all_rev[bkg_type] = np.array([
                bkg_rev.bkg_data[bkg_type]['rate'].T @ bkg_rev.rbn_mat 
                for bkg_rev in self.bkg_rev_list
            ])
        
        # Reshape livetime from (Nscw*Ndet,) to (Nscw, Ndet)
        livetime_reshaped = self.livetime.reshape(self.pt_num, self.det_num)
        
        # Process each background type
        bkg_output_dico = {}
        for bkg_type in bkg_types:
            # Index rebin arrays by rev for each scw: (Nscw, Ndet, Nchan)
            bkg_by_rev = rebin_bkg_all_rev[bkg_type][self.rev_indices]
            
            # Apply tracer and livetime scaling with broadcasting
            bkg = bkg_by_rev * self.tracer_norm[:, np.newaxis, np.newaxis] * livetime_reshaped[:, :, np.newaxis]
            
            # Reshape to 2D array: (Ndet*Nscw, Nchan) with indexing [det + Ndet*scw, chan]
            nchan = bkg.shape[2]
            bkg_output = bkg.transpose(1, 0, 2).reshape(-1, nchan)
            
            # Create 3D array with counts and Poisson errors: (Ndet*Nscw, Nchan, 2)
            bkg_with_err = np.zeros((bkg_output.shape[0], bkg_output.shape[1], 2))
            bkg_with_err[:, :, 0] = bkg_output  # counts
            bkg_with_err[:, :, 1] = np.sqrt(bkg_output)  # Poisson error
            bkg_output_dico[bkg_type] = bkg_with_err
        
        self.bkg_output_dico = bkg_output_dico
        return bkg_output_dico
    
    def plot_bkg(self, scw_idx, det, bkg_types=None, xaxis='energy', type_spec='counts'):
        '''Plot background spectrum for a scw and detector with energy error bars.
        
        Parameters:
        -----------
        bkg_types : str, list, or None
            If None, plots sum of all background types
            If str or list, plots specified background type(s)
        type_spec : 'counts' or 'per_kev'
        '''
        if not hasattr(self, 'bkg_output_dico'):
            print("Run calc_bkg() first")
            return
        
        idx = det + self.det_num * scw_idx
        
        # Determine which background types to plot
        if bkg_types is None:
            # Sum all background types
            bkg_counts = sum(self.bkg_output_dico[bt][idx, :, 0] for bt in self.bkg_output_dico.keys())
            bkg_errors = np.sqrt(sum(self.bkg_output_dico[bt][idx, :, 1]**2 for bt in self.bkg_output_dico.keys()))
            label = 'Total Background'
        else:
            # Plot specific background type(s)
            if isinstance(bkg_types, str):
                bkg_types = [bkg_types]
            bkg_counts = sum(self.bkg_output_dico[bt][idx, :, 0] for bt in bkg_types)
            bkg_errors = np.sqrt(sum(self.bkg_output_dico[bt][idx, :, 1]**2 for bt in bkg_types))
            label = ' + '.join(bkg_types)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if type_spec == 'counts':
            ax.errorbar(self.E_center, bkg_counts, xerr=self.E_width/2, yerr=bkg_errors, fmt='k-', label=label)
            ax.set_ylabel('Counts')
        elif type_spec == 'per_kev':
            ax.errorbar(self.E_center, bkg_counts/self.E_width, xerr=self.E_width/2, yerr=bkg_errors/self.E_width, fmt='k-', label=label)
            ax.set_ylabel('Counts/keV')
        
        ax.set_xlabel('Energy (keV)')
        ax.set_title(f'Background (scw {scw_idx}, det {det})')
        ax.legend()
        return ax
    
    def plot_bkg_by_detector(self, scw_idx, E_min, E_max, bkg_types=None, normalize=False):
        '''Plot background spectrum summed over energy range as a function of detector.
        '''
        if not hasattr(self, 'bkg_output_dico'):
            print("Run calc_bkg() first")
            return
        
        # Find energy channels within the specified range
        energy_mask = (self.E_center >= E_min) & (self.E_center <= E_max)
        
        # Determine which background types to use
        if bkg_types is None:
            bkg_types = list(self.bkg_output_dico.keys())
            label = 'Total Background'
        else:
            if isinstance(bkg_types, str):
                bkg_types = [bkg_types]
            label = ' + '.join(bkg_types)
        
        # Sum over specified energy range for all detectors in this scw
        counts_by_det = np.zeros(self.det_num)
        errors_by_det = np.zeros(self.det_num)
        for det in range(self.det_num):
            idx = det + self.det_num * scw_idx
            bkg_counts = sum(self.bkg_output_dico[bt][idx, energy_mask, 0] for bt in bkg_types)
            bkg_errors_sq = sum(self.bkg_output_dico[bt][idx, energy_mask, 1]**2 for bt in bkg_types)
            counts_by_det[det] = bkg_counts.sum()
            errors_by_det[det] = np.sqrt(bkg_errors_sq.sum())
        
        # Normalize if requested
        if normalize:
            mean_counts = np.mean(counts_by_det)
            counts_normalized = counts_by_det / mean_counts
            errors_normalized = errors_by_det / mean_counts
            ylabel = 'Normalized Counts'
            title_suffix = ' (normalized)'
            plot_counts = counts_normalized
            plot_errors = errors_normalized
        else:
            ylabel = 'Counts'
            title_suffix = ''
            plot_counts = counts_by_det
            plot_errors = errors_by_det
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.errorbar(x=np.arange(self.det_num), y=plot_counts, yerr=plot_errors, fmt='ko', capsize=5)
        ax.set_xlabel('Detector Number')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{label} (scw {scw_idx}, {E_min:.1f}-{E_max:.1f} keV){title_suffix}')
        ax.set_xticks(np.arange(self.det_num))
        ax.grid(True, alpha=0.3, axis='y')
        if normalize:
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Mean')
            ax.legend()
        return ax
    
    def write_output_bkg(self, output_dir='./', compress=True):
        '''Write background spectra to FITS files.
        
        Creates one FITS file per background type with structure:
        - Extension 'SPI.-BMOD-DSP' contains COUNTS and STAT_ERR columns
        - Each row contains arrays spanning all energy channels for one detector/scw
        - Shape: Ndet*Nscw rows, each with Nchan-length arrays
        '''
        if not hasattr(self, 'bkg_output_dico'):
            print("Run calc_bkg() first")
            return
        
        file_ext = '.fits.gz' if compress else '.fits'
        bkg_name_map = {'CONTINUUM': 'conti', 'LINES': 'lines'}
        bkgname_map = {'CONTINUUM': 'BG-conti', 'LINES': 'BG-lines'}
        
        # Write individual background files
        for bkg_type, data in self.bkg_output_dico.items():
            short_name = bkg_name_map.get(bkg_type, bkg_type.lower())
            counts = data[:, :, 0].astype(np.float32)
            errors = data[:, :, 1].astype(np.float32)
            nchan = counts.shape[1]
            
            cols = fits.ColDefs([
                fits.Column(name='COUNTS', format=f'{nchan}E', array=counts),
                fits.Column(name='STAT_ERR', format=f'{nchan}E', array=errors)
            ])
            bmod_hdu = fits.BinTableHDU.from_columns(cols)
            bmod_hdu.header.update({'EXTNAME': 'SPI.-BMOD-DSP', 'DET_NUM': self.det_num,
                                    'PT_NUM': self.pt_num, 'EBIN_NUM': self.ebin_num, 'TYPE': bkg_type})
            
            primary = fits.PrimaryHDU()
            primary.header.update({'AUTHOR': 'tbouchet', 'DATE': datetime.now().strftime('%Y-%m-%d %H:%M')})
            hdul = fits.HDUList([primary, bmod_hdu])
            filename = f'{output_dir}/output_bgmodel-{short_name}{file_ext}'
            hdul.writeto(filename, overwrite=True)
            print(f"Written {filename}")
        
        # Write grouping index file
        gdata = {k: [] for k in ['MEMBER_XTENSION', 'MEMBER_NAME', 'MEMBER_VERSION', 
                                  'MEMBER_POSITION', 'MEMBER_LOCATION', 'MEMBER_URI_TYPE', 'BKGNAME']}
        for bkg_type in self.bkg_output_dico.keys():
            short = bkg_name_map.get(bkg_type, bkg_type.lower())
            for k, v in [('MEMBER_XTENSION', 'BINTABLE'), ('MEMBER_NAME', 'SPI.-BMOD-DSP'),
                         ('MEMBER_VERSION', 1), ('MEMBER_POSITION', 1),
                         ('MEMBER_LOCATION', f'output_bgmodel-{short}{file_ext}'), ('MEMBER_URI_TYPE', 'URL'),
                         ('BKGNAME', bkgname_map.get(bkg_type, f'BG-{bkg_type.lower()}'))]:
                gdata[k].append(v)
        
        cols_grouping = [fits.Column(n, fmt, array=gdata[n]) for n, fmt in zip(
            ['MEMBER_XTENSION', 'MEMBER_NAME', 'MEMBER_VERSION', 'MEMBER_POSITION','MEMBER_LOCATION', 'MEMBER_URI_TYPE', 'BKGNAME'],
            ['8A', '32A', '1J', '1J', '256A', '3A', '32A'])]
        grouping_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols_grouping))
        grouping_hdu.header['EXTNAME'] = 'GROUPING'
        
        primary = fits.PrimaryHDU()
        primary.header.update({'IDXMEMBR':'SPI.-BMOD-DSP', 'BASETYPE':'DAL_GROUP', 'TELESCOP':'INTEGRAL',
                               'ORIGIN':'jmu', 'INSTRUME':'SPI', 'ISDCLEVL':'BKG_I','CREATOR':'python',
                               'AUTHOR': 'tbouchet', 'DATE': datetime.now().strftime('%Y-%m-%d %H:%M')
                               })
        fits.HDUList([primary, grouping_hdu]).writeto(f'{output_dir}/output_bgmodel_conti_sep_idx.fits.gz', overwrite=True)
        print(f"Written {output_dir}/output_bgmodel_conti_sep_idx.fits.gz")
    