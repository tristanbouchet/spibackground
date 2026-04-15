'''
Spectral data management.
Handles loading and retrieving raw spectra from archived data.
'''

from tqdm import tqdm
import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt


class Spectrum:
    '''Manages loading and access to raw spectral data'''
    
    def __init__(self, rawspec_sav_path):
        '''Initialize Spectrum manager with path to raw data directory
        '''
        self.rawspec_sav_path = rawspec_sav_path
        self.Ndet = 19
    
    def import_sav(self, pid):
        try:
            self.sav = readsav(f"{self.rawspec_sav_path}Private_low-rev{pid:04d}.sav")
        except FileNotFoundError:
            self.sav = None
        return self.sav
    
    def get_spectrum(self, pid, det):
        '''Load raw spectrum for a given pid and detector
        
        Also sets self.counts and self.count_err attributes from the loaded data.
        '''
        # Load spectra file for specific pid, if not done previously
        if self.sav is None:
            self.import_sav(pid)
            if self.sav is None:
                return None
            
        e_bounds = self.sav['spi_rev_spectra']['energy_boundaries'][0]
        self.channel = e_bounds['CHANNEL']
        self.e_mid = (e_bounds['e_min'] + e_bounds['e_max']) / 2
        spec = self.sav['spi_rev_spectra']['evts_det_spec'][0][det]
        
        # Extract and store counts and count_err as attributes
        self.counts = spec['counts']
        self.count_err = spec['stat_err']
        
        return self.counts, self.count_err, self.e_mid
    
    def get_pid_spectrum(self, pid):
        '''
        Load and sum spectrum for all detectors of a given pid
        Sums counts directly and stat_err quadratically across all detectors.
        '''
        # Load spectra file for pid
        sav=self.import_sav(pid)
        if sav is None: return None
        e_bounds = sav['spi_rev_spectra']['energy_boundaries'][0]
        self.e_mid = (e_bounds['e_min'] + e_bounds['e_max']) / 2
        
        # Initialize accumulators
        counts_sum = None
        count_err_sq_sum = None
        
        # Sum over all detectors
        for det in range(self.Ndet):
            spec = sav['spi_rev_spectra']['evts_det_spec'][0][det]
            counts = spec['counts']
            count_err = spec['stat_err']
            
            if counts_sum is None:
                counts_sum = counts.copy()
                count_err_sq_sum = count_err ** 2
            else:
                counts_sum += counts
                count_err_sq_sum += count_err ** 2
        
        # Quadratic sum for errors
        count_err_sum = np.sqrt(count_err_sq_sum)
        
        # Store as attributes
        self.counts = counts_sum
        self.count_err = count_err_sum
        
        return counts_sum, count_err_sum, self.e_mid, e_bounds
    
    def get_sumpid_spectrum(self, pid_list):
        '''Load and sum spectrum across multiple pids and all detectors'''
        counts_sum,count_err_sq_sum,e_mid,e_bounds = None,None,None,None
        self.valid_pid=[]
        # Sum over all pids
        for pid in tqdm(pid_list):
            get_pid_return = self.get_pid_spectrum(pid)
            if get_pid_return is None:
                # print(f'no spec for pid {pid}')
                continue
            self.valid_pid.append(pid)
            counts, count_err, e_mid_temp, e_bounds_temp = get_pid_return
            
            # first iteration
            if counts_sum is None:
                counts_sum = counts.copy()
                count_err_sq_sum = count_err ** 2
                e_mid = e_mid_temp
                e_bounds = e_bounds_temp
            else:
                counts_sum += counts
                count_err_sq_sum += count_err ** 2
        
        # Quadratic sum for errors
        count_err_sum = np.sqrt(count_err_sq_sum)
        
        # Store as attributes
        self.counts = counts_sum
        self.count_err = count_err_sum
        self.e_mid = e_mid
        print(f'finished summed spectrum {len(self.valid_pid)}/{len(pid_list)} revs')
        return counts_sum, count_err_sum, e_mid, e_bounds
    
    def plot(self, figsize=(12, 6), ax=None, label='Spectrum', emin=None, emax=None):
        '''Plot the current spectrum, optionally restricted to energy range.'''
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create energy mask if bounds specified
        if emin is not None or emax is not None:
            mask = (self.e_mid >= emin) & (self.e_mid <= emax)
            e_data = self.e_mid[mask]
            counts_data = self.counts[mask]
            counts_err_data = self.count_err[mask]
        else:
            e_data = self.e_mid
            counts_data = self.counts
            counts_err_data = self.count_err
        
        ax.errorbar(
            e_data, counts_data, yerr=counts_err_data, 
            fmt='o', label=label, capsize=3, markersize=4, alpha=0.7
        )
        
        ax.set_xlabel('Energy (keV)', fontsize=12)
        ax.set_ylabel('Counts/bin', fontsize=12)
        ax.set_title('Spectrum', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def __str__(self):
        from glob import glob
        all_sav=glob(f"/Users/tbastro/SPI_analysis/BACKGROUND/RAW_SPEC/Private_low-rev????.sav")
        return f"{len(all_sav)} rev available"
