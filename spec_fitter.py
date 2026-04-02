'''
Background spectrum fitting.
Fits spectral data using BkgModel classes with scipy optimization.
'''

import numpy as np
from scipy.io import readsav
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
from model_spec import *
import emcee
import functools
from time import time
from background_db import timer
from spectrum import Spectrum

class SpectrumFitter:
    '''Fits background spectra for a given energy range'''
    def __init__(self, spectrum: Spectrum, init_param_dir, e_fit_min, e_fit_max, type='PSD', model_name=None):
        self.spectrum = spectrum
        self.init_param_dir = init_param_dir
        self.e_fit_min = e_fit_min
        self.e_fit_max = e_fit_max
        if type=='PSD' or type=='SE':
            self.bin_size=0.5
        elif type=='HE':
            self.bin_size=1.
        
        self.Ndet = 19
        
        # Load initial parameters from sav file
        sav_param_file = f"{init_param_dir}/init_spec_params_{int(e_fit_min)}_{int(e_fit_max)}_fix.sav"
        self.sav_param = readsav(sav_param_file)
        if model_name is None:
            self.model_name = self.sav_param['fit_fun'].decode("utf-8")
        else:
            self.model_name=model_name
        self.model_class = BKG_MODELS[self.model_name]
        self.model = self.model_class(self.sav_param['xc'])
        
        # Store fit results
        self.fit_results = {}  # {(pid, det): {'params': ..., 'perr': ..., 'success': ...}}
    

    def _get_energy_mask(self, e_mid):
        '''Get boolean mask for energy range of interest'''
        return (e_mid >= self.e_fit_min) & (e_mid <= self.e_fit_max)
    
    def log_likelihood(self, params, e_data, counts_data, counts_err):
        '''Compute chi-squared log-likelihood'''
        self.model.init_params(params=params)
        model_flux = self.model.calc_tot(e_data)
        
        # Check for bad values
        if np.any(~np.isfinite(model_flux)) or np.any(model_flux < 0):
            return -np.inf
        
        # Standard chi-squared log-likelihood
        chi2 = np.sum(((counts_data - model_flux) / counts_err) ** 2)
        return -0.5 * chi2
    
    def fit_spectrum_mcmc(self, pid, det, walker_dim_factor=4, nsteps=500, verbose=False):
        '''Fit spectrum using MCMC (emcee)'''
        
        # Load data
        _, _, e_mid = self.spectrum.get_spectrum(pid, det)
        mask = self._get_energy_mask(e_mid)
        
        e_data = e_mid[mask]
        counts_data = self.spectrum.counts[mask]
        counts_err = self.spectrum.count_err[mask]
        F0 = self.bin_size * np.sum(counts_data)
        
        # Get initial parameters from scipy fit first  
        self.model.init_params(params=self.sav_param['left_det'])
        init_params_raw = self.model.rescale_params(F0, self.e_fit_min, self.e_fit_max)
        if np.abs(counts_data[-1] - counts_data[0]) > 1e-3:
            alpha_estim = np.log(counts_data[-1]/counts_data[0]) / np.log(self.e_fit_max/self.e_fit_min)
            init_params_raw[1] = alpha_estim
        
        # Define log-probability (likelihood only, no priors for now)
        def log_prob(params):
            return self.log_likelihood(params, e_data, counts_data, counts_err)
        
        # Initialize walkers in a small ball around initial params
        ndim = len(init_params_raw)
        nwalkers= walker_dim_factor * ndim
        pos = init_params_raw[np.newaxis, :] + np.random.normal(0, .1 * np.abs(init_params_raw), (nwalkers, ndim))
        
        # Run MCMC
        try:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
            sampler.run_mcmc(pos, nsteps, progress=verbose)
            
            # Extract results from chain (discard first half as burn-in)
            samples = sampler.get_chain(discard=nsteps//2, flat=True)
            
            # Best-fit is median of posterior
            popt = np.median(samples, axis=0)
            perr = np.std(samples, axis=0)
            success = True
            
        except Exception as e:
            if verbose:
                print(f"MCMC failed for pid={pid}, det={det}: {e}")
            popt = init_params_raw
            perr = np.full_like(init_params_raw, np.nan)
            success = False
        
        # Store results (same format as scipy fit)
        self.fit_results[(pid, det)] = {
            'init_params': init_params_raw,
            'params': popt,
            'perr': perr,
            'success': success
        }
        
        if verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"[MCMC {status}] pid={pid}, det={det}")
        
        return popt, perr, success
    
    @timer
    def fit_spectrum(self, pid, det=None, verbose=False, method='scipy', maxfev=10000, init_params=None, calc_spec=True):
        '''Fit spectrum. Options:
        - pid (single), det (int): fit single detector
        - pid (single), det (None): fit sum of all detectors for one pid
        - pid (list), det (None): fit sum of all pids and detectors
        '''
        if method == 'mcmc':
            return self.fit_spectrum_mcmc(pid, det, verbose=verbose)
        elif method not in ['scipy', 'least_squares']:
            raise ValueError(f"Unknown method: {method}. Use 'scipy', 'least_squares', or 'mcmc'.")
        
        if init_params is None:
            init_params = self.sav_param['left_det']
        
        # Load data based on pid/det combination
        if isinstance(pid, list) or isinstance(pid, np.ndarray):
            # Sum over multiple pids and all detectors
            if calc_spec: _, _, e_mid, _ = self.spectrum.get_sumpid_spectrum(pid)
            result_key = 'sum_pids'
        elif det is None:
            # Sum over all detectors for one pid
            if calc_spec: _, _, e_mid, _ = self.spectrum.get_pid_spectrum(pid)
            result_key = pid
        else:
            # Single detector, single pid
            if calc_spec: _, _, e_mid = self.spectrum.get_spectrum(pid, det)
            result_key = (pid, det)
        
        e_mid=self.spectrum.e_mid
        mask = self._get_energy_mask(e_mid)
        e_data = e_mid[mask]
        counts_data = self.spectrum.counts[mask]
        counts_err = self.spectrum.count_err[mask]
        F0 = self.bin_size * np.sum(counts_data) 
        
        # dead detectors
        if F0==0.:
            dead_param = np.zeros_like(init_params)
            self.fit_results[result_key] = {
                'init_params': init_params, 'params': dead_param, 'perr': dead_param, 'success': 'Dead'
            }
            if verbose:
                print(f"[Dead Detectors] {result_key}")
            return dead_param, dead_param, 'Dead'

        # Set initial parameters
        self.model.init_params(params=init_params)
        init_params = self.model.rescale_params(F0, self.e_fit_min, self.e_fit_max)
        if np.abs(counts_data[-1] - counts_data[0]) > 1e-3:
            alpha_estim = np.log(counts_data[-1]/counts_data[0]) / np.log(self.e_fit_max/self.e_fit_min)
            init_params[1] = alpha_estim
        
        # Perform fit
        try:
            if method == 'least_squares':
                # Use scipy.optimize.least_squares
                def residual(params):
                    self.model.init_params(params=params)
                    model_flux = self.model.calc_tot(e_data)
                    return (counts_data - model_flux) / counts_err
                
                result = least_squares(residual, init_params, max_nfev=maxfev)
                popt = result.x
                # Estimate uncertainties from Jacobian
                if result.jac is not None:
                    cov = np.linalg.inv(result.jac.T @ result.jac)
                    perr = np.sqrt(np.diag(cov))
                else:
                    perr = np.full_like(popt, np.nan)
                success = result.success
            else:
                # Use scipy.optimize.curve_fit (default)
                # Set bounds for parameters
                # should be integrated to Model class !
                lower_bounds = np.full(len(init_params), 0.0)
                upper_bounds = np.full(len(init_params), np.inf)
                for i in range(len(init_params)):
                    # let power-law index free
                    if i == 1:
                        lower_bounds[i] = -np.inf
                    # bounds E0 to minimum energy
                    elif (i - 3) % 4 == 0 and i >= 3:
                        lower_bounds[i] = self.e_fit_min
                    # upper-bound to energy related parameters
                    if i >= 3:
                        if (i - 3) % 4 == 0: # E0
                            upper_bounds[i] = self.e_fit_max
                        # elif (i - 3) % 4 == 1: # sigma
                        #     upper_bounds[i] = self.e_fit_max - self.e_fit_min
                        # elif (i - 3) % 4 == 2: # tau
                        #     upper_bounds[i] = self.e_fit_max - self.e_fit_min
                # print('lower bounds:',lower_bounds)
                # print('upper bounds:',upper_bounds)
                popt, pcov = curve_fit(
                    self.model.calc_fit, e_data, counts_data, p0=init_params, 
                    sigma=counts_err, absolute_sigma=True, maxfev=maxfev,
                    bounds=(lower_bounds, upper_bounds)
                    )
                perr = np.sqrt(np.diag(pcov))
                success = True
        except (RuntimeError, np.linalg.LinAlgError) as e:
            if verbose:
                print(f"Fit failed for {result_key}: {e}")
            popt = init_params
            perr = np.full_like(init_params, np.nan)
            success = False
        
        # Store results
        self.last_result = {
            'init_params': init_params,
            'params': popt,
            'perr': perr,
            'success': success
        }
        self.fit_results[result_key] = self.last_result
        
        if verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"[{status}] {result_key}")
        
        return popt, perr, success
    
    def fit_all_detectors(self, pid, verbose=False, method='scipy'):
        '''Fit all detectors for one pid'''
        results = {}
        for det in range(self.Ndet):
            params, perr, success = self.fit_spectrum(pid, det, verbose=verbose, method=method)
            results[det] = {
                'params': params,
                'perr': perr,
                'success': success
            }
        return results
    
    def fit_all_pids(self, pid_list, verbose=False, method='scipy'):
        '''Fit all detectors for all pids'''
        results = {}
        total = len(pid_list) * self.Ndet
        count = 0
        
        for pid in pid_list:
            for det in range(self.Ndet):
                self.fit_spectrum(pid, det, verbose=verbose, method=method)
                count += 1
                if verbose and count % 50 == 0:
                    print(f"Progress: {count}/{total}")
        
        return self.fit_results
    
    ############### Plot stuff ###############
    
    def plot_fit(self, pid, det, figsize=(12, 6), ax=None, show_initial=False, plot_lines=False, show_res=False):
        '''Plot data and model for a fitted spectrum'''

        if isinstance(pid, list) or isinstance(pid, np.ndarray):
            fit_result = self.fit_results['sum_pids']
            title_label=f'Rev = {pid[0]} - {pid[-1]}'
        elif det is None:
            fit_result = self.fit_results[pid]
            title_label=f'Rev = {pid}, All det'
        else:
            fit_result = self.fit_results[(pid,det)]
            title_label=f'Rev={pid}, Det={det}'
            _, _, e_mid = self.spectrum.get_spectrum(pid, det)
        
        
        mask = self._get_energy_mask(self.spectrum.e_mid)
        e_data = self.spectrum.e_mid[mask]
        counts_data = self.spectrum.counts[mask]
        counts_err = self.spectrum.count_err[mask]
        # Calculate fitted model
        self.model.init_params(params=fit_result['params'])
        model_flux = self.model.calc_tot(e_data)
        
        # Create plot
        if ax is None:
            if show_res:
                fig, (ax, ax_res) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
            else:
                fig, ax = plt.subplots(figsize=figsize)
                ax_res = None
        else:
            fig = ax.get_figure()
            ax_res = None
        
        # Plot data
        ax.errorbar(
            e_data, counts_data, yerr=counts_err, fmt='o', label='Data',capsize=3,markersize=4,alpha=0.7
            )
        
        
        if plot_lines:
            for i in range(self.model.n_lines):
                ax.plot(e_data, self.model.flux_dico['lines'][i] + self.model.flux_dico['cont'],
                        label=f'{self.model.param_dico['lines'][i][1]:.1f} keV')

        if show_initial:
            self.model.init_params(params=fit_result['init_params'])
            initial_flux = self.model.calc_tot(e_data)
            ax.plot(e_data, initial_flux, color='lightgrey', label='Initial', linewidth=2, linestyle='--')
            # Reset to fitted params
            self.model.init_params(params=fit_result['params'])
        
        # Plot fitted model
        ax.plot(e_data, model_flux, 'r-', label='Fit', linewidth=2)
        
        # Labels and formatting
        ax.set_ylabel('Counts/bin', fontsize=12)
        ax.set_title(title_label, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot residuals if requested
        if show_res and ax_res is not None:
            residuals = (counts_data - model_flux) / counts_err
            ax_res.errorbar(e_data, residuals, yerr=np.ones_like(residuals), fmt='+', color='#1f77b4')
            ax_res.axhline(y=0, color='r', linestyle='-', alpha=.7)
            ax_res.set_xlabel('Energy (keV)', fontsize=12)
            ax_res.set_ylabel('Residuals (σ)', fontsize=12)
            ax_res.grid(True, alpha=0.3)
        else:
            ax.set_xlabel('Energy (keV)', fontsize=12)
        
        return fig, ax
    
    def plot_all_detectors(self, pid, figsize=(16, 12)):
        '''Plot fitted spectra for all detectors of one pid'''
        fig, axes = plt.subplots(4, 5, figsize=figsize)
        axes = axes.flatten()
        
        for det in range(self.Ndet):
            ax = axes[det]
            try:
                self.plot_fit(pid, det, ax=ax)
            except ValueError:
                # Fit hasn't been run yet, skip
                ax.text(0.5, 0.5, f'Det {det}\nNo fit', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide the extra subplot
        axes[-1].set_visible(False)
        
        # Remove x labels from non-bottom plots and y labels from non-leftmost plots
        for det in range(self.Ndet):
            if det < 15:  # All except bottom row
                axes[det].set_xlabel('')
            if det % 5 != 0:  # All except leftmost column
                axes[det].set_ylabel('')
        
        fig.tight_layout()
        return fig, axes
    
    ############### Print stuff ###############

    def __str__(self):
        s=f"Last fit: [{self.last_result['success']}]\n"
        val_dict=self.model.reshape_params(self.last_result['params'])
        err_dict=self.model.reshape_params(self.last_result['perr'])
        for key in val_dict.keys():
            val, err = val_dict[key], err_dict[key]
            s += f'--- {key} ---\n'
            if key=='cont':
                s += f" Cm= {val[0]:.1e} ± {err[0]:.1e}, alpha= {val[1]:.1f} ± {err[1]:.1f}\n"
            elif key=='lines':
                for i in range(self.model.n_lines):
                    s += f" E0= {val[i][1]:>6.1f} ± {err[i][1]:>6.1f}, A= {val[i][0]:.1e} ± {err[i][0]:.1e}, "
                    s += f"sig= {val[i][2]:.1e} ± {err[i][2]:.1e}, tau= {val[i][3]:.1e} ± {err[i][3]:.1e}\n"
        return s
        

    def get_fit_summary(self, pid_list=None, det_list=None):
        '''Return fit statistics'''
        if pid_list is None and det_list is None:
            results_to_check = self.fit_results.items()
        else:
            results_to_check = [
                (key, val) for key, val in self.fit_results.items()
                if (pid_list is None or key[0] in pid_list) and
                   (det_list is None or key[1] in det_list)
            ]
        
        success_count = sum(1 for _, r in results_to_check if r['success'])
        total_count = len(list(results_to_check))
        
        summary = {
            'total_fits': total_count,
            'successful_fits': success_count,
            'failed_fits': total_count - success_count,
            'success_rate': success_count / total_count if total_count > 0 else 0.0,
            'energy_range': (self.e_fit_min, self.e_fit_max),
            'model_name': self.model_name,
        }
        
        return summary
    
    def get_parameters(self, pid, det):
        '''Get fitted parameters and uncertainties'''
        if (pid, det) not in self.fit_results:
            raise ValueError(f"No fit result for pid={pid}, det={det}")
        
        result = self.fit_results[(pid, det)]
        return result['params'], result['perr']
