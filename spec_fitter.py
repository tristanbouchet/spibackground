'''
Background spectrum fitting.
Fits spectral data using BkgModel classes with scipy optimization.
'''

import numpy as np
from scipy.io import readsav, savemat
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
from model_spec import *
import emcee
import functools
from time import time
from tqdm import tqdm
from background_db import timer
from spectrum import Spectrum

class SpectrumFitter:
    '''Fits background spectra for a given energy range'''
    def __init__(self, spectrum: Spectrum, init_param_dir, e_fit_min, e_fit_max, type='PSD', model_name=None):
        self.spectrum = spectrum
        self.init_param_dir = init_param_dir
        self.e_fit_min = e_fit_min
        self.e_fit_max = e_fit_max
        self.type= type
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
    
    # @timer
    def fit_spectrum(self, pid, det=None, verbose=False, method='scipy', maxfev=10000, init_params=None, 
                     calc_spec=True, with_bounds=True):
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
            init_params = self.sav_param['left_det'].copy()
        
        # Load data based on pid/det combination
        if isinstance(pid, list) or isinstance(pid, np.ndarray):
            # Sum over multiple pids and all detectors
            if calc_spec:
                spec_res=self.spectrum.get_sumpid_spectrum(pid)
            result_key = 'sum_pids'
        elif det is None:
            # Sum over all detectors for one pid
            if calc_spec:
                spec_res=self.spectrum.get_pid_spectrum(pid)
            result_key = pid
        else:
            # Single detector, single pid
            if calc_spec:
                spec_res=self.spectrum.get_spectrum(pid, det)
            result_key = (pid, det)
        
        if spec_res is None:
            return None
        
        e_mid=self.spectrum.e_mid
        mask = self._get_energy_mask(e_mid)
        e_data = e_mid[mask]
        counts_data = self.spectrum.counts[mask]
        counts_err = self.spectrum.count_err[mask]
        # Integrate flux over data bins
        F0 = self.bin_size * np.sum(counts_data) 
        
        # dead detectors
        if F0==0.:
            dead_param = np.zeros_like(init_params)
            self.fit_results[result_key] = {
                'init_params': init_params, 'params': dead_param, 'perr': dead_param, 'success': True, 'redchi2':0.
            }
            if verbose:
                print(f"[Dead Detectors] {result_key}")
            return dead_param, dead_param, 'Dead'

        # Set initial parameters
        alpha_estim = 0.
        if np.abs(counts_data[-1] - counts_data[0]) > 1e-3:
            alpha_estim = np.log(counts_data[-1]/counts_data[0]) / np.log(self.e_fit_max/self.e_fit_min)
            
        init_params[1] = alpha_estim
        self.model.init_params(params=init_params)
        init_params = self.model.rescale_params(F0, self.e_fit_min, self.e_fit_max)
        ndata = len(e_mid[mask])
        npar = self.model.n_par
        dof = ndata-npar
        # Perform fit
        # Define parameter boundaries (for cls_plaw_function)
        bounds = (-np.inf, np.inf)
        if with_bounds:
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
            bounds = (lower_bounds, upper_bounds)
        try:
            if method == 'least_squares':
                # Use scipy.optimize.least_squares
                def residual(params):
                    self.model.init_params(params=params)
                    model_flux = self.model.calc_tot(e_data)
                    return (counts_data - model_flux) / np.abs(counts_err)
                
                result = least_squares(residual, x0=init_params, max_nfev=maxfev)
                popt = result.x
                # print(result)
                # Estimate uncertainties from Jacobian
                if result.jac is not None:
                    # cov = np.linalg.inv(result.jac.T @ result.jac)
                    # perr = np.sqrt(np.diag(cov))
                    perr=0.
                else:
                    perr = np.full_like(popt, np.nan)
                success = result.success
                chi2 = np.sum(result.fun**2)
            else:
                # Use scipy.optimize.curve_fit (default)
                result = curve_fit(
                    self.model.calc_fit, e_data, counts_data, p0=init_params, 
                    sigma=counts_err, absolute_sigma=True, maxfev=maxfev, bounds=bounds,
                    full_output=True
                    )
                popt, pcov, infodict, mesg, ier = result
                perr = np.sqrt(np.diag(pcov))
                success = True
                res_sq = ((counts_data - self.model.calc_fit(e_data, *popt)) / counts_err)**2
                chi2= np.sum(res_sq)

            self.last_result = result    
            redchi2 = chi2/dof

        except (RuntimeError) as e:
        # except (RuntimeError, np.linalg.LinAlgError) as e:
            if verbose:
                print(f"Fit failed for {result_key}: {e}")
            popt = init_params
            perr = np.full_like(init_params, np.nan)
            success = False
            redchi2=0.

        # Store results
        self.last_fit = {
            'init_params': init_params,
            'params': popt,
            'perr': perr,
            'success': success,
            'redchi2': redchi2,
        }
        self.fit_results[result_key] = self.last_fit
        
        if verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"[{status}] {result_key}")
        
        return popt, perr, success
    
    def save_last_fit(self):
        print('Saving last fit result to .npy file...')
        file_name = f"fit_par_{self.type}_{self.e_fit_min}_{self.e_fit_max}keV.npy"
        np.save(file_name, self.last_fit['params'])

    def fit_all_detectors(self, pid, verbose=False, method='scipy', maxfev=10000, init_params=None, 
                     calc_spec=True, with_bounds=True):
        '''Fit all detectors for one pid'''
        results = {}
        for det in range(self.Ndet):
            params, perr, success = self.fit_spectrum(pid, det, verbose, method, maxfev, init_params, 
                     calc_spec, with_bounds)
            results[det] = {'params': params, 'perr': perr, 'success': success}

        return results
    
    def fit_all_pids(self, pid_list, verbose=False, method='scipy', maxfev=10000, init_params=None, 
                     calc_spec=True, with_bounds=True, save_to_file=True):
        '''Fit all detectors for all pids'''
        
        valid_pid_list, ctime_list=[], []

        for pid in tqdm(pid_list):
            self.spectrum.import_sav(pid)
            
            if self.spectrum.sav is not None:
                # print(f'pid {pid} valid')
                valid_pid_list.append(pid)
                ctime_list.append(self.spectrum.sav['spi_rev_spectra']['tmean'][0])

                for det in range(self.Ndet):
                    # print(f'det {det}')
                    self.fit_spectrum(pid, det, verbose, method, maxfev, init_params, 
                        calc_spec, with_bounds)
            else:
                pass
                # print(f'pid {pid} unvalid')
        
        print(f"{len(valid_pid_list)}/{len(pid_list)} pid are valid")


        print(f"Filling final parameter table")
        
        # sav_keys = ['spec_params_det', 'orbits', 'x_idx_range', 'xc', 'fit_func', 'ctime']
        # sav_type = [np.ndarray, np.ndarray, np.ndarray, np.float64, bytes, np.ndarray]
        self.sav_dico={}

        # Create 4D array: [value/error, params, detector, pid]
        num_pids = len(valid_pid_list)
        # Get number of parameters from first fit result
        first_fit = self.fit_results[(valid_pid_list[0], 0)]
        num_params = len(first_fit['params'])
        spec_params_det = np.zeros((2, num_params, self.Ndet, num_pids), dtype=np.float64)
        success_table = np.zeros((self.Ndet, num_pids), dtype=bool)
        redchi2_table = np.zeros((self.Ndet, num_pids), dtype=np.float64)
        num_failed_fit = 0
        for pid_idx, pid in enumerate(valid_pid_list):
            for det in range(self.Ndet):
                if (pid, det) in self.fit_results:
                    result = self.fit_results[(pid, det)]
                    spec_params_det[0, :, det, pid_idx] = result['params']
                    spec_params_det[1, :, det, pid_idx] = result['perr']
                    success_table[det, pid_idx] = result['success']
                    redchi2_table[det, pid_idx] = result['redchi2']
                    if not result['success']: num_failed_fit+=1

        print(f'{num_failed_fit}/{num_pids*self.Ndet} fits failed')
        self.sav_dico['spec_params_det'] = np.asarray(spec_params_det, dtype=np.float64)
        e_mid=self.spectrum.e_mid
        mask = self._get_energy_mask(e_mid)
        self.sav_dico['orbits'] = np.asarray(valid_pid_list, dtype=np.int32)
        self.sav_dico['x_idx_range'] = np.asarray(self.spectrum.channel[mask], dtype=np.int32)
        self.sav_dico['xc'] = np.asarray(self.sav_param['xc'], np.float64)
        self.sav_dico['fit_func'] = self.sav_param['fit_fun'] # bytes
        self.sav_dico['ctime'] = np.asarray(ctime_list, dtype=np.float32)
        # extra tables
        self.sav_dico['success'] = success_table
        self.sav_dico['redchi2'] = redchi2_table

        if save_to_file:
            import pickle
            filename = f"com_spec_params_e{self.e_fit_min}_{self.e_fit_max}_revidx_{valid_pid_list[0]:04d}-{valid_pid_list[-1]:04d}.pkl"
            # savemat(filename, self.sav_dico, oned_as='row')
            with open(filename, 'wb') as f:
                pickle.dump(self.sav_dico, f)

            print(f"Saved to {filename}")
            print(f"  - x_idx_range: shape {self.sav_dico['x_idx_range'].shape}")
            print(f"  - orbits: shape {self.sav_dico['orbits'].shape}")
            print(f"  - ctime: shape {self.sav_dico['ctime'].shape}")
            print(f"  - spec_params_det: shape {self.sav_dico['spec_params_det'].shape}")
            print(f"  - fit_fun: {self.sav_dico['fit_func']}")
            print(f"  - xc {self.sav_dico['xc']}")


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
        s=f"Last fit: [{self.last_fit['success']}]\n"
        val_dict=self.model.reshape_params(self.last_fit['params'])
        err_dict=self.model.reshape_params(self.last_fit['perr'])
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


if __name__=='__main__':
        
    pid_list = np.arange(43, 2800, 1000)
    e_fit_min, e_fit_max = 453, 490

    spectrum = Spectrum(rawspec_sav_path='/data1/ipp_afs_mirror/integral/data/databases/spec_response/spi_rev_spectra/spi_rev_spectra-PSD/')
    fitter = SpectrumFitter(
        spectrum,
        init_param_dir='/data1/ipp_afs_mirror/integral/software/local/idl/tsmcmc/init_files',
        e_fit_min=e_fit_min, e_fit_max=e_fit_max,
    )
    # pid_list = [20,43] #[43, 1000, 2000]  # Example with multiple pids
    all_results = fitter.fit_all_pids(pid_list, verbose=False, method='scipy',maxfev=50000, init_params=None,
                                    with_bounds=False , save_to_file=True)