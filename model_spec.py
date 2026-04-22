import numpy as np
from scipy.special import erfc, log_ndtr
from scipy.stats import exponnorm # convolved exp and gauss distribution

GAUSSCONST = np.sqrt(np.pi/2)
LOGGAUSSCONST = np.log(GAUSSCONST)
SQRT2PI = np.sqrt(2*np.pi)

#################### Math functions ####################

class EnergyConversion:
    '''Static methods for energy-index conversions'''
    @staticmethod
    def idx_to_energy_SE(x):
        return 18.25 + x * 0.5
    
    @staticmethod
    def idx_to_energy_HE(x):
        return 18.5 + x
    
    @staticmethod
    def energy_to_idx_HE(x):
        return x - 18.5

def log_erfc(x):
    '''
    DEFUNCT
    logarithm of the complementary error function
    relies on the logarithm of gaussian cumulative distribution (log_ndtr) from scipy
    this handles very large/small values well
    '''
    return np.log(2.0) + log_ndtr(-x * np.sqrt(2.0))

def distorted_gauss(E, A, E0, sig, tau):
    '''
    convolved line shape (gaussian with exponential)
    A is in counts/bin
    '''
    # for small tau, exp ~ dirac dist -> line ~ gauss
    if tau<=1e-3:
        return GAUSSCONST * A * np.exp(-(E-E0)**2/ (2*sig**2))
    # the exponnorm distribution from scipy has a right-tail, chosing (x=-E, mu=-E0) gives a left-tail
    else:
        return SQRT2PI * A * sig * exponnorm.pdf(-E, tau/sig, loc=-E0, scale=sig)

def power_law(E, Em, Cm, alpha):
    '''Cm in counts/bin'''
    return Cm * (E/Em)**alpha

def linear_fct(E, Em, Cm, alpha):
    '''Cm in counts/bin'''
    return Cm + (E - Em)*alpha

#################### Background models ####################

class BkgModel:
    '''background model base class'''
    def __init__(self, Em):
        self.Em = Em
        self.total_int=None
        self.param_dico={}
    
    def init_params(self, params):
        self.params = np.array(params)
        self.n_par = len(self.params)
        self.check_par()
        self.param_dico = self.reshape_params(self.params)
    
    def reshape_params(self, params):
        '''reshape parameter 1D array into dico with 2D array for lines'''
        raise NotImplementedError
    
    def check_par(self):
        '''check if number of param consistent with model
        condition is necessary, but not sufficient
        '''
        raise NotImplementedError
    
    def flatten_params(self):
        self.param_dico
    
    def estim_integral(self, emin, emax):
        '''
        estimate integral of model between energy bounds
        only using amplitude parameters (Cm for powerlaw, A*sig for convolved exp-gaussian)
        '''
        par_cont=self.param_dico['cont']
        cont_int= par_cont[0] * ((emax**(par_cont[1] + 1) - emin**(par_cont[1] + 1)) / ((par_cont[1]+1) * self.Em**par_cont[1]))
        line_int = SQRT2PI * np.sum([line_par[0]*line_par[2] for line_par in self.param_dico['lines']])
        self.total_int = cont_int + line_int
        return self.total_int
    
    def rescale_params(self, F0, emin, emax):
        '''
        re-scale the amplitude parameters using the ratio of total flux data/model
        '''
        if self.total_int is None:
            self.total_int = self.estim_integral(emin, emax)
        r_scale = F0 / self.total_int
        self.param_dico['cont'][0] = r_scale * self.param_dico['cont'][0]
        for i in range(self.n_lines):
            self.param_dico['lines'][i][0] = r_scale * self.param_dico['lines'][i][0]

        self.params = np.append(self.param_dico['cont'].flatten(), self.param_dico['lines'].flatten())
        return self.params
    
    def calc(self, E):
        '''can be over-written'''
        cont = power_law(E, self.Em, *self.param_dico['cont'])
        all_lines = np.array([distorted_gauss(E, *line_par) for line_par in self.param_dico['lines']])
        return {'cont':cont, 'lines':all_lines}
    
    def calc_tot(self, E):
        '''return sum of all the flux components (for fitting purpose)'''
        self.flux_dico = self.calc(E)
        return self.flux_dico['cont'] + np.sum(self.flux_dico['lines'], axis=0)

    def calc_fit(self, E, *params):
        '''return sum of all the flux components (for fitting purpose)'''
        # print(params)
        self.init_params(params)
        return self.calc_tot(E)

    def __call__(self, E):
        return self.calc(E)


class ClsPLModel(BkgModel):
    '''
    convolve line shape (gaussian with exponential) + power-law continuum
    use a different (A,E0,sig,tau) for each line
    result is in counts/bin
    '''
    def reshape_params(self, params):
        '''returns dico of params from 1D params list
        it does not modify self directly in order to be usable for other stuff
        '''
        dico={}
        dico['cont']= params[:2]
        dico['lines']= params[2:].reshape(-1, 4)
        return dico

    def check_par(self):
        if (self.n_par - 2)%4 != 0:
            raise IndexError
        self.n_lines = (self.n_par - 2)//4

class ClsLinModel(BkgModel):
    '''
    convolve line shape (gaussian with exponential) + power-law continuum
    use a different (A,E0,sig,tau) for each line
    result is in counts/bin
    '''
    def reshape_params(self, params):
        '''returns dico of params from 1D params list
        it does not modify self directly in order to be usable for other stuff
        '''
        dico={}
        dico['cont']= params[:2]
        dico['lines']= params[2:].reshape(-1, 4)
        return dico

    def check_par(self):
        if (self.n_par - 2)%4 != 0:
            raise IndexError
        self.n_lines = (self.n_par - 2)//4
    
    def calc(self, E):
        cont = linear_fct(E, self.Em, *self.param_dico['cont'])
        all_lines = np.array([distorted_gauss(E, *line_par) for line_par in self.param_dico['lines']])
        return {'cont':cont, 'lines':all_lines}
    
class Cls2PLModel(BkgModel):
    '''
    convolve line shape (gaussian with exponential) + power-law continuum
    use a different (A,E0,sig) for each line, but the same tau for all lines
    result is in counts/bin
    '''
    def reshape_params(self, params):
        dico={}
        dico['cont']= params[:2]
        # resize A,E0,sig
        lines_temp = params[2:-1].reshape(-1, 3)
        # duplicate and stack tau as an additional column
        dico['lines'] = np.hstack(
            (lines_temp,
             np.full((lines_temp.shape[0], 1), params[-1]))
             )
        return dico
    
    def check_par(self):
        if (self.n_par - 3)%3 != 0:
            raise IndexError
        self.n_lines = (self.n_par - 3)//3

BKG_MODELS = {
    'cls_plaw_function': ClsPLModel,
    'cls_plaw_function2': Cls2PLModel,
    'cls_lin_function': ClsLinModel,
}
"""Dictionary mapping function name in .sav files with the class name"""
