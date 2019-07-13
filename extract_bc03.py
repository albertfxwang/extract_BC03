import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import warnings

from dust_extinction import calzetti, cardelli
from add_emlines import add_emission_lines
from igm_attenuation import inoue_tau

rootdir = '/Users/albert/local/bc03/'
modelsdir = os.path.join(rootdir, 'models')
strack = 'Padova1994'

class TemplateSED_BC03(object):

    metallicity_key = {0.0001: 'm22', 0.0004: 'm32', 0.004: 'm42',
                       0.008: 'm52', 0.02: 'm62', 0.05: 'm72', 0.1: 'm82'}
    # NOTE: currently only work for Padova1994 stellar evolutionary tracks
    #       for the Padova2000 tracks, the metallicities are denoted by m122, m132, etc.
    #       for the Geneva1994 tracks, only the solar metallicities (i.e. m62) track models are provided

    sfh_key = {'ssp': 0, 'exp': 1, 'single': 2, 'constant': 3}
    library_atlas_key = {'stelib': 'Stelib_Atlas',
                         'BaSeL': 'BaSeL3.1_Atlas',
                         'xmiless': 'Miles_Atlas'}
    imf_dir_key = {'salp': 'salpeter', 'chab': 'chabrier', 'kroup': 'kroupa'}

    def __init__(self,
                 age, sfh, metallicity=None, input_ised=None, input_sfh=None,
                 tau=None, Av=None, emlines=False, dust='calzetti',
                 redshift=None, igm=True,
                 sfr=1, gasrecycle=False, epsilon=0.001, tcutsfr=20,
                 units='flambda', W1=1, W2=1e7,
                 lya_esc=0.2, lyc_esc=0,
                 imf='chab', res='hr', uid=None,
                 rootdir=rootdir, modelsdir=modelsdir, library_version=2003, library='stelib',
                 workdir=None, cleanup=True, del_input=False, verbose=True):
        """
        metallicity:     0.0001(m22), 0.0004(m32), 0.004(m42), 0.008(m52), 0.02(m62), 0.05(m72) [BC2003 option]
        age:             0 < age < 13.5 Gyr [BC2003 option]
        sfh:             Star formation history [BC2003 option]
                            - 'constant':   constant SFR (requires SFR, TCUTSFR)
                            - 'exp':        exponentially declining  (requires TAU, TCUTSFR, GASRECYCLE[, EPSILON])
                            - 'ssp':        single stellar pop
                            - 'single':     single burst  (requires TAU - length of burst)
                            - 'custom':     custom SFH file (two column file -- col#1: age [yr]; col#2: SFR [Mo/yr])    NOTE: not yet implemented
        tau:             e-folding timescale for exponentially declining SFH [BC2003 option]
        Av:              dust content (A_v)
        emlines:         adds emission lines
        dust:            dust extinction law
                            - None:       No dust extinction to be applied
                            - 'calzetti':   Apply Calzetti (2000) dust law
                            - 'cardelli':   Apply Cardelli (1989) dust law
        z:               Redshift for the SED
        igm:             Apply IGM attentuation?
        SFR:             Star formation rate [BC2003 option]
        gasrecycle:      Recycle the gas [BC2003 option]
        epsilon:         Fraction of recycled gas [BC2003 option]
        tcutsfr:         Time at which SFR drops to 0
        units:           Units to return the sed in [BC2003 option]
                            - 'lnu':       ergs/s/Hz
                            - 'llambda':   ergs/s/Angs
        W1, W2:          Limits of the wavelength range to compute the SED in [BC2003 option]
        imf:             Initial Mass Function ('salp', 'chab' or 'kroup' if using 2012 version) [BC2003 option]
        res:             Resolution of the SED [BC2003 option]
                            - 'hr':         High resolution
                            - 'lr':         Low resolution
        uid:             Unique ID for the SED
        rootdir:         Root directory for the GALAXEv installation
        modelsdir:       Root directory for model atlases
        workdir:         Working directory to store temporary files
        library_version: Specify which version of BC03 -- 2003, 2012, 2016
        library:         Specify specific library (only valid for 2012(16) version) -- 'stelib','BaSeL'(,'xmiless')
                            - None: for 2003 version
        input_ised:      Option to directly specify what input ISED file to use
        cleanup:         Cleanup the temporary files?
        del_input:       Delete input files for the BC03 executables
        verbose:         Print messages to terminal?
        """

        self.sfh_key = TemplateSED_BC03.sfh_key
        # {'ssp':0,'exp':1,'single':2,'constant':3}
        self.library_atlas_key = TemplateSED_BC03.library_atlas_key
        # {'stelib' : 'Stelib_Atlas', 'BaSeL' : 'BaSeL3.1_Atlas', 'xmiless' : 'Miles_Atlas' }
        self.imf_dir_key = TemplateSED_BC03.imf_dir_key
        # {'salp':'salpeter','chab':'chabrier','kroup':'kroupa'}
        self.metallicity_key = TemplateSED_BC03.metallicity_key
        self.inv_metallicity_key = dict([[v, k] for k, v in self.metallicity_key.items()])

        self.res = res
        self.age = age
        self.sfh = sfh
        self.sfr = sfr
        self.tau = tau
        self.gasrecycle = gasrecycle
        self.epsilon = epsilon
        self.tcutsfr = tcutsfr
        self.Av = Av
        self.emlines = emlines
        self.dust = dust
        self.redshift = redshift
        self.igm = igm
        self.units = units
        self.input_sfh   = input_sfh
        self.W1 = W1
        self.W2 = W2
        self.lya_esc     = lya_esc
        self.lyc_esc     = lyc_esc
        self.rootdir = rootdir
        self.modelsdir = modelsdir
        self.library = library
        self.library_version = library_version
        self.del_input = del_input
        self.sed = None

        if input_ised:
            if ".ised" in input_ised:
                input_ised = input_ised[:-5]
            warnings.warn(
                'Ignoring IMF and Metallicity args and using provided input ISED file: %s' % input_ised)
            self.input_ised = input_ised
            self.imf = input_ised.split('_')[-2]
            self.metallicity = self.inv_metallicity_key[input_ised.split('_')[-3]]
        else:
            self.metallicity = metallicity
            self.imf = imf
            #------------ set up paths and file names
            if self.library_version == 2003:
                assert self.imf != 'kroupa', ' ERR: 2003 models do not include the Kroupa IMF!'
                self.model_dir = os.path.join(modelsdir, str(self.library_version), strack, self.imf_dir_key[self.imf])
                self.input_ised = 'bc2003_' + self.res + '_' + self.metallicity_key[self.metallicity] + '_' + self.imf + '_ssp'
            else:
                assert self.library is not None, ' ERR: not specifying the stellar evolutionary tracks!'
                if self.library_version == 2012:
                    subdir = strack
                else:
                    subdir = self.library_atlas_key[self.library]
                self.model_dir = os.path.join(modelsdir, str(self.library_version), subdir, self.imf_dir_key[self.imf])
                self.input_ised = 'bc2003_' + self.res + '_' + self.library + '_' + self.metallicity_key[self.metallicity] + '_' + self.imf + '_ssp'

        self.Q = {}
        self.M_unnorm = {}

        self.read_age_input()
        self.check_input()

        self.workdir = workdir + '/' if workdir else os.getcwd() + '/'
        self.uid = uid
        self.ssp_output = self.uid + '_ssp'
        self.csp_output = self.uid + '_csp'
        self.cleanup = cleanup
        self.verbose = verbose

        self.define_env()
        self.mk_csp_input()
        self.mk_gpl_input()

    def read_age_input(self):
        """
        Verify that the requested number of SED ages is compatible with the range
        that is supported by the specified library version. If sufficient SEDs can be
        generated then store the specified ages (units Gyr) internally.
        """
        if self.library_version == 2003:
            self.age_limit = 24
        elif self.library_version == 2012 or self.library_version == 2016:
            self.age_limit = 100

        if not (isinstance(self.age, np.ndarray) or isinstance(self.age, list)):
            self.ages = str(self.age)
            self.age = np.array([self.age, ])
        elif len(self.age) <= self.age_limit:
            self.age = np.asarray(self.age)
            self.ages = ','.join(np.round(self.age, 6).astype(str))
        else:
            raise Exception('Cannot provide more than %i ages!' % self.age_limit)

    def check_input(self):
        """
        Validate constructor arguments.
        """
        if any(self.age > 13.8):
            raise Exception(
                "SED age (%s) provided is older than the Universe (13.8 Gyr)!" % str(self.age))
        if self.sfh not in self.sfh_key.keys():
            raise Exception("Incorrect SFH provided: " + str(self.sfh) + "\n"
                            "Please choose from:" + str(self.sfh_key.keys()))
        if self.sfh=='custom':
            if not self.input_sfh:
                raise Exception("No input SFH file provided.")
            else:
                if not os.path.isfile(self.input_sfh):
                    raise Exception("Specified input SFH not found at %s." % self.input_sfh)
        if self.metallicity not in self.metallicity_key.keys():
            raise Exception("Incorrect metallicity provided: " + str(self.metallicity) + "\n"
                            "Please choose from:" + str(self.metallicity_key.keys()))
        if self.imf not in self.imf_dir_key.keys():
            raise Exception("Incorrect IMF provided: " + str(self.imf) + "\n"
                            "Please choose from:" + str(self.imf_dir_key.keys()))
        if self.res not in ['hr', 'lr']:
            raise Exception("Incorrect resolution provided: " + str(self.res) + "\n"
                            "Please choose from: 'hr','lr'")
        if self.units not in ['flambda', 'fnu']:
            raise Exception("Incorrect flux units provided: " + str(self.units) + "\n"
                            "Please choose from: 'flambda','fnu'")
        if self.dust not in ['None', 'calzetti', 'cardelli']:
            raise Exception("Incorrect dust law provided: " + str(self.dust) + "\n"
                            "Please choose from: 'none','calzetti','cardelli'")
        if self.redshift is not None and self.redshift < 0:
            raise Exception("Incorrect redshift provided: " + str(self.redshift) + "\n"
                            "Please provide a positive value.")
        if self.redshift is None and self.igm:
            warnings.warn("No redshift provided, and thus IGM attentuation cannot be applied.")
        if self.library_version not in [2003, 2012, 2016]:
            raise Exception("Invalid library_version: " + str(self.library_version) + "\n"
                            "Please choose from: 2003, 2012, 2016")
        if self.library not in ['stelib', 'BaSeL', 'xmiless', 'None']:
            raise Exception("Incorrect library choice: " + str(self.library) + "\n"
                            "Please choose from: 'stelib','BaSeL', 'xmiless'")

    def define_env(self):

        self.env_string = "export FILTERS=" + self.rootdir + "src/FILTERBIN.RES;" \
                          "export A0VSED=" + self.rootdir + "src/A0V_KURUCZ_92.SED;" \
                          "export RF_COLORS_ARRAYS=" + self.rootdir + "src/RF_COLORS.filters;"
        if self.library_version == 2012 or self.library_version == 2016:
            self.env_string += "export SUNSED=" + self.rootdir + "src/SUN_KURUCZ_92.SED;"

    def del_file(self, f):
        """
        Helper to safely delete files during cleanup.
        """
        if os.path.isfile(f):
            os.remove(f)

    def generate_sed(self):
        """
        Main driver function that executes the required stages for SED generation and
        initiates cleanup of auxilliary files if required.
        """
        # copy SED library file from atlas and convert from ASCII to binary if required.
        self.do_bin_ised()

        self.do_csp()
        if self.cleanup:
            self.csp_cleanup()

        self.do_gpl()
        if self.cleanup:
            self.gpl_cleanup()

        self.read_gpl()
        if self.cleanup:
            self.post_gpl_cleanup()

        if self.emlines:
            self.add_emlines()

        if self.dust:
            self.add_dust()

        if self.redshift:
            self.redshift_evo()

    def do_bin_ised(self):
        """
        Copy required SED model files from the model atlas directory and, if necessary
        (the SEDs are provided as ASCII files) execute the bin_ised utility to convert them
        to a binary format.
        """
        if os.path.isfile(os.path.join(self.model_dir, self.input_ised + '.ised')):
            shutil.copyfile(os.path.join(self.model_dir, self.input_ised + '.ised'),
                            os.path.join(self.workdir, self.ssp_output + '.ised'))
        elif os.path.isfile(os.path.join(self.model_dir, self.input_ised + '.ised_ASCII')):
            shutil.copyfile(os.path.join(self.model_dir, self.input_ised + '.ised_ASCII'),
                            os.path.join(self.workdir, self.ssp_output + '.ised_ASCII'))
            if self.verbose:
                subprocess.call(self.rootdir + 'src/bin_ised ' + self.ssp_output + '.ised_ASCII',
                                cwd=self.workdir, shell=True)
            else:
                subprocess.call(self.rootdir + 'src/bin_ised ' + self.ssp_output + '.ised_ASCII',
                                cwd=self.workdir, shell=True, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))
            if self.del_input: self.del_file(self.workdir + self.ssp_output + '.ised_ASCII')
        else:
            raise Exception('Template %s not found in %s.' % (self.input_ised, self.model_dir))

    def mk_csp_input(self):
        """
        Generate input command line tokens for the csp (csp_galexev) utility.
        """
        self.csp_input = {}
        self.csp_input['CSPINPUT'] = self.ssp_output
        self.csp_input['DUST'] = 'N'
        self.csp_input['REDSHIFT'] = str(0)
        self.csp_input['SFHCODE'] = str(self.sfh_key[self.sfh])
        self.csp_input['SFR'] = str(self.sfr)
        self.csp_input['TAU'] = str(self.tau)
        self.csp_input['GASRECYCLE'] = 'Y' if self.gasrecycle else 'N'
        self.csp_input['EPSILON'] = str(self.epsilon)
        self.csp_input['TCUTSFR'] = str(self.tcutsfr)
        self.csp_input['CSPOUTPUT'] = self.csp_output
        self.csp_input['INPUT_SFH'] = self.input_sfh

    def do_csp(self):
        """
        Synthesize and invoke the command line for the csp (csp_galexev) utility.
        This utility actually computes the spectral evolution of composite stellar
        populations.
        """
        csp_input_string = self.csp_input['CSPINPUT'] + '\n'
        csp_input_string += self.csp_input['DUST'] + '\n'
        if self.library_version == 2012 or self.library_version == 2016:
            csp_input_string += self.csp_input['REDSHIFT'] + '\n'
        csp_input_string += self.csp_input['SFHCODE'] + '\n'

        if self.sfh == 'ssp':
            pass
        elif self.sfh == 'exp':
            csp_input_string += self.csp_input['TAU'] + '\n'
            csp_input_string += self.csp_input['GASRECYCLE'] + '\n'
            if self.csp_input['GASRECYCLE'] == 'Y':
                csp_input_string += self.csp_input['EPSILON'] + '\n'
            csp_input_string += self.csp_input['TCUTSFR'] + '\n'
        elif self.sfh == 'single':
            csp_input_string += self.csp_input['TAU'] + '\n'
        elif self.sfh == 'constant':
            csp_input_string += self.csp_input['SFR'] + '\n'
            csp_input_string += self.csp_input['TCUTSFR'] + '\n'
        elif self.sfh == 'custom':
            csp_input_string += self.csp_input['INPUT_SFH'] + '\n'

        csp_input_string += self.csp_input['CSPOUTPUT'] + '\n'

        if self.verbose:
            print('Input parameters for csp_galaxev:')
            for key, value in self.csp_input.items():
                print('{} => {}'.format(key, value))
            print('\n')

        with open(self.workdir + self.uid + '_csp.in', 'w') as f:
            f.write(csp_input_string)
        if self.verbose:
            subprocess.call(self.env_string + self.rootdir + 'src/csp_galaxev < ' + self.uid + '_csp.in',
                            cwd=self.workdir, shell=True)
        else:
            subprocess.call(self.env_string + self.rootdir + 'src/csp_galaxev < ' + self.uid + '_csp.in',
                            cwd=self.workdir, shell=True, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))

        if self.del_input: self.del_file(self.workdir + self.uid + '_csp.in')


    def csp_cleanup(self):
        """
        Remove any non-essential files that were generated by the csp (csp_galexev)
        utility.
        """
        self.del_file(self.workdir + self.csp_output + '.1ABmag')
        self.del_file(self.workdir + self.csp_output + '.1color')
        self.del_file(self.workdir + self.csp_output + '.2color')
        self.del_file(self.workdir + self.csp_output + '.5color')
        self.del_file(self.workdir + self.csp_output + '.6lsindx_ffn')
        self.del_file(self.workdir + self.csp_output + '.6lsindx_sed')
        self.del_file(self.workdir + self.csp_output + '.6lsindx_sed_lick_system')
        self.del_file(self.workdir + self.csp_output + '.7lsindx_ffn')
        self.del_file(self.workdir + self.csp_output + '.7lsindx_sed')
        self.del_file(self.workdir + self.csp_output + '.7lsindx_sed_lick_system')
        self.del_file(self.workdir + self.csp_output + '.8lsindx_sed_fluxes')
        self.del_file(self.workdir + 'bc03.rm')

        if self.library_version == 2012 or self.library_version == 2016:
            self.del_file(self.workdir + self.csp_output + '.9color')
            self.del_file(self.workdir + self.csp_output + '.acs_wfc_color')
            self.del_file(self.workdir + self.csp_output + '.legus_uvis1_color')
            self.del_file(self.workdir + self.csp_output + '.wfc3_color')
            self.del_file(self.workdir + self.csp_output + '.wfc3_uvis1_color')
            self.del_file(self.workdir + self.csp_output + '.wfpc2_johnson_color')
            self.del_file(self.workdir + self.csp_output + '.w_age_rf')
            self.del_file(self.workdir + 'fort.24')

    def mk_gpl_input(self):
        """
        Generate input command line tokens for the gpl (galaxevpl) utility.
        """
        self.gpl_input = {}
        self.gpl_input['GPLINPUT'] = self.csp_output
        if self.units == 'flambda':
            self.gpl_input['W1W2W0F0Z'] = str(self.W1) + ',' + str(self.W2)
        elif self.units == 'fnu':
            self.gpl_input['W1W2W0F0Z'] = str(-self.W1) + ',' + str(self.W2)
        self.gpl_input['AGES'] = self.ages
        self.gpl_input['GPLOUTPUT'] = self.csp_output + '.spec'

    def do_gpl(self):
        """
        Synthesize and invoke the command line for the gpl (galaxevpl) utility.
        """
        gpl_input_string = self.gpl_input['GPLINPUT'] + '\n'

        if self.library_version == 2003:
            gpl_input_string += self.gpl_input['W1W2W0F0Z'] + '\n'
            gpl_input_string += self.gpl_input['AGES'] + '\n'
        elif self.library_version == 2012 or self.library_version == 2016:
            gpl_input_string += self.gpl_input['AGES'] + '\n'
            gpl_input_string += self.gpl_input['W1W2W0F0Z'] + '\n'

        gpl_input_string += self.gpl_input['GPLOUTPUT'] + '\n'

        if self.verbose:
            print('Input string for galaxevpl:')
            for key, value in self.gpl_input.items():
                print('{} => {}'.format(key, value))
            print('\n')

        with open(self.workdir + self.uid + '_gpl.in', 'w') as f:
            f.write(gpl_input_string)
        if self.verbose:
            subprocess.call(self.rootdir + 'src/galaxevpl < ' + self.uid + '_gpl.in',
                            cwd=self.workdir, shell=True)
        else:
            subprocess.call(self.rootdir + 'src/galaxevpl < ' + self.uid + '_gpl.in',
                            cwd=self.workdir, shell=True, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))
        if self.del_input: self.del_file(self.workdir + self.uid + '_gpl.in')

    def gpl_cleanup(self):
        """
        Remove any non-essential files that were generated by the gpl (galaxevpl)
        utility.
        """
        self.del_file(self.workdir + self.ssp_output + '.ised')
        self.del_file(self.workdir + self.csp_output + '.ised')

    def read_gpl(self):
        """
        Reads selected data from the files generated by the galaxevpl utility and
        constructs the 'sed' member datum, which is a multidimensional numpy array
        whose first column ('wave') specifies the wavelength in Angström and subsequent
        columns specify the luminosity (in frequency or wavelength units depending upon
        the value of the class's 'units' construction option) at each of the ages (in Gyr)
        specified upon construction.
        """
        # define column names and data types for extraction of synthetic spectra
        dtype = [('wave', float), ] + [('spec{}'.format(i + 1), float)
                                                      for i in range(len(self.age))]
        # Extract the synthetic sperctra that were generated for the specified post-starburst ages
        self.sed = np.genfromtxt(self.workdir + self.csp_output + '.spec', dtype=dtype)
        # Extract the log of the rate of H-ionizing photons (in Hz) for the
        # synthetic spectra as a function of post-starburst age
        age3, Q = np.genfromtxt(self.workdir + self.csp_output +
                                '.3color', usecols=(0, 5), unpack=True)
        # Extract the appropriate normalization factor for the synthetic spectra
        # as a function of post-starburst age
        age4, M = np.genfromtxt(self.workdir + self.csp_output +
                                '.4color', usecols=(0, 6), unpack=True)

        # Loop over union of synthetic spectra labels and the ages for which they were generated
        for x, age in zip(self.sed.dtype.names[1:], self.age):

            # scale from solar units to erg/s/Å
            self.sed[x] = self.sed[x] * 3.826e33
            self.sed[x][self.sed["wave"] < 912.] = self.sed[x][self.sed["wave"] < 912.] * self.lyc_esc

            # compute the log of the age in years
            log_age = np.log10(age * 1e9)
            diff = abs(age3 - log_age)
            # store the closest log of the rate of H-ionizing photons (in Hz) for a synthetic
            # spectrum with an age equal to that currently under consideration.
            self.Q[x] = Q[diff == min(diff)][0]

            diff = abs(age4 - log_age)
            # store the closest appropriate normalization factor for a synthetic
            # spectrum with an age equal to that currently under consideration.
            self.M_unnorm[x] = M[diff == min(diff)][0]

    def post_gpl_cleanup(self):

        os.unlink(self.workdir + self.csp_output + '.spec')
        os.unlink(self.workdir + self.csp_output + '.3color')
        os.unlink(self.workdir + self.csp_output + '.4color')

    def add_emlines(self):

        for x in self.sed.dtype.names[1:]:
            self.sed[x] = add_emission_lines(self.sed['wave'], self.sed[x], self.Q[x], self.metallicity, self.units, lya_esc=self.lya_esc)

    def add_dust(self):

        for x in self.sed.dtype.names[1:]:
            if self.dust == 'calzetti':
                self.sed[x] = self.sed[x] * \
                    np.exp(-calzetti(self.sed['wave'], self.Av))
            elif self.dust == 'cardelli':
                self.sed[x] = self.sed[x] * \
                    np.exp(-cardelli(self.sed['wave'], self.Av))

    def redshift_evo(self):

        self.sed['wave'] *= (1 + self.redshift)
        if self.igm:
            for x in self.sed.dtype.names[1:]:
                self.sed[x] = self.sed[x] * np.exp(-inoue_tau(self.sed['wave'], self.redshift))

    def plot_sed(self, save=None):

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=75, tight_layout=True)

        colors = plt.cm.gist_rainbow_r(np.linspace(0.1, 0.95, len(self.sed.dtype.names[1:])))
        for age, x, c in zip(self.age, self.sed.dtype.names[1:], colors):
            ax.plot(self.sed['wave'], self.sed[x], c=c, lw=1.5, alpha=0.8, label="Age = %.g Gyr" % age)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength ($\AA$)')
        if self.units == 'flambda':
            ax.set_ylabel(r'F$_\lambda$ [ergs/s/$\AA$]')
        elif self.units == 'fnu':
            ax.set_ylabel(r'F$_\nu$ [ergs/s/Hz]')

        title = "BC03 SED {:s} model -- ".format(str(self.library_version)) + \
                "IMF=" + self.imf_dir_key[self.imf] + ", " \
                "Z=" + str(self.metallicity) + ", " \
                "SFH=" + self.sfh + ", "
        if self.sfh == 'exp':
            title += r"$\tau$=" + str(self.tau) + ", "
        if self.sfh == 'single':
            title += r"$\Delta$=" + str(self.tau) + ", "
        title += "EmLines=" + str(self.emlines) + ", "
        title += "Av=" + str(self.Av) + ", " if self.dust else "Av=0, "
        title += "redshift=" + str(self.redshift)

        ax.set_title(title)
        ax.legend(fontsize=16)
        ax.set_xlim(5e2, 1e7)
        ax.set_ylim(1e20, 1e31)

        if save:
            fig.savefig(save)
        else:
            plt.show()
