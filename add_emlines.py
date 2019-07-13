import astropy.io.fits as pyfits
import numpy as np
import os

# extractBC03Root = os.getenv('BC03ROOT', '.')
extractBC03Root = os.path.expanduser('~/Dropbox/Python/extract_BC03')

line_ratio_file = os.path.join(extractBC03Root,'emline_ratios.fits')
line_ratios = pyfits.getdata(line_ratio_file)

# ADD EMISSION LINES


def print_available_lines():
    lineRatioHdus = pyfits.open(line_ratio_file)
    print(lineRatioHdus[1].data.columns)


def add_emission_lines(sed_waves, sed_spec, Q, metallicity, units, lya_esc=0.2):
    """
    sed_waves - is the array of wavelengths.
    sed_spec - is the array of spectrum values.
    Q - is the log of the rate of H-ionizing photons (in Hz) for the galaxy spectrum.
    metallicity - is the metallicity in solar units.
    units - determines whether the spectrum is specified in wavelength or frequency space.
    """
    l_hb = 4861.
    l_ha = 6563.
    l_lya = 1216.

    lum_hb = 4.757e-13 * 10**Q              # See Hbeta_flux_calc.pdf
    lum_ha = 1.37e-12 * 10**Q               # Ha/Hb = 2.87
    lum_lya = 1.04e-11 * 10**Q * lya_esc    # Lya/Ha = 7.59

    ### Rest of the line fluxes are specified
    ### line ratios w.r.t. Hb flux
    ### following Anders+03 (A&A, 401, 1063)
    if metallicity in [0.0001, 0.0004]:
        ratios = line_ratios['M32_RATIO'][0]
    elif metallicity in [0.004, ]:
        ratios = line_ratios['M42_RATIO'][0]
    elif metallicity in [0.008, 0.02, 0.05, 0.1]:
        ratios = line_ratios['M52_62_72_RATIO'][0]
    else:
        raise Exception('Incorrect metallicity provided in add_lines().')

    line_centers = np.concatenate(([l_lya, l_hb, l_ha], line_ratios['LAMBDA'][0]))
    line_lums = np.concatenate(([lum_lya, lum_hb, lum_ha], ratios * lum_hb))

    sigma = 3  # Angs

    for line_center, line_lum in zip(line_centers, line_lums):

        line_prof = (1./np.sqrt(2*np.pi)/sigma) * np.exp(-0.5*(sed_waves - line_center)**2 / sigma**2)
        line_prof = line_lum * line_prof
        if units == 'flambda':
            sed_spec = sed_spec + line_prof
        elif units == 'fnu':
            sed_spec = sed_spec / (sed_waves**2 / 3e18)
            sed_spec = sed_spec + line_prof
            sed_spec = sed_spec * (sed_waves**2 / 3e18)
        else:
            raise Exception('Wrong units in add_emission_lines().')

    return sed_spec
