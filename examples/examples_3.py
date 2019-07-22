from importlib import reload
import numpy as np
import extract_bc03 as bc03; reload(bc03)

workdir = '/Users/albert/workplace/learnBC03/'
metal = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]
metal_str = ['m22', 'm32', 'm42', 'm52', 'm62', 'm72']
Av_vals = [0., 1., 2.]
# age = [3.e-3, 0.01, 0.05, 0.2, 1., 5., 10.]     # in units of Gyr
logtage = np.linspace(-2.,1.,8)
age = 10.**logtage
age = np.sort(np.append(age, [0.1, 0.5, 1., 3., 5.]))

lya_esc=1.
lyc_esc=1.

for ii, m in enumerate(metal):
    for Av in Av_vals:
        template = bc03.TemplateSED_BC03(metallicity=m, age=age, sfh='ssp', tau=None, Av=Av,
                                         dust='calzetti', emlines=True,
                                         redshift=None, uid='bc2012_BaSel_ssp_chab_'+metal_str[ii]+'_Av'+str(int(Av)),
                                         igm=True, imf='chab', res='lr', units='flambda',
                                         workdir=workdir, lya_esc=lya_esc, lyc_esc=lyc_esc,
                                         library_version=2012, library='BaSeL',
                                         cleanup=False, del_input=True, verbose=True)
        template.generate_sed()

