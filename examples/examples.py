from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import extract_bc03 as bc03; reload(bc03)

template = bc03.TemplateSED_BC03(metallicity=0.02, age=[1, 2, 3, 5, 10], sfh='exp', tau=2, Av=1,
                            # input_ised='bc2003_lr_BaSeL_m52_chab_ssp.ised',
                            dust='calzetti', emlines=True,
                            redshift=2, uid='z2',
                            igm=True, imf='chab', res='hr', units='flambda',
                            workdir='/Users/albert/workplace/learnBC03',
                            library_version=2016, library='xmiless',
                            cleanup=True, del_input=True, verbose=True)
template.generate_sed()

fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=75, tight_layout=True)

cind = np.linspace(0.2, 0.95, 5)
for x, c in zip(template.sed.dtype.names[1:], plt.cm.Greys(cind)):
    ax.plot(template.sed['wave'], template.sed[x], c=c, lw=1.25, alpha=0.8)

template.add_emlines()
for x, c in zip(template.sed.dtype.names[1:], plt.cm.Blues(cind)):
    ax.plot(template.sed['wave'], template.sed[x], c=c, lw=1.25, alpha=0.8)

template.add_dust()
for x, c in zip(template.sed.dtype.names[1:], plt.cm.Reds(cind)):
    ax.plot(template.sed['wave'], template.sed[x], c=c, lw=1.25, alpha=0.8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Wavelength [$\AA$]')
ax.set_ylabel('L$_\lambda$ [ergs/s/$\AA$]')
ax.set_xlim(1e2, 1e8)
ax.set_ylim(1e17, 1e31)
plt.show()

