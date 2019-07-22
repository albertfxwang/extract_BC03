from PlotParams import *
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import extract_bc03 as bc03; reload(bc03)
import sys, os

workdir = '/Users/albert/workplace/learnBC03/'
metal = 0.004
age = [3.e-3, 0.01, 0.05, 0.2, 1., 5., 10.]
lya_esc=0.2
lyc_esc=0.

template_a = bc03.TemplateSED_BC03(metallicity=metal, age=age, sfh='ssp', tau=None, Av=2.,
                                   dust='calzetti', emlines=False,
                                   redshift=None, uid='ssp_Av2',
                                   igm=True, imf='chab', res='lr', units='flambda',
                                   workdir=workdir, lya_esc=lya_esc, lyc_esc=lyc_esc,
                                   library_version=2012, library='BaSeL',
                                   cleanup=False, del_input=True, verbose=True)
template_a.generate_sed()

template_b = bc03.TemplateSED_BC03(metallicity=metal, age=age, sfh='ssp', tau=None, Av=1.,
                                   dust='calzetti', emlines=False,
                                   redshift=None, uid='ssp_Av1',
                                   igm=True, imf='chab', res='lr', units='flambda',
                                   workdir=workdir, lya_esc=lya_esc, lyc_esc=lyc_esc,
                                   library_version=2012, library='BaSeL',
                                   cleanup=False, del_input=True, verbose=True)
template_b.generate_sed()

template_c = bc03.TemplateSED_BC03(metallicity=metal, age=age, sfh='ssp', tau=None, Av=0.,
                                   dust='calzetti', emlines=False,
                                   redshift=None, uid='ssp_Av0',
                                   igm=True, imf='chab', res='lr', units='flambda',
                                   workdir=workdir, lya_esc=lya_esc, lyc_esc=lyc_esc,
                                   library_version=2012, library='BaSeL',
                                   cleanup=False, del_input=True, verbose=True)
template_c.generate_sed()

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Plotting
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
fig, axs = plt.subplots(1, 3, figsize=(16, 4), dpi=75, tight_layout=True)

#-------- normalize in terms of the flux at ~5500 Angstrom
ind = np.argmin(np.abs(template_a.sed['wave'] - 5500.))

cind = np.linspace(0.4, 0.98, len(age))
for x, c in zip(template_a.sed.dtype.names[1:], plt.cm.Greys(cind)):
    # axs[0].plot(template.sed['wave'], template.sed[x]/template.sed[template.sed.dtype.names[-1]][ind], c=c, lw=1.25, alpha=0.8)
    axs[0].plot(template_a.sed['wave'], template_a.sed[x] / template_a.sed[x][ind], c=c, lw=1.25, alpha=0.8)
axs[0].set_title('Av = 2')

for x, c in zip(template_b.sed.dtype.names[1:], plt.cm.Blues(cind)):
    # axs[1].plot(template_noEL.sed['wave'], template_noEL.sed[x]/template_noEL.sed[template_noEL.sed.dtype.names[-1]][ind], c=c, lw=1.25, alpha=0.8)
    axs[1].plot(template_b.sed['wave'], template_b.sed[x] / template_b.sed[x][ind], c=c, lw=1.25, alpha=0.8)
axs[1].set_title('Av = 1')

for x, c in zip(template_c.sed.dtype.names[1:], plt.cm.Reds(cind)):
    # axs[2].plot(template_nodust.sed['wave'], template_nodust.sed[x]/template_nodust.sed[template_nodust.sed.dtype.names[-1]][ind], c=c, lw=1.25, alpha=0.8)
    axs[2].plot(template_c.sed['wave'], template_c.sed[x] / template_c.sed[x][ind], c=c, lw=1.25, alpha=0.8)
axs[2].set_title('Av = 0')

axs[0].set_ylabel('L$_\lambda$ [ergs/s/$\AA$]')
for ax in axs:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength [$\AA$]')
    ax.set_xlim(8.e2, 1.e5)
    ax.set_ylim(1.e-4, 4.e2)
    set_gca(ax, ticks_fs=gca_fs - 3)

plt.show()
