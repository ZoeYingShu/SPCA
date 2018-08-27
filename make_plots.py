
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

import bliss

def plot_photometry(time0, flux0, xdata0, ydata0, psfxw0, psfyw0, 
                    time, flux, xdata, ydata, psfxw, psfyw, breaks=[], savepath='', peritime=''):
    '''
    Makes a multi-panel plot from photometry outputs.
    params:
    -------
        time0 : 1D array 
            array of time stamps. Discarded points not removed.
        flux0 : 1D array
            array of flux values for each time stamps. Discarded points not removed.
        xdata0 : 1D array
            initial modelled the fluxes for each time stamps. Discarded points not removed.
        ydata0: 1D array
            initial modelled astrophysical flux variation for each time stamps. 
            Discarded points not removed.
        psfxw0: 1D array
            Point-Spread-Function (PSF) width along the x-direction. Discarded points not removed.
        psfyw0: 1D array
            Point-Spread-Function (PSF) width along the x-direction. Discarded points not removed.
        time  : 1D array 
            array of time stamps. Discarded points removed.
        flux  : 1D array
            array of flux values for each time stamps. Discarded points removed.
        xdata  : 1D array
            initial modelled the fluxes for each time stamps. Discarded points removed.
        ydata : 1D array
            initial modelled astrophysical flux variation for each time stamps. Discarded points removed.
        psfxw : 1D array
            Point-Spread-Function (PSF) width along the x-direction. Discarded points removed.
        psfyw : 1D array
            Point-Spread-Function (PSF) width along the x-direction. Discarded points removed.
        break : 1D array
            time of the breaks from one AOR to another.
        savepath : str
            path to directory where the plot will be saved
        pertime  : float
            time of periapsis
    returns:
    --------
        none
    '''
    
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 12))
    #fig.suptitle("XO-3b Observation")

    axes[0].plot(time0, flux0,  'r.', markersize=1, alpha = 0.7)
    axes[0].plot(time, flux,  'k.', markersize=2, alpha = 1.0)
    axes[0].set_ylabel("Relative Flux $F$")
    axes[0].set_xlim((np.min(time0), np.max(time0)))

    axes[1].plot(time0, xdata0,  'r.', markersize=1, alpha = 0.7)
    axes[1].plot(time, xdata,  'k.', markersize=2, alpha = 1.0)
    axes[1].set_ylabel("x-centroid $x_0$")

    axes[2].plot(time0, ydata0,  'r.', markersize=1, alpha = 0.7)
    axes[2].plot(time, ydata, 'k.', markersize=2, alpha = 1.0)
    axes[2].set_ylabel("y-centroid $y_0$")

    axes[3].plot(time0, psfxw0,  'r.', markersize=1, alpha = 0.7)
    axes[3].plot(time, psfxw, 'k.', markersize=2, alpha = 1.0)
    axes[3].set_ylabel("x PSF-width $\sigma _x$")

    axes[4].plot(time0, psfyw0,  'r.', markersize=1, alpha = 0.7)
    axes[4].plot(time, psfyw,  'k.', markersize=2, alpha = 1.0)
    axes[4].set_ylabel("y PSF-width $\sigma _y$")
    axes[4].set_xlabel('Time (BMJD)')

    for i in range(5):
        axes[i].axvline(x=peritime, color ='C1', alpha=0.8, linestyle = 'dashed')
        for j in range(len(breaks)):
            axes[i].axvline(x=breaks[j], color ='k', alpha=0.3, linestyle = 'dashed')

    fig.subplots_adjust(hspace=0)
    pathplot = savepath + '01_Raw_data.pdf'
    fig.savefig(pathplot, bbox_inches='tight')
    return


def plot_centroids(xdata0, ydata0, xdata, ydata, savepath=''):
    '''
    Makes a multi-panel plot from photometry outputs.
    params:
    -------
        xdata0 : 1D array
            initial modelled the fluxes for each time stamps. Discarded points not removed.
        ydata0: 1D array
            initial modelled astrophysical flux variation for each time stamps. 
            Discarded points not removed.
        xdata  : 1D array
            initial modelled the fluxes for each time stamps. Discarded points removed.
        ydata : 1D array
            initial modelled astrophysical flux variation for each time stamps. Discarded points removed.
        savepath : str
            path to directory where the plot will be saved
    returns:
    --------
        none
    '''
    
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    ax.set_title('Distribution of centroids')
    
    ax.plot(xdata0, ydata0,  'r.', markersize=1, alpha = 0.7)
    ax.plot(xdata, ydata,  'k.', markersize=2, alpha = 1.0)
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")

    fig.subplots_adjust(hspace=0)
    pathplot = savepath + 'Centroids.pdf'
    fig.savefig(pathplot, bbox_inches='tight')
    return


def plot_knots(xdata, ydata, flux, astroModel, knot_nrst_lin,
               tmask_good_knotNdata, knots_x, knots_y, 
               knots_x_mesh, knots_y_mesh, nBin, knotNdata, savepath=None):
    '''Plot the Bliss map'''
    
    fB_avg = bliss.map_flux_avgQuick(flux, astroModel, knot_nrst_lin, nBin, knotNdata)
    delta_xo, delta_yo = knots_x[1] - knots_x[0], knots_y[1] - knots_y[0]
    
    star_colrs = knotNdata[tmask_good_knotNdata]
    
    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.scatter(xdata, ydata,color=(0,0,0),alpha=0.2,s=2,marker='.')
    plt.gca().set_aspect((knots_x[-1]-knots_x[0])/(knots_y[-1]-knots_y[0]))
    plt.xlabel('Pixel x',size='x-large');
    plt.ylabel('Pixel y',size='x-large');
    plt.title('Knot Mesh',size='large')
    plt.xlim([knots_x[0] - 0.5*delta_xo, knots_x[-1] + 0.5*delta_xo])
    plt.ylim([knots_y[0] - 0.5*delta_yo, knots_y[-1] + 0.5*delta_yo])
    plt.locator_params(axis='x',nbins=8)
    plt.locator_params(axis='y',nbins=8)
    my_stars = plt.scatter(knots_x_mesh[tmask_good_knotNdata],
                           knots_y_mesh[tmask_good_knotNdata],
                           c=star_colrs, cmap=matplotlib.cm.Purples,
                           edgecolor='k',marker='*',s=175,vmin=1)
    plt.colorbar(my_stars, label='Linked Centroids',shrink=0.75)
    plt.scatter(knots_x_mesh[tmask_good_knotNdata == False],
                knots_y_mesh[tmask_good_knotNdata == False], 
                color=(1,0.75,0.75), marker='x',s=35)
    legend = plt.legend(('Centroids','Good Knots','Bad Knots'),
                        loc='lower right',bbox_to_anchor=(0.975,0.025),
                        fontsize='small',fancybox=True)
    legend.legendHandles[1].set_color(matplotlib.cm.Purples(0.67)[:3])
    legend.legendHandles[1].set_edgecolor('black')

    if savepath!=None:
        pathplot = savepath+'BLISS_Knots.pdf'
        plt.savefig(pathplot, bbox_inches='tight')
    else:
        plt.show()


def plot_detec_syst(time, data, init):
    plt.figure(figsize=(10,3))
    plt.plot(time, data, '+', label='data')
    plt.plot(time, init, '+', label='guess')
    plt.title('Initial Guess')
    plt.xlabel('Time (BMJD)')
    plt.ylabel('Relative Flux')	
    
    return

def plot_init_guess(time, data, astro, detec_full, savepath, mode):
    '''
    Makes a multi-panel plots for the initial light curve guesses.
    params:
    -------
        time  : 1D array 
            array of time stamps
        data  : 1D array
            array of flux values for each time stamps
        astro : 1D array
            initial modelled astrophysical flux variation for each time stamps
        detec_full : 1D array
            initial modelled flux variation due to the detector for each time stamps
        savepath : str
            path to directory where the plot will be saved
    returns:
    --------
        none
    '''
    
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,9))
    
    axes[0].plot(time, data, '.', label='data')
    axes[0].plot(time, astro*detec_full, '.', label='guess')
    
    axes[1].plot(time, data/detec_full, '.', label='Corrected')
    axes[1].plot(time, astro, '.', label='Astrophysical')
    
    axes[2].plot(time, data/detec_full, '.', label='Corrected')
    axes[2].plot(time, astro, '.', label='Astrophysical')
    axes[2].set_ylim(0.998, 1.005)
    
    axes[3].plot(time, data/detec_full-astro, '.', label='residuals')
    axes[3].axhline(y=0, linewidth=2, color='black')
    
    axes[0].set_ylabel('Relative Flux')
    axes[2].set_ylabel('Relative Flux')
    axes[2].set_xlabel('Time (BMJD)')
    
    axes[0].legend(loc=3)
    axes[1].legend(loc=3)
    axes[2].legend(loc=3)
    axes[3].legend(loc=3)
    axes[3].set_xlim(np.min(time), np.max(time))
    
    fig.subplots_adjust(hspace=0)
    pathplot = savepath + '02_Initial_Guess.pdf'
    fig.savefig(pathplot, bbox_inches='tight')
    return

def plot_bestfit(x, flux, astro, detec_full, mode, breaks, savepath, peritime=-np.inf):
    fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize=(8, 10))
    
    axes[0].set_xlim(np.min(x), np.max(x))
    axes[0].plot(x, flux, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[0].plot(x, astro*detec_full, '.', color = 'r', markersize = 2.5, alpha = 0.4)
    #axes[0].set_ylim(0.975, 1.0125)
    axes[0].set_ylabel('Raw Flux')

    axes[1].plot(x, flux/detec_full, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[1].plot(x, astro, color = 'r', linewidth=2)
    axes[1].set_ylabel('Calibrated Flux')
    #axes[1].set_ylim(0.9825, 1.0125)
    
    axes[2].axhline(y=1, color='k', linewidth = 2, linestyle='dashed', alpha = 0.5)
    axes[2].plot(x, flux/detec_full, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[2].plot(x, astro, color = 'r', linewidth=2)
    axes[2].set_ylabel('Calibrated Flux')
    #axes[2].set_ylim(0.9975, 1.0035)
    axes[2].set_ylim(ymin=0.9975)

    axes[3].plot(x, flux/detec_full - astro, 'k.', markersize = 4, alpha = 0.15)
    axes[3].axhline(y=0, color='r', linewidth = 2)
    axes[3].set_ylabel('Residuals')
    axes[3].set_xlabel('Orbital Phase')
    #axes[3].set_ylim(-0.007, 0.007)

    for i in range(len(axes)):
        axes[i].axvline(x=peritime, color ='C1', alpha=0.8, linestyle = 'dashed')
        for j in range(len(breaks)):
            axes[i].axvline(x=(breaks[j]), color ='k', alpha=0.3, linestyle = 'dashed')
    #fig.align_ylabels()
    
    fig.subplots_adjust(hspace=0)
    plotname = savepath + 'MCMC_'+mode+'_2.pdf'
    fig.savefig(plotname, bbox_inches='tight')
    return
