import numpy as np
from scipy import interpolate

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

from photutils import aperture_photometry
from photutils import CircularAperture, EllipticalAperture, RectangularAperture
from photutils.utils import calc_total_error

import os, sys, csv, glob, warnings

from collections import Iterable

from .Photometry_Common import get_fnames, get_stacks, get_time, oversampling, sigma_clipping, bgsubtract, binning_data

def centroid_FWM(image_data, xo=None, yo=None, wx=None, wy=None, scale=1, bounds=(14, 18, 14, 18)):
    """Gets the centroid of the target by flux weighted mean and the PSF width of the target.

    Args:
        image_data (ndarray): Data cube of images (2D arrays of pixel values).
        xo (list, optional): List of x-centroid obtained previously. Default is None.
        yo (list, optional):  List of y-centroids obtained previously. Default is None.
        wx (list, optional):  List of PSF width (x-axis) obtained previously. Default is None.
        wy (list, optional): List of PSF width (x-axis) obtained previously. Default is None.
        scale (int, optional): If the image is oversampled, scaling factor for centroid and bounds, i.e, give centroid in terms of the pixel value of the initial image.
        bounds (tuple, optional): Bounds of box around the target to exclude background . Default is (14, 18, 14, 18).
    
    Returns:
        tuple: xo, yo, wx, wy (list, list, list, list). The updated lists of x-centroid, y-centroid,
            PSF width (x-axis), and PSF width (y-axis).
    """
    
    if xo is None:
        xo=[]
    if yo is None:
        yo=[]
    if wx is None:
        wx=[]
    if wy is None:
        wy=[]
    
    lbx, ubx, lby, uby = np.array(bounds)*scale
    starbox = image_data[:, lbx:ubx, lby:uby]
    h, w, l = starbox.shape
    # get centroid
    Y, X    = np.mgrid[:w,:l]
    cx      = (np.sum(np.sum(X*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1))) + lbx
    cy      = (np.sum(np.sum(Y*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1))) + lby
    try:
        cx      = sigma_clip(cx, sigma=4, maxiters=2, cenfunc=np.ma.median)
        cy      = sigma_clip(cy, sigma=4, maxiters=2, cenfunc=np.ma.median)
    except TypeError:
        cx      = sigma_clip(cx, sigma=4, iters=2, cenfunc=np.ma.median)
        cy      = sigma_clip(cy, sigma=4, iters=2, cenfunc=np.ma.median)
    xo.extend(cx/scale)
    yo.extend(cy/scale)
    # get PSF widths
    X, Y    = np.repeat(X[np.newaxis,:,:], h, axis=0), np.repeat(Y[np.newaxis,:,:], h, axis=0)
    cx, cy  = np.reshape(cx, (h, 1, 1)), np.reshape(cy, (h, 1, 1))
    X2, Y2  = (X + lbx - cx)**2, (Y + lby - cy)**2
    widx    = np.sqrt(np.sum(np.sum(X2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
    widy    = np.sqrt(np.sum(np.sum(Y2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
    try:
        widx    = sigma_clip(widx, sigma=4, maxiters=2, cenfunc=np.ma.median)
        widy    = sigma_clip(widy, sigma=4, maxiters=2, cenfunc=np.ma.median)
    except TypeError:
        widx    = sigma_clip(widx, sigma=4, iters=2, cenfunc=np.ma.median)
        widy    = sigma_clip(widy, sigma=4, iters=2, cenfunc=np.ma.median)
    wx.extend(widx/scale)
    wy.extend(widy/scale)
    return xo, yo, wx, wy

def A_photometry(image_data, bg_err, ape_sum = None, ape_sum_err = None,
    cx = 15, cy = 15, r = 2.5, a = 5, b = 5, w_r = 5, h_r = 5, 
    theta = 0, shape = 'Circular', method='center'):
    """Performs aperture photometry, first by creating the aperture then summing the flux within the aperture.

    Args:
        image_data (3D array): Data cube of images (2D arrays of pixel values).
        bg_err (1D array): Array of uncertainties on pixel value.
        ape_sum (1D array, optional): Array of flux to append new flux values to.
            If None, the new values will be appended to an empty array
        ape_sum_err (1D array, optional): Array of flux uncertainty to append new flux uncertainty values to.
            If None, the new values will be appended to an empty array.
        cx (int, optional): x-coordinate of the center of the aperture. Default is 15.
        cy (int, optional): y-coordinate of the center of the aperture. Default is 15.
        r (int, optional): If shape is 'Circular', r is the radius for the circular aperture. Default is 2.5.
        a (int, optional): If shape is 'Elliptical', a is the semi-major axis for elliptical aperture (x-axis). Default is 5.
        b (int, optional): If shape is 'Elliptical', b is the semi-major axis for elliptical aperture (y-axis). Default is 5.
        w_r (int, optional): If shape is 'Rectangular', w_r is the full width for rectangular aperture (x-axis). Default is 5.
        h_r (int, optional): If shape is 'Rectangular', h_r is the full height for rectangular aperture (y-axis). Default is 5.
        theta (int, optional): If shape is 'Elliptical' or 'Rectangular', theta is the angle of the rotation angle in radians
            of the semimajor axis from the positive x axis. The rotation angle increases counterclockwise. Default is 0.
        shape (string, optional): shape is the shape of the aperture. Possible aperture shapes are 'Circular',
            'Elliptical', 'Rectangular'. Default is 'Circular'.
        method (string, optional): The method used to determine the overlap of the aperture on the pixel grid. Possible 
            methods are 'exact', 'subpixel', 'center'. Default is 'center'.

    Returns:
        tuple: ape_sum (1D array) Array of flux with new flux appended.
            ape_sum_err (1D array) Array of flux uncertainties with new flux uncertainties appended.

    """
    
    if ape_sum is None:
        ape_sum = []
    if ape_sum_err is None:
        ape_sum_err = []
    
    l, h, w  = image_data.shape
    tmp_sum  = []
    tmp_err  = []
    movingCentroid = (isinstance(cx, Iterable) and isinstance(cy, Iterable))
    if not movingCentroid:
        position = [cx, cy]
        if   (shape == 'Circular'):
            aperture = CircularAperture(position, r=r)
        elif (shape == 'Elliptical'):
            aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
        elif (shape == 'Rectangular'):
            aperture = RectangularAperture(position, w=w_r, h=h_r, theta=theta)
    for i in range(l):
        if movingCentroid:
            position = [cx[i], cy[i]]
            if   (shape == 'Circular'):
                aperture = CircularAperture(position, r=r)
            elif (shape == 'Elliptical'):
                aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
            elif (shape == 'Rectangular'):
                aperture = RectangularAperture(position, w=w_r, h=h_r, theta=theta)
        data_error = calc_total_error(image_data[i,:,:], bg_err[i], effective_gain=1)
        phot_table = aperture_photometry(image_data[i,:,:], aperture, error=data_error, method=method)
        tmp_sum.extend(phot_table['aperture_sum'])
        tmp_err.extend(phot_table['aperture_sum_err'])
    # removing outliers
    try:
        tmp_sum = sigma_clip(tmp_sum, sigma=4, maxiters=2, cenfunc=np.ma.median)
        tmp_err = sigma_clip(tmp_err, sigma=4, maxiters=2, cenfunc=np.ma.median)
    except TypeError:
        tmp_sum = sigma_clip(tmp_sum, sigma=4, iters=2, cenfunc=np.ma.median)
        tmp_err = sigma_clip(tmp_err, sigma=4, iters=2, cenfunc=np.ma.median)
    ape_sum.extend(tmp_sum)
    ape_sum_err.extend(tmp_err)
    return ape_sum, ape_sum_err


def get_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
    save = True, save_full = '/ch2_datacube_full_AORs579.dat', bin_data = True, 
    bin_size = 64, save_bin = '/ch2_datacube_binned_AORs579.dat', plot = True, 
    plot_name= 'Lightcurve.pdf', oversamp = False, saveoversamp = True, reuse_oversamp = False,
    planet = 'CoRoT-2b', r = 2.5, shape = 'Circular', edge='hard', addStack = False,
    stackPath = '', ignoreFrames = None, maskStars = None, moveCentroid=False):
    """Given a directory, looks for data (bcd.fits files), opens them and performs photometry.

    Args:
        datapath (string): Directory where the spitzer data is stored.
        savepath (string): Directory the outputs will be saved.
        AORsnip (string):  Common first characters of data directory eg. 'r579'
        channel (string): Channel used for the observation eg. 'ch1' for channel 1
        subarray (bool): True if observation were taken in subarray mode. False if observation were taken in full-array mode.
        shape (string, optional): shape is the shape of the aperture. Possible aperture shapes are 'Circular',
            'Elliptical', 'Rectangular'. Default is 'Circular'.
        edge (string, optional): A string specifying the type of aperture edge to be used. Options are 'hard', 'soft',
            and 'exact' which correspond to the 'center', 'subpixel', and 'exact' methods. Default is 'hard'.
        save (bool, optional): True if you want to save the outputs. Default is True.
        save_full (string, optional): Filename of the full unbinned output data. Default is '/ch2_datacube_full_AORs579.dat'.
        bin_data (bool, optional): True you want to get binned data. Default is True.
        bin_size (int, optional): If bin_data is True, the size of the bins. Default is 64.
        save_bin (string, optional): Filename of the full binned output data. Default is '/ch2_datacube_binned_AORs579.dat'.
        plot (bool, optional): True if you want to plot the time resolved lightcurve. Default is True.
        plot_name (string, optional): If plot and save is True, the filename of the plot to be saved as. Default is True.
        oversamp (bool, optional): True if you want to oversample the image by a factor of 2. Default is False.
        save_oversamp (bool, optional): True if you want to save oversampled images. Default is True.
        reuse_oversamp (bool, optional): True if you want to reuse oversampled images that were previously saved.
            Default is False.
        planet (string, optional): The name of the planet. Default is CoRoT-2b.
        r (float, optional): The radius to use for aperture photometry in units of pixels. Default is 2.5 pixels.
        ignoreFrames (list, optional) A list of frames to be masked when performing aperature photometry (e.g. first
            frame to remove first-frame systematic).
        maskStars (list, optional): An array-like object where each element is an array-like object with the RA and DEC
            coordinates of a nearby star which should be masked out when computing background subtraction.
        moveCentroid (bool, optional): True if you want the centroid to be centered on the flux-weighted mean centroids
            (will default to 15,15 when a NaN is returned), otherwise aperture will be centered on 15,15
            (or 30,30 for 2x oversampled images). Default is False.

    Raises: 
        Error: If Photometry method is not supported/recognized by this pipeline.
    
    """

    if not subarray:
        # FIX: Throw an actual error
        print('Error: Full frame photometry is not yet supported.')
        return
    
    if shape!='Circular' and shape!='Elliptical' and shape!='Rectangular':
        # FIX: Throw an actual error
        print('Warning: No such aperture shape "'+shape+'". Using Circular aperture instead.')
        shape='Circular'
    
    if edge.lower()=='hard' or edge.lower()=='center' or edge.lower()=='centre':
        method = 'center'
    elif edge.lower()=='soft' or edge.lower()=='subpixel':
        method = 'subpixel'
    elif edge.lower()=='exact':
        method = 'exact'
    else:
        # FIX: Throw an actual error
        print("Warning: No such method \""+edge+"\". Using hard edged aperture")
        method = 'center'
    
    if savepath[-1]!='/':
        savepath += '/'
    
    if ignoreFrames is None:
        ignoreFrames = []
    if maskStars is None:
        maskStars = []
    
    # Ignore warning
    warnings.filterwarnings('ignore')

    # get list of filenames and nb of files
    fnames, lens = get_fnames(datapath, AOR_snip, channel)
    if addStack:
        stacks = get_stacks(stackPath, datapath, AOR_snip, channel)

    # variables declaration 
    percent       = 0                                # to show progress while running the code
    tossed        = 0                                # Keep tracks of number of frame discarded 
    badframetable = []                               # list of filenames of the discarded frames
    flux          = []                               # flux obtained from aperture photometry
    flux_err      = []                               # error on flux obtained from aperture photometry
    time          = []                               # time array
    xo            = []                               # centroid value along the x-axis
    yo            = []                               # centroid value along the y-axis
    xw            = []                               # PSF width along the x-axis
    yw            = []                               # PSF width along the y-axis
    bg_flux       = []                               # background flux
    bg_err        = []                               # background flux error 
    
    # variables declaration for binned data
    binned_flux          = []                        # binned flux obtained from aperture photometry
    binned_flux_std      = []                        # std.dev in binned error on flux obtained from aperture photometry
    binned_time          = []                        # binned time array
    binned_time_std      = []                        # std.dev in binned time array
    binned_xo            = []                        # binned centroid value along the x-axis
    binned_xo_std        = []                        # std.dev in binned centroid value along the x-axis
    binned_yo            = []                        # binned centroid value along the y-axis
    binned_yo_std        = []                        # std.dev in binned centroid value along the y-axis
    binned_xw            = []                        # binned PSF width along the x-axis
    binned_xw_std        = []                        # std.dev in binned PSF width along the x-axis
    binned_yw            = []                        # binned PSF width along the y-axis
    binned_yw_std        = []                        # std.dev in binned PSF width along the y-axis
    binned_bg            = []                        # binned background flux
    binned_bg_std        = []                        # std.dev in binned background flux
    binned_bg_err        = []                        # binned background flux error 
    binned_bg_err_std    = []                        # std.dev in binned background flux error 
    
    # data reduction & aperture photometry part
    if subarray:
        j=0 #counter to keep track of which correction stack we're using
        for i in range(len(fnames)):
            # open fits file
            hdu_list = fits.open(fnames[i])
            image_data0 = hdu_list[0].data
            # get time
            time = get_time(hdu_list, time, ignoreFrames)
            #add background correcting stack if requested
            if addStack:
                while i > np.sum(lens[:j+1]):
                    j+=1 #if we've moved onto a new AOR, increment j
                stackHDU = fits.open(stacks[j])
                image_data0 += stackHDU[0].data
            #ignore any consistently bad frames
            if ignoreFrames != []:
                image_data0[ignoreFrames] = np.nan
            h, w, l = image_data0.shape
            # convert MJy/str to electron count
            convfact = hdu_list[0].header['GAIN']*hdu_list[0].header['EXPTIME']/hdu_list[0].header['FLUXCONV']
            image_data1 = convfact*image_data0
            # sigma clip
            fname = fnames[i]
            image_data2, tossed, badframetable = sigma_clipping(image_data1, i ,fname[fname.find('/bcd/')+5:], 
                                                                badframetable=badframetable, tossed=tossed)
            
            if maskStars != []:
                # Mask any other stars in the frame to avoid them influencing the background subtraction
                hdu_list[0].header['CTYPE3'] = 'Time-SIP' #Just need to add a type so astropy doesn't complain
                w = WCS(hdu_list[0].header, naxis=[1,2])
                mask = image_data2.mask
                for st in maskStars:
                    coord = SkyCoord(st[0], st[1])
                    x,y = np.rint(skycoord_to_pixel(coord, w)).astype(int)
                    x = x+np.arange(-1,2)
                    y = y+np.arange(-1,2)
                    x,y = np.meshgrid(x,y)
                    mask[x,y] = True
                image_data2 = np.ma.masked_array(image_data2, mask=mask)
            
            # bg subtract
            image_data3, bg_flux, bg_err = bgsubtract(image_data2, bg_flux, bg_err)
            # oversampling
            if oversamp:
                if reuse_oversamp:
                    savename = savepath + 'Oversampled/' + fnames[i].split('/')[-1].split('_')[-4] + '.pkl'
                    if os.path.isfile(savename):
                        image_data3 = np.load(savename)
                    else:
                        print('Warning: Oversampled images were not previously saved! Making new ones now...')
                        image_data3 = np.ma.masked_invalid(oversampling(image_data3))
                        if (saveoversamp == True):
                            # THIS CHANGES FROM ONE SET OF DATA TO ANOTHER!!!
                            image_data3.dump(savename)
                else:
                    image_data3 = np.ma.masked_invalid(oversampling(image_data3))
                    
                if saveoversamp:
                    # THIS CHANGES FROM ONE SET OF DATA TO ANOTHER!!!
                    savename = savepath + 'Oversampled/' + fnames[i].split('/')[-1].split('_')[-4] + '.pkl'
                    image_data3.dump(savename)
                
                # Aperture Photometry
                # get centroids & PSF width
                xo, yo, xw, yw = centroid_FWM(image_data3, xo, yo, xw, yw, scale = 2)
                
                # aperture photometry
                if moveCentroid:
                    xo_new = np.array(xo[image_data3.shape[0]*i:])
                    yo_new = np.array(yo[image_data3.shape[0]*i:])
                    xo_new[np.where(np.isnan(xo_new))[0]] = 15*2
                    yo_new[np.where(np.isnan(yo_new))[0]] = 15*2
                    xo_new = list(xo_new)
                    yo_new = list(yo_new)
                    flux, flux_err = A_photometry(image_data3, bg_err[-h:], flux, flux_err,
                                                  cx=xo, cy=yo, r=2*r, a=2*5, b=2*5, w_r=2*5, h_r=2*5,
                                                  shape=shape, method=method)
                else:
                    flux, flux_err = A_photometry(image_data3, bg_err[-h:], flux, flux_err,
                                                  cx=2*15, cy=2*15, r=2*r, a=2*5, b=2*5, w_r=2*5, h_r=2*5,
                                                  shape=shape, method=method)
            else :
                # get centroids & PSF width
                xo, yo, xw, yw = centroid_FWM(image_data3, xo, yo, xw, yw)
                # aperture photometry
                if moveCentroid:
                    xo_new = np.array(xo[image_data3.shape[0]*i:])
                    yo_new = np.array(yo[image_data3.shape[0]*i:])
                    xo_new[np.where(np.isnan(xo_new))[0]] = 15
                    yo_new[np.where(np.isnan(yo_new))[0]] = 15
                    xo_new = list(xo_new)
                    yo_new = list(yo_new)
                    flux, flux_err = A_photometry(image_data3, bg_err[-h:], flux, flux_err,
                                                  cx=xo_new, cy=yo_new, r=r, shape=shape, method=method)
                else:
                    flux, flux_err = A_photometry(image_data3, bg_err[-h:], flux, flux_err,
                                                  r=r, shape=shape, method=method)

    else:
        # FIX: The full frame versions of the code will go here
        pass

    if bin_data:
        binned_flux, binned_flux_std = binning_data(np.asarray(flux), bin_size)
        binned_time, binned_time_std = binning_data(np.asarray(time), bin_size)
        binned_xo, binned_xo_std     = binning_data(np.asarray(xo), bin_size)
        binned_yo, binned_yo_std     = binning_data(np.asarray(yo), bin_size)
        binned_xw, binned_xw_std     = binning_data(np.asarray(xw), bin_size)
        binned_yw, binned_yw_std     = binning_data(np.asarray(yw), bin_size)
        binned_bg, binned_bg_std     = binning_data(np.asarray(bg_flux), bin_size)
        binned_bg_err, binned_bg_err_std = binning_data(np.asarray(bg_err), bin_size)

        #sigma clip binned data to remove wildly unacceptable data
        try:
            binned_flux_mask = sigma_clip(binned_flux, sigma=10, maxiters=2)
        except TypeError:
            binned_flux_mask = sigma_clip(binned_flux, sigma=10, iters=2)
        if np.ma.is_masked(binned_flux_mask):
            binned_time[binned_flux_mask==binned_flux] = np.nan
            binned_time_std[binned_flux_mask==binned_flux] = np.nan
            binned_xo[binned_flux_mask==binned_flux] = np.nan
            binned_xo_std[binned_flux_mask==binned_flux] = np.nan
            binned_yo[binned_flux_mask==binned_flux] = np.nan
            binned_yo_std[binned_flux_mask==binned_flux] = np.nan
            binned_xw[binned_flux_mask==binned_flux] = np.nan
            binned_xw_std[binned_flux_mask==binned_flux] = np.nan
            binned_yw[binned_flux_mask==binned_flux] = np.nan
            binned_yw_std[binned_flux_mask==binned_flux] = np.nan
            binned_bg[binned_flux_mask==binned_flux] = np.nan
            binned_bg_std[binned_flux_mask==binned_flux] = np.nan
            binned_bg_err[binned_flux_mask==binned_flux] = np.nan
            binned_bg_err_std[binned_flux_mask==binned_flux] = np.nan
            binned_flux_std[binned_flux_mask==binned_flux] = np.nan
            binned_flux[binned_flux_mask==binned_flux] = np.nan

    if plot:
        if bin_data:
            plotx = binned_time
            ploty0 = binned_flux
            ploty1 = binned_xo
            ploty2 = binned_yo
        else:
            plotx = time
            ploty0 = flux
            ploty1 = xo
            ploty2 = yo
            
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15,5))
        fig.suptitle(planet, fontsize="x-large")
        
        axes[0].plot(plotx, ploty0,'k+', color='black')
        axes[0].set_ylabel("Stellar Flux (electrons)")

        axes[1].plot(plotx, ploty1, '+', color='black')
        axes[1].set_ylabel("$x_0$")

        axes[2].plot(plotx, ploty2, 'r+', color='black')
        axes[2].set_xlabel("Time (BMJD))")
        axes[2].set_ylabel("$y_0$")
        fig.subplots_adjust(hspace=0)
        axes[2].ticklabel_format(useOffset=False)
        
        if save:
            pathplot = savepath + plot_name
            fig.savefig(pathplot)
        
        plt.show()
        plt.close()

    if save:
        FULL_data = np.c_[flux, flux_err, time, xo, yo, xw, yw, bg_flux, bg_err]
        FULL_head = 'Flux, Flux Uncertainty, Time, x-centroid, y-centroid, x-PSF width, y-PSF width, bg flux, bg flux err'
        pathFULL  = savepath + save_full
        np.savetxt(pathFULL, FULL_data, header = FULL_head)
        if bin_data:
            BINN_data = np.c_[binned_flux, binned_flux_std, binned_time, binned_time_std, binned_xo, binned_xo_std, binned_yo,
                              binned_yo_std, binned_xw, binned_xw_std, binned_yw, binned_yw_std,  binned_bg, binned_bg_std,
                              binned_bg_err, binned_bg_err_std]
            BINN_head = 'Flux, Flux std, Time, Time std, x-centroid, x-centroid std, y-centroid, y-centroid std, x-PSF width, x-PSF width std, y-PSF width, y-PSF width std, bg flux, bg flux std, bg flux err, bg flux err std]'
            pathBINN  = savepath + save_bin
            np.savetxt(pathBINN, BINN_data, header = BINN_head)

    return




import unittest

class TestAperturehotometryMethods(unittest.TestCase):

    # Test that centroiding gives the expected values and doesn't swap x and y
    def test_FWM_centroiding(self):
        fake_images = np.zeros((4,32,32))
        for i in range(fake_images.shape[0]):
            fake_images[i,14+i,15] = 2
        xo, yo, _, _ = centroid_FWM(fake_images)
        self.assertTrue(np.all(xo==np.ones_like(xo)*15.))
        self.assertTrue(np.all(yo==np.arange(14,18)))

    # Test that circular aperture photometry properly follows the input centroids and gives the expected values
    def test_circularAperture(self):
        fake_images = np.zeros((4,32,32))
        for i in range(fake_images.shape[0]):
            fake_images[i,14+i,15] = 2
        xo = np.ones(fake_images.shape[0])*15
        yo = np.arange(14,18)
        flux, _ = A_photometry(fake_images, np.zeros_like(xo), cx=xo, cy=yo, r=1.,
                               shape='Circular', method='center')
        self.assertTrue(np.all(flux==np.ones_like(flux)*2.))

if __name__ == '__main__':
    unittest.main()
