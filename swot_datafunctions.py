# Import libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime

# For signal analysis
import scipy
from scipy.fft import fft, ifft, fftfreq


# ========================================================================
# Determine gradient
def gradient(X, lon, lat, dist):
    # Define shape
    N_y, N_x = np.shape(X)

    # Return gradients
    grad_y, grad_x = np.gradient(X)

    return grad_x, grad_y, lon, lat
# ========================================================================


# ========================================================================
# 2D filter data
def filter2d(X, filterKM, gridKM, type="boxcar"):

    match type:
        case "boxcar":
            # Do a 2D-boxcar filtering process
            filter_size = int(filterKM/gridKM)
            kern = np.ones(filter_size)[:,np.newaxis] @ np.ones(filter_size)[:,np.newaxis].T

        case "gaussian":
            # Do a 2D-gaussian filtering process
            filter_sd = 0.5*filterKM / 1.96
            filter_size = int(filter_sd * gridKM*4)
            kern = scipy.signal.windows.gaussian(filter_size, filter_sd)[:,np.newaxis] @ scipy.signal.windows.gaussian(filter_size, filter_sd)[:,np.newaxis].T
        
    x_long = scipy.signal.convolve2d(X, kern, boundary="symm", mode="same") / kern.sum()
            
    return x_long
# ========================================================================


# ========================================================================
# Determine gradients from KaRIn data
def karinGradients(ssha, lon, lat, heading, gradient_dist, gradient_dist_km):

    N, M = ssha.shape

    # Gradient Cross and Along-track
    grad_cross, grad_along, lon, lat = gradient(ssha, lon, lat, gradient_dist)
    grad_cross /= gradient_dist_km # Convert unit to m / km
    grad_along /= gradient_dist_km # Convert unit to m / km
    margins = [gradient_dist//2, int(np.ceil(gradient_dist/2))]

    # Convert gradients to N and E coordinates
    orbit_angle = np.deg2rad(heading)
    orbit_angle = np.repeat(orbit_angle[:,np.newaxis], axis=1, repeats=M)

    # Correct direction
    heading_mean = np.nanmedian(heading)
    # Angle with respect to true north of the horizontal component of the spacecraft Earth-relative velocity vector.
    # Values between 0 and 90 deg indicate that the velocity vector has a northward component, 
    # and values between 90 and 180 deg indicate that the velocity vector has a southward component.

    if heading_mean > 90: # Southward
        correction_angle = np.deg2rad(180) - orbit_angle
        # Along track
        grad_along_x = grad_along*np.sin(correction_angle)
        grad_along_y = -grad_along*np.cos(correction_angle)

        # Cross track
        grad_cross_x = -grad_cross*np.cos(correction_angle)
        grad_cross_y = -grad_cross*np.sin(correction_angle)

    else:                 # Northward
        # Along track
        grad_along_x = grad_along*np.sin(orbit_angle)
        grad_along_y = grad_along*np.cos(orbit_angle)

        # Cross track
        grad_cross_x = grad_cross*np.cos(orbit_angle)
        grad_cross_y = -grad_cross*np.sin(orbit_angle)


    
    
    grad_x = grad_cross_x + grad_along_x
    grad_y = grad_cross_y + grad_along_y

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Surface Geostrophic current
    g = 9.82
    Omega = 7.292e-5 # s^-1
    f = 2*Omega * np.sin(np.deg2rad(lat))

    geo_u = -g/f * (grad_y/1000) # convert gradient to m/m
    geo_v = g/f * (grad_x/1000) # convert gradient to m/m
    geo_mag = np.sqrt(geo_u**2 + geo_v**2)

    return lon, lat, grad_x, grad_y, grad_mag, geo_u, geo_v, geo_mag
# ========================================================================



# ========================================================================
# Gradient analysis of KaRIn data
def karin_filtering(datadict, aoi=[-180,180,-90,90], FILTER_SIZE_KM = 10, GRID_SIZE_KM = 2, type = "boxcar"):
    filter_size = int(FILTER_SIZE_KM/GRID_SIZE_KM)

    for c in datadict.keys():
        for p in datadict[c].keys():

            # Get segment for computation
            N, M = datadict[c][p]["latitude"].shape
            lat_mask = np.logical_and(datadict[c][p]["latitude"][:,M//2]>aoi[2], datadict[c][p]["latitude"][:,M//2]<aoi[3])
            lat = np.copy(datadict[c][p]["latitude"][lat_mask,:])
            lon = np.copy(datadict[c][p]["longitude"][lat_mask,:])
            heading = datadict[c][p]["velocity_heading"][lat_mask]

            # MSS
            for basis in ["ellip","geoid", "mss"]:

                if type == None:
                    lamb_types = ["all"]
                else:
                    lamb_types = ["short", "long", "all"]

                for lamb in lamb_types:

                    ssh = np.copy(datadict[c][p]["ssh_"+basis][lat_mask,:])

                    ssh_long = filter2d(ssh, FILTER_SIZE_KM, GRID_SIZE_KM, type=type)
                    ssh_short = ssh - ssh_long

                    if lamb == "short":
                        ssh_filt = ssh_short
                        grad_size_km = GRID_SIZE_KM
                        grad_size = int(grad_size_km/GRID_SIZE_KM)
                    elif lamb == "long":
                        ssh_filt = ssh_long
                        grad_size_km = FILTER_SIZE_KM
                        grad_size = int(grad_size_km/GRID_SIZE_KM)
                    elif lamb == "all":
                        ssh_filt = ssh
                        grad_size_km = GRID_SIZE_KM
                        grad_size = int(grad_size_km/GRID_SIZE_KM)

                    # Add subfields
                    keyname = "ssh_"+basis+"_"+lamb
                    datadict[c][p][keyname] = {}
                    datadict[c][p][keyname] = {}

                    gradient_dist = int(grad_size_km/GRID_SIZE_KM) # distance between grid cells for determening the gradient
                    grad_lon, grad_lat, grad_x, grad_y, grad_mag, geo_u, geo_v, geo_mag = karinGradients(ssh_filt, lon, lat, heading, grad_size, grad_size_km)
                    
                    # Save data
                    datadict[c][p][keyname]["ssh"] = ssh_filt
                    datadict[c][p][keyname]["lon"] = lon
                    datadict[c][p][keyname]["lat"] = lat

                    datadict[c][p][keyname]["g_lon"] = grad_lon
                    datadict[c][p][keyname]["g_lat"] = grad_lat
                    datadict[c][p][keyname]["g_x"] = grad_x
                    datadict[c][p][keyname]["g_y"] = grad_y
                    datadict[c][p][keyname]["g_mag"] = grad_mag
                    
                    datadict[c][p][keyname]["g_u"] = geo_u
                    datadict[c][p][keyname]["g_v"] = geo_v
                    datadict[c][p][keyname]["g_uv_mag"] = geo_mag
# ========================================================================


# ========================================================================
# Compute the PSD (power spectral density)
def powerSpectralDensity(ssh, sampling, unit="m", tapering_f = "boxcar", tapering=8):
    """
    # Setup
    N = ssh.size
    T = sampling * N # Sampling Time

    # Interpolate gaps
    mask = np.isnan(ssh)
    ssh[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ssh[~mask])
    
    # Remove ssh mean
    ssh_mean = np.mean(ssh)
    ssh = ssh - ssh_mean

    # Detrend
    fit = scipy.stats.linregress(np.arange(ssh.size),ssh)
    ssh = ssh+ssh*(-0.5*fit.slope)*np.arange(ssh.size)

    # Window tapering function
    match tapering_f:
        case "cosine":
            win_size = tapering
            window = scipy.signal.windows.cosine(win_size*2)
            tapering = np.ones(ssh.shape[0])
            tapering[:win_size] *= window[:win_size]
            tapering[-win_size:] *= window[-win_size:]
        case "tukey":
            win_size = tapering
            window = scipy.signal.windows.tukey(ssh.size, alpha=2*win_size/ssh.size)
            tapering = window
        case None:
            tapering = np.ones(ssh.shape[0])
    ssh *= tapering

    # Perform FFT
    N = ssh.size
    SSH = fft(ssh)[:N//2]
    wavenumber = fftfreq(N, sampling)[:N//2]

    # Perform unit correction
    match unit:
        case "m":
            SSH *= 1
        case "cm":
            SSH *= 100
        case "mm":
            SSH *= 1000
                
    # Calculate power spectral density
    psd = 2 * (sampling)/(N+0.5)* ((abs(SSH))**2)

    # Remove first element (is inf)
    wavenumber = wavenumber[1:]
    psd = psd[1:]
    
    # Convert units
    psd /= 1000 # Convert from [unit]^2/cpm to [unit]^2/cpkm
    wavenumber *= 1000 # Convert from cpm to cpkm

    return psd, wavenumber
    """
    
    # Setup
    N = ssh.size

    # Interpolate gaps
    mask = np.isnan(ssh)
    ssh[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ssh[~mask])

    # Detrend
    ssh = scipy.signal.detrend(ssh, axis=0, type='linear')

    match tapering_f:
        case "boxcar":
            kernel = scipy.signal.windows.boxcar(N)
        case "tukey":
            kernel = scipy.signal.windows.tukey(N, alpha=0.1)
        case "hann":
            kernel = scipy.signal.windows.hann(N)
    ssh *= kernel

    # Perform FFT
    SSH = fft(ssh)[:N//2]
    k = fftfreq(N, sampling)[:N//2]
                
    # Calculate power spectral density
    kernel_norm = kernel.sum()/N
    psd = 2 * (sampling)/(N*kernel_norm)* ((abs(SSH))**2)
    
    # Perform unit correction
    match unit:
        case "m":
            psd *= 1
        case "cm":
            psd *= 1e4

    # Convert units
    psd /= 1000 # Convert from [unit]^2/cpm to [unit]^2/cpkm
    k *= 1000 # Convert from cpm to cpkm

    # Remove first element (is inf)
    k = k[1:]
    psd = psd[1:]
    
    return psd, k
# ========================================================================

# ========================================================================
# Compute the periodogram of a matrix X
# powerSpectralDensity(ssh, sampling, unit="m", tapering_f = None, tapering=8):
def periodogram(X, axis=0, conf_lvl=68, sampling=2000, unit="m", tapering_f = None, tapering=8, stacked_data=False, skip_n=1):
    
    if stacked_data == True:
        N, M, Y = X.shape
    else:
        N, M = X.shape

    # Setup axis
    if stacked_data == True:
        
        # Determine nans
        if axis == 0:
            nan_counts = np.sum(np.isnan(X[:,:,0]), axis=axis) / N
        elif axis == 1:
            nan_counts = np.sum(np.isnan(X[:,:,0]), axis=axis) / M
        first_id_nonnan = np.where(nan_counts<1)[0][0]
        
        if axis == 0:
            x_profile = X[:,first_id_nonnan,0]
        elif axis == 1:
            x_profile = X[first_id_nonnan,:,0]
    else:
        # Determine nans
        if axis == 0:
            nan_counts = np.sum(np.isnan(X[:,:]), axis=axis) / N
        elif axis == 1:
            nan_counts = np.sum(np.isnan(X[:,:]), axis=axis) / M
        first_id_nonnan = np.where(nan_counts<1)[0][0]

        if axis == 0:
            x_profile = X[:,first_id_nonnan]
        elif axis == 1:
            x_profile = X[first_id_nonnan,:]

    psd_, wavenumbers = powerSpectralDensity(x_profile, sampling=sampling, unit=unit, 
                                             tapering_f=tapering_f, tapering=tapering)
    psd_stack = np.zeros(psd_.size)
    
    if stacked_data == True:
        for stack in range(Y):
            if axis == 0:
                for s in range(0,M,skip_n):
                    x_profile = X[:,s,stack]

                    # check if empty segment
                    if (~np.isnan(x_profile)).sum() == 0:
                        continue
                        
                    psd_, wavenumber = powerSpectralDensity(x_profile, sampling=sampling, unit=unit, 
                                                            tapering_f=tapering_f, tapering=tapering)
                    
                    # Add to list
                    psd_ = np.interp(wavenumbers, wavenumber, psd_)
                    psd_stack = np.c_[psd_stack, psd_]
            elif axis == 1:
                for s in range(0,N,skip_n):
                    x_profile = X[s,:,stack]

                    # check if empty segment
                    if (~np.isnan(x_profile)).sum() == 0:
                        continue
                        
                    psd_, wavenumber = powerSpectralDensity(x_profile, sampling=sampling, unit=unit, 
                                                            tapering_f=tapering_f, tapering=tapering)
                    
                    # Add to list
                    psd_ = np.interp(wavenumbers, wavenumber, psd_)
                    psd_stack = np.c_[psd_stack, psd_]

    else:
        if axis == 0:
            for s in range(0,M,skip_n):
                x_profile = X[:,s]

                # check if empty segment
                if (~np.isnan(x_profile)).sum() == 0:
                    continue
                    
                psd_, wavenumber = powerSpectralDensity(x_profile, sampling=sampling, unit=unit, 
                                                        tapering_f=tapering_f, tapering=tapering)
                
                # Add to list
                psd_ = np.interp(wavenumbers, wavenumber, psd_)
                psd_stack = np.c_[psd_stack, psd_]
        elif axis == 1:
            for s in range(0,N,skip_n):
                x_profile = X[s,:]

                # check if empty segment
                if (~np.isnan(x_profile)).sum() == 0:
                    continue
                    
                psd_, wavenumber = powerSpectralDensity(x_profile, sampling=sampling, unit=unit, 
                                                        tapering_f=tapering_f, tapering=tapering)
                
                # Add to list
                psd_ = np.interp(wavenumbers, wavenumber, psd_)
                psd_stack = np.c_[psd_stack, psd_]

    psd_stack = psd_stack[:,1:] # remove the zeros from initializing the list

    psd = np.median(psd_stack,axis=1)
    
    cd_interval = np.percentile(psd_stack,[100-conf_lvl,conf_lvl],axis=1)


    return psd, wavenumber, cd_interval
# ========================================================================
