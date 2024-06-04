### Functions required to run python cm removal script
# Date 6/3/2024
# Based on march 24 file

import math
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import hvplot.pandas
import sys
sys.path.append('/efs/SHIFT-Python-Utilities/')
from shift_python_utilities.intake_shift import shift_catalog
import rioxarray as rxr
import rasterio as rio
import geopandas as gpd
from shapely.geometry import Polygon
from functools import partial

import holoviews as hv
hv.extension('bokeh')
import hvplot.xarray
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.integrate import simps
import warnings
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

warnings.filterwarnings("ignore")


# Purpose: Identify and compute the continuum of the spectrum using convex hull
# Output: the function returns the values of the continuum for each wavelength, 
# appoximates the highest or outermost boundaries of the spectral data by providing a background curve
def get_continuum_line(points):
    # Unpack points into x (wavelengths) and y reflectance
    # assume points is a 2D array where each row is a point
    x, y = points.T
    # Add to the points to ensure that the convex hull includes all spectrum edges
    augmented = np.concatenate([points, [(x[0], np.min(y)-1), (x[-1], np.min(y)-1)]], axis=0)
    hull = ConvexHull(augmented)
    # extract the points based on the convex hull vertices that correspond to the actual data points, not agumented ones
    continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
  
    continuum_function = interp1d(*continuum_points.T, fill_value="extrapolate")

    return continuum_function(x)

# Apply continuum removal. The result is a spectrum that has been correced for basline
# Purpose: Remove the influence of the continuum from spectral data making the features of interest like absorption bands more pronounced
def continuum_removal(points, show=False):
    # Extract wavelengths (x) and reflectance (y) from points 
    x, y = points.T
    
    augmented = np.concatenate([points, [(x[0], np.min(y)-1), (x[-1], np.min(y)-1)]], axis=0)
    hull = ConvexHull(augmented)
    continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
    continuum_function = interp1d(*continuum_points.T, fill_value="extrapolate")
    
    # Normalize the original reflectance values by the continuum to highlight absorption features.
    yprime = y / continuum_function(x)

    if show:
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(x, y, label='Data')
        axes[0].plot(*continuum_points.T, label='Continuum')
        axes[0].legend()
        axes[1].plot(x, yprime, label='Data / Continuum')
        axes[1].legend()
    # return np.c_[x, yprime]
    return yprime

# We also added a function that takes the reflectance values with continuum removed and wavelengths to compute the area.
# Scaling is related to the scale of the wavelengths.
def get_area(continuum_removed, wavelengths):
    return simps(1-continuum_removed,wavelengths) ## why are we multiplying by 1000??? 

def get_depth(values):
    return 1- np.min(values)

def get_depth_wvl(values, wavelengths):
    min_index_array = np.where(values == np.min(values))[0]
    wvl_pos = wavelengths[min_index_array]
    return wvl_pos

# Linear interpolation to find exact WL and WR
def interpolate_wavelength(wavelengths, index1, index2, value1, value2, half_max):
    return wavelengths[index1] + ((half_max - value1) / (value2 - value1)) * (wavelengths[index2] - wavelengths[index1])

def get_width(values, wavelengths):
    # Calculate half-maximum value
    half_max = 1.0 - (get_depth(values) / 2)

    # Find the left and right indices where the spectral curve crosses the half-maximum value
    left_index = np.where(values <= half_max)[0][0]
    right_index = np.where(values <= half_max)[0][-1]

    WL = interpolate_wavelength(wavelengths, left_index - 1, left_index, values[left_index - 1], values[left_index], half_max)
    WR = interpolate_wavelength(wavelengths, right_index, right_index + 1, values[right_index], values[right_index + 1], half_max)

    # Calculate FWHM
    FWHM = WR - WL
    return FWHM, WL, WR



#### ACTUALLY applying the functions 
# This function takes the reflectances and wavelengths, check if the reflectances of the pixel are none.
# If nan, return none to avoid errors, if there is no Nan, remove the continuum and then computes the area.
## Apply the actual functions  functions 
def calc_continuum_area(reflectances, wavelengths):
    if np.isnan(reflectances).any():
        return np.nan
    
    try:
        points = np.c_[wavelengths, reflectances]
        removed_continuum = continuum_removal(points)
        return get_area(removed_continuum, wavelengths)
    except Exception:
        return np.nan
    

# remove the continuumn and return the continuum removed spectra
def continuum_remove_spec(reflectances, wavelengths):
    if np.isnan(reflectances).any():
        return np.nan

    try:
        points = np.c_[wavelengths, reflectances]
        removed_continuum = continuum_removal(points)
        return removed_continuum
    except Exception:
        return np.nan

# retrieve the continuum line itself 
def retrieve_continuum_line(reflectances, wavelengths):
    if np.isnan(reflectances).any():
        return np.nan

    try:
        points = np.c_[wavelengths, reflectances]
        continuum_line = get_continuum_line(points)
        return continuum_line
    except Exception:
        return np.nan