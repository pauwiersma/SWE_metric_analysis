"""
Module containing a class for calculating SWE (Snow Water Equivalent) metrics and signatures.

This class provides methods for analyzing snow-related metrics including:
- SWE maximum values and timing
- Snow water storage calculations
- Melt rates and timing
- Performance metrics (KGE, NSE) for snow simulations
- Elevation-based analyses

The methods are designed to work with both time series (1D) and spatial (3D) data.
"""

import numpy as np
import xarray as xr
from typing import Union, Tuple, Optional
import pandas as pd
import HydroErr as he
from scipy.stats import pearsonr, variation

class SWEMetrics:
    def __init__(self, dem=None, elev_bands=None):
        """
        Initialize SWEMetrics class.
        
        Args:
            dem (xr.DataArray, optional): Digital elevation model data
            elev_bands (np.ndarray, optional): Elevation band boundaries
        """
        self.dem = dem
        self.elev_bands = elev_bands
        if dem is not None and elev_bands is None:
            self.elev_bands = self.calc_elev_bands(N_bands = 10)

    def calc_sum2d(self, swe3d):    
        """
        Calculate the sum of SWE over latitude and longitude, normalized by the number of valid cells.
        """
        sum2d = swe3d.sum(dim=['lat', 'lon'])
        cell_count = np.sum(~np.isnan(self.dem))
        sum2d = sum2d / cell_count
        return sum2d
    def calc_t_swe_max(self, swe_t):
        """
        Calculate the time of maximum SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        t_swe_max = np.nanargmax(swe_t)
        return t_swe_max if not np.isnan(t_swe_max) else np.nan

    def calc_t_swe_max_vs_elevation(self, swe3d):
        """
        Calculate the time of maximum SWE for different elevation bands.
        """
        t_swe_max = xr.apply_ufunc(self.calc_t_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
        t_swe_max_vs_elevation = self.calc_var_vs_elevation(t_swe_max)
        return t_swe_max_vs_elevation

    def calc_t_swe_max_catchment(self, swe3d):
        """
        Calculate the time of maximum SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        t_swe_max = self.calc_t_swe_max(swe_t)
        return t_swe_max

    def calc_t_swe_max_grid(self, swe3d):
        """
        Calculate the time of maximum SWE for each grid cell.
        """
        t_swe_max_grid = xr.apply_ufunc(self.calc_t_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
        return t_swe_max_grid
    
    # Maximum SWE
    def calc_swe_max(self, swe_t):
        """
        Calculate the maximum SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        SWE_max = np.nanmax(swe_t)
        return SWE_max
    
    
    def calc_swe_max_catchment(self, swe3d):
        """
        Calculate the maximum SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        SWE_max = self.calc_swe_max(swe_t)
        return SWE_max
    
    def calc_swe_max_vs_elevation(self, swe3d):
        """
        Calculate the maximum SWE for different elevation bands.
        """
        SWE_max = xr.apply_ufunc(self.calc_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
        SWE_max_vs_elevation = self.calc_var_vs_elevation(SWE_max)
        return SWE_max_vs_elevation

    def calc_swe_max_grid(self, swe3d):
        """
        Calculate the maximum SWE for each grid cell.
        """
        SWE_max_grid = xr.apply_ufunc(self.calc_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
        return SWE_max_grid

    # Snow Water Storage (SWS)
    def calc_sws(self, swe_t):
        """
        Calculate Snow Water Storage (SWS), the area under the SWE curve.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        sws = np.nansum(swe_t)
        return sws

    def calc_sws_vs_elevation(self, swe3d):
        """
        Calculate Snow Water Storage (SWS) for different elevation bands.
        """
        sws = xr.apply_ufunc(self.calc_sws, swe3d, input_core_dims=[['time']], vectorize=True)
        sws_vs_elevation = self.calc_var_vs_elevation(sws)
        return sws_vs_elevation

    def calc_sws_catchment(self, swe3d):
        """
        Calculate the total Snow Water Storage (SWS) for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        sws = self.calc_sws(swe_t)
        return sws

    def calc_sws_grid(self, swe3d):
        """
        Calculate Snow Water Storage (SWS) for each grid cell.
        """
        sws_grid = xr.apply_ufunc(self.calc_sws, swe3d, input_core_dims=[['time']], vectorize=True)
        return sws_grid

    # Melt Rate
    def calc_melt_rate(self, swe_t):
        """
        Calculate the mean melt rate of SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        diff = np.diff(swe_t)
        neg_diff = diff[diff < 0]
        mean_melt_rate = np.mean(neg_diff) * -1
        return mean_melt_rate

    def calc_melt_rate_vs_elevation(self, swe3d):
        """
        Calculate the mean melt rate of SWE for different elevation bands.
        """
        melt_rate = xr.apply_ufunc(self.calc_melt_rate, swe3d, input_core_dims=[['time']], vectorize=True)
        melt_rate_vs_elevation = self.calc_var_vs_elevation(melt_rate)
        return melt_rate_vs_elevation

    def calc_melt_rate_catchment(self, swe3d):
        """
        Calculate the mean melt rate of SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        melt_rate = self.calc_melt_rate(swe_t)
        return melt_rate

    def calc_melt_rate_grid(self, swe3d):
        """
        Calculate the mean melt rate of SWE for each grid cell.
        """
        melt_rate_grid = xr.apply_ufunc(self.calc_melt_rate, swe3d, input_core_dims=[['time']], vectorize=True)
        return melt_rate_grid

    # 7-Day Melt Rate
    def calc_7_day_melt(self, swe_t):
        """
        Calculate the maximum 7-day melt rate of SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        diff = np.diff(swe_t)
        neg_diff = diff[diff < 0] * -1
        rol7 = pd.Series(neg_diff).rolling(window=7).max().max()
        return rol7

    def calc_7_day_melt_vs_elevation(self, swe3d):
        """
        Calculate the maximum 7-day melt rate of SWE for different elevation bands.
        """
        melt7 = xr.apply_ufunc(self.calc_7_day_melt, swe3d, input_core_dims=[['time']], vectorize=True)
        melt7_vs_elevation = self.calc_var_vs_elevation(melt7)
        return melt7_vs_elevation

    def calc_7_day_melt_catchment(self, swe3d):
        """
        Calculate the maximum 7-day melt rate of SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        melt7 = self.calc_7_day_melt(swe_t)
        return melt7

    def calc_7_day_melt_grid(self, swe3d):
        """
        Calculate the maximum 7-day melt rate of SWE for each grid cell.
        """
        melt7_grid = xr.apply_ufunc(self.calc_7_day_melt, swe3d, input_core_dims=[['time']], vectorize=True)
        return melt7_grid

    # Total SWE
    def calc_melt_sum(self, swe_t):
        """
        Calculate the total SWE.
        """
        diff = np.diff(swe_t)
        pos_swe = diff[diff < 0]
        melt_sum = np.nansum(pos_swe)*-1
        return melt_sum
    
    def calc_sf_sum(self, swe_t):
        """
        Calculate the total SWE.
        """
        diff = np.diff(swe_t)
        pos_swe = diff[diff > 0]
        melt_sum = np.nansum(pos_swe)
        return melt_sum

    def calc_melt_sum_vs_elevation(self, swe3d):
        """
        Calculate the total SWE for different elevation bands.
        """
        melt_sum = xr.apply_ufunc(self.calc_melt_sum, swe3d, input_core_dims=[['time']], vectorize=True)
        melt_sum = xr.where(melt_sum > 0, melt_sum, np.nan)
        melt_sum_vs_elevation = self.calc_var_vs_elevation(melt_sum)
        return melt_sum_vs_elevation

    def calc_melt_sum_catchment(self, swe3d):
        """
        Calculate the total SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        melt_sum = self.calc_melt_sum(swe_t)
        return melt_sum

    def calc_melt_sum_grid(self, swe3d):
        """
        Calculate the total SWE for each grid cell.
        """
        melt_sum_grid = xr.apply_ufunc(self.calc_melt_sum, swe3d, input_core_dims=[['time']], vectorize=True)
        melt_sum_grid = xr.where(swe3d.isnull().all(dim='time'), np.nan, melt_sum_grid)
        return melt_sum_grid

    # Start and End Dates of SWE

    def calc_swe_dates(self,swe_t, smooth_window=5, threshold_frac=0.1):
        """
        Identify start and end of SWE season based on smoothed SWE values and thresholding.

        Parameters:
        -----------
        swe_t : array-like
            Time series of SWE values
        smooth_window : int
            Window size for smoothing (default: 5)
        threshold_frac : float
            Fraction of max SWE to use as threshold (default: 0.1)

        Returns:
        --------
        list
            [start_index, end_index] of the main snow period, or [np.nan, np.nan]
        """
        import numpy as np
        import pandas as pd

        if swe_t is None or len(swe_t) < 3 or np.all(np.isnan(swe_t)):
            return [np.nan, np.nan]

        swe_t = np.array(swe_t)
        swe_max = np.nanmax(swe_t)
        if swe_max <= 0:
            return [np.nan, np.nan]

        # Smooth the SWE to reduce noise
        swe_smooth = pd.Series(swe_t).rolling(window=smooth_window, center=True, min_periods=1).mean()

        # Threshold-based mask
        threshold = threshold_frac * swe_max
        mask = swe_smooth > threshold

        # Find continuous segments above threshold
        segments = []
        in_segment = False
        for i, val in enumerate(mask):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                end = i - 1
                segments.append((start, end))
                in_segment = False
        if in_segment:
            segments.append((start, len(mask) - 1))

        if not segments:
            return [np.nan, np.nan]

        # Choose longest segment
        longest = max(segments, key=lambda x: x[1] - x[0])
        return [longest[0], longest[1]]

    # def calc_swe_dates(self, swe_t):
    #     """
    #     Calculate the start and end dates of SWE based on the 95th percentile of the maximum value.
        
    #     Parameters:
    #     -----------
    #     swe_t : array-like
    #         Time series of SWE values
        
    #     Returns:
    #     --------
    #     list
    #         [start_index, end_index] of the main snow period, or [np.nan, np.nan] if not identifiable
    #     """
    #     # Input validation
    #     if swe_t is None or len(swe_t) < 3:
    #         return [np.nan, np.nan]
        
    #     if np.all(np.isnan(swe_t)):
    #         return [np.nan, np.nan]
            
    #     # Calculate threshold based on 95th percentile of max
    #     swe_max = np.nanmax(swe_t)
    #     if swe_max <= 0:  # No real snow accumulation
    #         return [np.nan, np.nan]
            
    #     threshold = swe_max - np.nanquantile(swe_t, 0.95)
        
    #     # Get sign changes - these are potential crossing points
    #     above_threshold = swe_t > threshold
    #     sign_changes = np.where(np.diff(above_threshold))[0]
        
    #     # If no crossings, check if data starts above threshold
    #     if len(sign_changes) == 0:
    #         if above_threshold[0]:
    #             # Starts above threshold but never crosses down
    #             return [0, len(swe_t)-1]
    #         else:
    #             # Never crosses threshold
    #             return [np.nan, np.nan]
        
    #     # Handle case where data starts above threshold
    #     if above_threshold[0]:
    #         # First crossing is the end date
    #         if len(sign_changes) >= 1:
    #             return [0, sign_changes[0]]
    #         return [0, len(swe_t)-1]
        
    #     # Normal case - find the longest period above threshold
    #     if len(sign_changes) < 2:
    #         return [np.nan, np.nan]  # Need at least one complete period
        
    #     # Convert to start/end pairs
    #     periods = []
    #     for i in range(0, len(sign_changes), 2):
    #         if i+1 < len(sign_changes):
    #             start = sign_changes[i]
    #             end = sign_changes[i+1]
    #             periods.append((start, end, end-start))
        
    #     # If no complete periods found
    #     if not periods:
    #         return [np.nan, np.nan]
        
    #     # Find longest period
    #     longest_period = max(periods, key=lambda x: x[2])
        
    #     return [longest_period[0], longest_period[1]]
    # def calc_swe_dates(self, swe_t):
    #     """
    #     Calculate the start and end dates of SWE based on the 95th percentile.
    #     """
    #     q5 = np.nanmax(swe_t) - np.nanquantile(swe_t, 0.95)
    #     intercepts = np.where(np.diff(np.sign(swe_t - q5)))[0]
    #     if len(intercepts) == 1 and swe_t[0] > q5:
    #         return [0, intercepts[0]]
    #     if len(intercepts) > 2:
    #         distances = np.diff(intercepts)
    #         if len(distances) == 0:
    #             return [np.nan, np.nan]
    #         argmax = np.argmax(distances)
    #         intercepts = intercepts[argmax:argmax + 2]
    #     elif len(intercepts) < 2 or len(intercepts) != 2:
                
    #         return [np.nan, np.nan]
    #     return intercepts

    def calc_swe_start(self, swe_t):
        """
        Calculate the start date of SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        intercepts = self.calc_swe_dates(swe_t)
        return intercepts[0]

    def calc_swe_end(self, swe_t):
        """
        Calculate the end date of SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        intercepts = self.calc_swe_dates(swe_t)
        return intercepts[1]

    def calc_swe_start_vs_elevation(self, swe3d):
        """
        Calculate the start date of SWE for different elevation bands.
        """
        starts = xr.apply_ufunc(self.calc_swe_start, swe3d, input_core_dims=[['time']], vectorize=True, output_dtypes=[np.float64])
        starts_vs_elevation = self.calc_var_vs_elevation(starts)
        return starts_vs_elevation

    def calc_swe_end_vs_elevation(self, swe3d):
        """
        Calculate the end date of SWE for different elevation bands.
        """
        ends = xr.apply_ufunc(self.calc_swe_end, swe3d, input_core_dims=[['time']], vectorize=True, output_dtypes=[np.float64])
        ends_vs_elevation = self.calc_var_vs_elevation(ends)
        return ends_vs_elevation

    def calc_swe_start_end_catchment(self, swe3d):
        """
        Calculate the start and end dates of SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        intercepts = self.calc_swe_dates(swe_t)
        return intercepts

    def calc_swe_start_grid(self, swe3d):
        """
        Calculate the start date of SWE for each grid cell.
        """
        starts_grid = xr.apply_ufunc(self.calc_swe_start, swe3d, input_core_dims=[['time']], vectorize=True)
        return starts_grid

    def calc_swe_end_grid(self, swe3d):
        """
        Calculate the end date of SWE for each grid cell.
        """
        ends_grid = xr.apply_ufunc(self.calc_swe_end, swe3d, input_core_dims=[['time']], vectorize=True)
        return ends_grid

    # Kling-Gupta Efficiency (KGE)
    def calc_melt_kge(self,  swe_sim_t, swe_obs_t):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for melt rates.
        """
        dif_obs = np.diff(swe_obs_t)
        dif_sim = np.diff(swe_sim_t)
        neg_dif_obs = np.where(dif_obs < 0, dif_obs * -1, 0)
        neg_dif_sim = np.where(dif_sim < 0, dif_sim * -1, 0)
        kge = he.kge_2009(neg_dif_sim, neg_dif_obs)
        return kge

    def calc_melt_kge_vs_elevation(self, swe_sim, swe_obs):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for melt rates for different elevation bands.
        """
        kges = xr.apply_ufunc(self.calc_melt_kge, swe_sim, swe_obs, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64])
        kges_vs_elevation = self.calc_var_vs_elevation(kges)
        return kges_vs_elevation

    def calc_melt_kge_catchment(self, swe_sim, swe_obs):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for melt rates for the entire catchment.
        """
        swe_obs_t = self.calc_sum2d(swe_obs).to_pandas()
        swe_sim_t = self.calc_sum2d(swe_sim).to_pandas()
        kge = self.calc_melt_kge(swe_sim_t, swe_obs_t)
        return kge

    def calc_melt_kge_grid(self, swe_sim, swe_obs):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for melt rates for each grid cell.
        """
        kge_grid = xr.apply_ufunc(self.calc_melt_kge,  swe_sim,swe_obs, input_core_dims=[['time'], ['time']], vectorize=True)
        return kge_grid

    def calc_melt_nse(self, swe_sim_t, swe_obs_t):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates.
        """
        dif_obs = np.diff(swe_obs_t)
        dif_sim = np.diff(swe_sim_t)
        neg_dif_obs = np.where(dif_obs < 0, dif_obs * -1, 0)
        neg_dif_sim = np.where(dif_sim < 0, dif_sim * -1, 0)
        nse = he.nse(neg_dif_sim, neg_dif_obs)
        return nse
    
    def calc_melt_nse_vs_elevation(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates for different elevation bands.
        """
        nses = xr.apply_ufunc(self.calc_melt_nse, swe_sim, swe_obs, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64])
        nses_vs_elevation = self.calc_var_vs_elevation(nses)
        return nses_vs_elevation

    def calc_melt_nse_catchment(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates for the entire catchment.
        """
        swe_obs_t = self.calc_sum2d(swe_obs).to_pandas()
        swe_sim_t = self.calc_sum2d(swe_sim).to_pandas()
        nse = self.calc_melt_nse(swe_sim_t, swe_obs_t)
        return nse

    def calc_melt_nse_grid(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates for each grid cell.
        """
        nse_grid = xr.apply_ufunc(self.calc_melt_nse, swe_sim, swe_obs, input_core_dims=[['time'], ['time']], vectorize=True)
        return nse_grid
    def calc_snowfall_nse(self, swe_sim_t, swe_obs_t):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for snowfall.
        """
        dif_obs = np.diff(swe_obs_t)
        dif_sim = np.diff(swe_sim_t)
        pos_dif_obs = np.where(dif_obs > 0, dif_obs, 0)
        pos_dif_sim = np.where(dif_sim > 0, dif_sim, 0)
        nse = he.nse(pos_dif_sim, pos_dif_obs)
        return nse
    def calc_snowfall_nse_vs_elevation(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe
        Efficiency (NSE) for snowfall for different elevation bands.
        """
        nses = xr.apply_ufunc(self.calc_snowfall_nse, swe_sim, swe_obs, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64])
        nses_vs_elevation = self.calc_var_vs_elevation(nses)
        return nses_vs_elevation
    def calc_snowfall_nse_catchment(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe
        Efficiency (NSE) for snowfall for the entire catchment.
        """
        swe_obs_t = self.calc_sum2d(swe_obs).to_pandas()
        swe_sim_t = self.calc_sum2d(swe_sim).to_pandas()
        nse = self.calc_snowfall_nse(swe_sim_t, swe_obs_t)
        return nse
    def calc_snowfall_nse_grid(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe
        Efficiency (NSE) for snowfall for each grid cell.
        """
        nse_grid = xr.apply_ufunc(self.calc_snowfall_nse, swe_sim, swe_obs, input_core_dims=[['time'], ['time']], vectorize=True)
        return nse_grid

    # Nash-Sutcliffe Efficiency (NSE)
    def calc_swe_nse(self, swe_sim_t, swe_obs_t):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE.
        """
        nse = he.nse(swe_sim_t, swe_obs_t)
        return nse
    def calc_swe_rmse(self, swe_sim_t,swe_obs_t ):
        """
        Calculate the Root Mean Square Error (RMSE) for SWE.
        """
        rmse = he.rmse(swe_sim_t, swe_obs_t)
        return rmse
    def calc_swe_kge(self,swe_sim_t, swe_obs_t):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for SWE.
        """
        kge = he.kge_2009(swe_sim_t, swe_obs_t)
        return kge

    def calc_swe_nse_vs_elevation(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE for different elevation bands.
        """
        nses = xr.apply_ufunc(self.calc_swe_nse, swe_sim, swe_obs, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64])
        nses_vs_elevation = self.calc_var_vs_elevation(nses)
        return nses_vs_elevation

    def calc_swe_nse_catchment(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE for the entire catchment.
        """
        swe_obs_t = self.calc_sum2d(swe_obs).to_pandas()
        swe_sim_t = self.calc_sum2d(swe_sim).to_pandas()
        nse = self.calc_swe_nse(swe_sim_t, swe_obs_t)
        return nse

    def calc_swe_nse_grid(self, swe_sim, swe_obs):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE for each grid cell.
        """
        nse_grid = xr.apply_ufunc(self.calc_swe_nse, swe_sim, swe_obs, input_core_dims=[['time'], ['time']], vectorize=True)
        return nse_grid

    def calc_elev_bands(self, N_bands = 10):
        """
        Calculate elevation bands, making sure each bands has the same number of pixels.
        """
        dem = self.dem
        elev_flat = dem.values.flatten()
        elev_flat = pd.Series(elev_flat[~np.isnan(elev_flat)]).sort_values().reset_index(drop=True)
        quantiles = elev_flat.quantile(np.linspace(0, 1, N_bands + 1)).reset_index(drop=True)
        rounded_quantiles = np.round(quantiles, 0)
        self.elev_bands = rounded_quantiles
        return rounded_quantiles

    


    # Helper function to calculate variable vs elevation
    def calc_var_vs_elevation(self, var2d):
        """
        Calculate the mean of a variable in elevation bands defined by the calculated elevation bands.
        """
        dem = self.dem
        var2d_flat = var2d.values.flatten()
        dem_flat = dem.values.flatten()

        df = pd.DataFrame({'elevation': dem_flat, 'var': var2d_flat})
        df = df.dropna()
        # df['elevation_band'] = (df['elevation'] // 50) * 50
        # num_quantiles = 10  # You can adjust this number as needed
        # df['elevation_band'] = pd.qcut(df['elevation'], q=num_quantiles, labels=False)
        if not hasattr(self, 'elev_bands'):
            self.calc_elev_bands()
        elev_bands = self.elev_bands
        df['elevation_band'] = pd.cut(df['elevation'], bins=elev_bands, labels=False)

        var_by_band = (
            df.groupby('elevation_band')['var']
            .mean()
            .rename_axis('elevation_band')
            .reset_index(name='var')
            .set_index('elevation_band')
        )
        return var_by_band
        
    def calculate_spaef(self, swe_sim, swe_obs):
        """
        Calculate the SPAEF metric for assessing spatial patterns of SWE.
        Takes 3D data, turns it into 2D with melt_sum, and then calculates the SPAEF.

        Parameters:
        obs (xr.DataArray): Observed 3D data.
        sim (xr.DataArray): Simulated 3D data.

        Returns:
        float: SPAEF value.
        """
        # Ensure the input data are aligned
        # obs, sim = xr.align(obs, sim)

        obs = self.calc_melt_sum_grid(swe_obs)
        sim = self.calc_melt_sum_grid(swe_sim)

        # Flatten the data and remove NaNs
        obs_flat = obs.values.flatten()
        sim_flat = sim.values.flatten()
        mask = ~np.isnan(obs_flat) & ~np.isnan(sim_flat)
        obs_flat = obs_flat[mask]
        sim_flat = sim_flat[mask]

        # Calculate A: Pearson correlation coefficient
        A = pearsonr(obs_flat, sim_flat)[0]

        # Calculate B: Fraction of the coefficient of variation
        B = (variation(sim_flat) / variation(obs_flat))
        # Calculate C: Histogram intersection
        obs_hist, bin_edges = np.histogram(obs_flat, bins='auto', density=True)
        sim_hist, _ = np.histogram(sim_flat, bins=len(obs_hist), density=True)
        C = np.sum(np.minimum(obs_hist, sim_hist)) / np.sum(obs_hist)
        # C basically gives the total overlap of the two histogram. So if you then to C-1 you get the error

        # Calculate SPAEF
        SPAEF = 1 - np.sqrt((A - 1)**2 + (B - 1)**2 + (C - 1)**2)

        return SPAEF


    # def _calc_sum2d(self, swe3d):
    #     """
    #     Calculate the sum of SWE over latitude and longitude, normalized by the number of valid cells.
    #     Args:
    #         swe3d: 3D SWE data (time, lat, lon)
    #     Returns:
    #         Sum of SWE normalized by cell count
    #     """
    #     sum2d = swe3d.sum(dim=['lat', 'lon'])
    #     if self.dem is not None:
    #         cell_count = np.sum(~np.isnan(self.dem))
    #         sum2d = sum2d / cell_count
    #     return sum2d

    # def _calc_t_swe_max(self, swe_t):
    #     """
    #     Calculate the time index of maximum SWE.
    #     Args:
    #         swe_t: SWE time series
    #     Returns:
    #         Index of maximum SWE or np.nan
    #     """
    #     if np.all(np.isnan(swe_t)):
    #         return np.nan
    #     t_swe_max = np.nanargmax(swe_t)
    #     return t_swe_max if not np.isnan(t_swe_max) else np.nan

    # def _calc_swe_max(self, swe_t):
    #     """
    #     Calculate the maximum SWE.
    #     Args:
    #         swe_t: SWE time series
    #     Returns:
    #         Maximum SWE (or np.nan if all nan)
    #     """
    #     if np.all(np.isnan(swe_t)):
    #         return np.nan
    #     SWE_max = np.nanmax(swe_t)
    #     return SWE_max

    # def _calc_sws(self, swe_t):
    #     """
    #     Calculate Snow Water Storage (SWS), the area under the SWE curve.
    #     Args:
    #         swe_t: SWE time series
    #     Returns:
    #         SWS (or np.nan if all nan)
    #     """
    #     if np.all(np.isnan(swe_t)):
    #         return np.nan
    #     sws = np.nansum(swe_t)
    #     return sws

    # def _calc_melt_rate(self, swe_t):
    #     """
    #     Calculate the mean melt rate of SWE (negative diff averaged).
    #     Args:
    #         swe_t: SWE time series
    #     Returns:
    #         Mean melt rate (or np.nan if all nan)
    #     """
    #     if np.all(np.isnan(swe_t)):
    #         return np.nan
    #     diff = np.diff(swe_t)
    #     neg_diff = diff[diff < 0]
    #     mean_melt_rate = np.mean(neg_diff) * (-1) if len(neg_diff) > 0 else np.nan
    #     return mean_melt_rate

    # def _calc_7_day_melt(self, swe_t):
    #     """
    #     Calculate the maximum 7-day melt rate of SWE.
    #     Args:
    #         swe_t: SWE time series
    #     Returns:
    #         Maximum 7-day melt rate (or np.nan if all nan)
    #     """
    #     if np.all(np.isnan(swe_t)):
    #         return np.nan
    #     diff = np.diff(swe_t)
    #     neg_diff = diff[diff < 0] * (-1)
    #     rol7 = pd.Series(neg_diff).rolling(window=7, min_periods=1).max().max() if len(neg_diff) > 0 else np.nan
    #     return rol7

    # def _calc_melt_sum(self, swe_t):
    #     """
    #     Calculate the total melt sum (sum of negative diff).
    #     Args:
    #         swe_t: SWE time series
    #     Returns:
    #         Total melt sum
    #     """
    #     diff = np.diff(swe_t)
    #     pos_swe = diff[diff < 0]
    #     melt_sum = np.nansum(pos_swe) * (-1)
    #     return melt_sum

    # def _calc_sf_sum(self, swe_t):
    #     """
    #     Calculate the total snowfall sum (sum of positive diff).
    #     Args:
    #         swe_t: SWE time series
    #     Returns:
    #         Total snowfall sum
    #     """
    #     diff = np.diff(swe_t)
    #     pos_swe = diff[diff > 0]
    #     sf_sum = np.nansum(pos_swe)
    #     return sf_sum

    # def _calc_swe_dates(self, swe_t, smooth_window=5, threshold_frac=0.1):
    #     """
    #     Identify start and end of SWE season based on smoothed SWE values and thresholding.
    #     Parameters:
    #         swe_t: SWE time series (array-like)
    #         smooth_window (int): Window size for smoothing (default 5)
    #         threshold_frac (float): Fraction of max SWE to use as threshold (default 0.1)
    #     Returns:
    #         list: [start_index, end_index] of the main snow period, or [np.nan, np.nan] if not identifiable.
    #     """
    #     if swe_t is None or len(swe_t) < 3 or np.all(np.isnan(swe_t)):
    #         return [np.nan, np.nan]
    #     swe_t = np.array(swe_t)
    #     swe_max = np.nanmax(swe_t)
    #     if swe_max <= 0:
    #         return [np.nan, np.nan]
    #     swe_smooth = pd.Series(swe_t).rolling(window=smooth_window, center=True, min_periods=1).mean()
    #     threshold = threshold_frac * swe_max
    #     mask = swe_smooth > threshold
    #     segments = []
    #     in_segment = False
    #     for i, val in enumerate(mask):
    #         if val and not in_segment:
    #             start = i
    #             in_segment = True
    #         elif not val and in_segment:
    #             end = i - 1
    #             segments.append((start, end))
    #             in_segment = False
    #     if in_segment:
    #         segments.append((start, len(mask) - 1))
    #     if not segments:
    #         return [np.nan, np.nan]
    #     longest = max(segments, key=lambda x: x[1] - x[0])
    #     return [longest[0], longest[1]]

    # def _calc_elev_bands(self, dem, N_bands=10):
    #     """
    #     Calculate elevation bands, making sure each band has the same number of pixels.
        
    #     Args:
    #         dem (xr.DataArray): Digital elevation model data
    #         N_bands (int): Number of elevation bands (default 10)
        
    #     Returns:
    #         np.ndarray: Array of elevation band boundaries
    #     """
    #     elev_flat = dem.values.flatten()
    #     elev_flat = pd.Series(elev_flat[~np.isnan(elev_flat)]).sort_values().reset_index(drop=True)
    #     quantiles = elev_flat.quantile(np.linspace(0, 1, N_bands + 1)).reset_index(drop=True)
    #     rounded_quantiles = np.round(quantiles, 0)
    #     return rounded_quantiles

    # def _calc_var_vs_elevation(self, var2d):
    #     """
    #     Calculate the mean of a variable in elevation bands defined by the calculated elevation bands.
        
    #     Args:
    #         var2d (xr.DataArray): 2D variable to analyze
        
    #     Returns:
    #         pd.DataFrame: Mean values of the variable for each elevation band
    #     """
    #     if self.dem is None:
    #         raise ValueError("DEM must be provided to calculate elevation-based metrics")
            
    #     var2d_flat = var2d.values.flatten()
    #     dem_flat = self.dem.values.flatten()

    #     df = pd.DataFrame({'elevation': dem_flat, 'var': var2d_flat})
    #     df = df.dropna()
        
    #     if self.elev_bands is None:
    #         self.elev_bands = self._calc_elev_bands(self.dem)
        
    #     df['elevation_band'] = pd.cut(df['elevation'], bins=self.elev_bands, labels=False)

    #     var_by_band = (
    #         df.groupby('elevation_band')['var']
    #         .mean()
    #         .rename_axis('elevation_band')
    #         .reset_index(name='var')
    #         .set_index('elevation_band')
    #     )
    #     return var_by_band

    # # Public methods for time series analysis
    # def calc_t_swe_max(self, swe_t):
    #     """Public method to calculate time of max SWE"""
    #     return self._calc_t_swe_max(swe_t)

    # def calc_swe_max(self, swe_t):
    #     """Public method to calculate max SWE"""
    #     return self._calc_swe_max(swe_t)

    # def calc_sws(self, swe_t):
    #     """Public method to calculate snow water storage"""
    #     return self._calc_sws(swe_t)

    # def calc_melt_rate(self, swe_t):
    #     """Public method to calculate melt rate"""
    #     return self._calc_melt_rate(swe_t)

    # def calc_7_day_melt(self, swe_t):
    #     """Public method to calculate 7-day melt"""
    #     return self._calc_7_day_melt(swe_t)

    # def calc_melt_sum(self, swe_t):
    #     """Public method to calculate melt sum"""
    #     return self._calc_melt_sum(swe_t)

    # def calc_sf_sum(self, swe_t):
    #     """Public method to calculate snowfall sum"""
    #     return self._calc_sf_sum(swe_t)

    # def calc_swe_dates(self, swe_t, smooth_window=5, threshold_frac=0.1):
    #     """Public method to calculate SWE dates"""
    #     return self._calc_swe_dates(swe_t, smooth_window, threshold_frac)

    # # Public methods for spatial analysis
    # def calc_t_swe_max_grid(self, swe3d):
    #     """Calculate time of max SWE for each grid cell"""
    #     return xr.apply_ufunc(self._calc_t_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)

    # def calc_swe_max_grid(self, swe3d):
    #     """Calculate max SWE for each grid cell"""
    #     return xr.apply_ufunc(self._calc_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)

    # def calc_sws_grid(self, swe3d):
    #     """Calculate snow water storage for each grid cell"""
    #     return xr.apply_ufunc(self._calc_sws, swe3d, input_core_dims=[['time']], vectorize=True)

    # def calc_melt_rate_grid(self, swe3d):
    #     """Calculate melt rate for each grid cell"""
    #     return xr.apply_ufunc(self._calc_melt_rate, swe3d, input_core_dims=[['time']], vectorize=True)

    # def calc_7_day_melt_grid(self, swe3d):
    #     """Calculate 7-day melt for each grid cell"""
    #     return xr.apply_ufunc(self._calc_7_day_melt, swe3d, input_core_dims=[['time']], vectorize=True)

    # def calc_melt_sum_grid(self, swe3d):
    #     """Calculate melt sum for each grid cell"""
    #     melt_sum_grid = xr.apply_ufunc(self._calc_melt_sum, swe3d, input_core_dims=[['time']], vectorize=True)
    #     melt_sum_grid = xr.where(swe3d.isnull().all(dim='time'), np.nan, melt_sum_grid)
    #     return melt_sum_grid

    # # Public methods for catchment analysis
    # def calc_t_swe_max_catchment(self, swe3d):
    #     """Calculate time of max SWE for catchment"""
    #     swe_t = self._calc_sum2d(swe3d)
    #     return self._calc_t_swe_max(swe_t)

    # def calc_swe_max_catchment(self, swe3d):
    #     """Calculate max SWE for catchment"""
    #     swe_t = self._calc_sum2d(swe3d)
    #     return self._calc_swe_max(swe_t)

    # def calc_sws_catchment(self, swe3d):
    #     """Calculate snow water storage for catchment"""
    #     swe_t = self._calc_sum2d(swe3d)
    #     return self._calc_sws(swe_t)

    # def calc_melt_rate_catchment(self, swe3d):
    #     """Calculate melt rate for catchment"""
    #     swe_t = self._calc_sum2d(swe3d)
    #     return self._calc_melt_rate(swe_t)

    # def calc_7_day_melt_catchment(self, swe3d):
    #     """Calculate 7-day melt for catchment"""
    #     swe_t = self._calc_sum2d(swe3d)
    #     return self._calc_7_day_melt(swe_t)

    # def calc_melt_sum_catchment(self, swe3d):
    #     """Calculate melt sum for catchment"""
    #     swe_t = self._calc_sum2d(swe3d)
    #     return self._calc_melt_sum(swe_t)

    # # Public methods for elevation-based analysis
    # def calc_t_swe_max_vs_elevation(self, swe3d):
    #     """Calculate time of max SWE vs elevation"""
    #     t_swe_max = self.calc_t_swe_max_grid(swe3d)
    #     return self._calc_var_vs_elevation(t_swe_max)

    # def calc_swe_max_vs_elevation(self, swe3d):
    #     """Calculate max SWE vs elevation"""
    #     SWE_max = self.calc_swe_max_grid(swe3d)
    #     return self._calc_var_vs_elevation(SWE_max)

    # def calc_sws_vs_elevation(self, swe3d):
    #     """Calculate snow water storage vs elevation"""
    #     sws = self.calc_sws_grid(swe3d)
    #     return self._calc_var_vs_elevation(sws)

    # def calc_melt_rate_vs_elevation(self, swe3d):
    #     """Calculate melt rate vs elevation"""
    #     melt_rate = self.calc_melt_rate_grid(swe3d)
    #     return self._calc_var_vs_elevation(melt_rate)

    # def calc_7_day_melt_vs_elevation(self, swe3d):
    #     """Calculate 7-day melt vs elevation"""
    #     melt7 = self.calc_7_day_melt_grid(swe3d)
    #     return self._calc_var_vs_elevation(melt7)

    # def calc_melt_sum_vs_elevation(self, swe3d):
    #     """Calculate melt sum vs elevation"""
    #     melt_sum = self.calc_melt_sum_grid(swe3d)
    #     melt_sum = xr.where(melt_sum > 0, melt_sum, np.nan)
    #     return self._calc_var_vs_elevation(melt_sum)

    # # Performance metrics methods
    # def calc_melt_kge(self, swe_obs_t, swe_sim_t):
    #     """Calculate KGE for melt rates"""
    #     dif_obs = np.diff(swe_obs_t)
    #     dif_sim = np.diff(swe_sim_t)
    #     neg_dif_obs = np.where(dif_obs < 0, dif_obs * (-1), 0)
    #     neg_dif_sim = np.where(dif_sim < 0, dif_sim * (-1), 0)
    #     return he.kge_2009(neg_dif_obs, neg_dif_sim)

    # def calc_melt_nse(self, swe_obs_t, swe_sim_t):
    #     """Calculate NSE for melt rates"""
    #     dif_obs = np.diff(swe_obs_t)
    #     dif_sim = np.diff(swe_sim_t)
    #     neg_dif_obs = np.where(dif_obs < 0, dif_obs * (-1), 0)
    #     neg_dif_sim = np.where(dif_sim < 0, dif_sim * (-1), 0)
    #     return he.nse(neg_dif_obs, neg_dif_sim)

    # def calc_snowfall_nse(self, swe_obs_t, swe_sim_t):
    #     """Calculate NSE for snowfall"""
    #     dif_obs = np.diff(swe_obs_t)
    #     dif_sim = np.diff(swe_sim_t)
    #     pos_dif_obs = dif_obs[dif_obs > 0]
    #     pos_dif_sim = dif_sim[dif_sim > 0]
    #     if len(pos_dif_obs) == 0 or len(pos_dif_sim) == 0:
    #         return np.nan
    #     return he.nse(pos_dif_obs, pos_dif_sim)

    # def calc_swe_nse(self, swe_obs_t, swe_sim_t):
    #     """Calculate NSE for SWE"""
    #     return he.nse(swe_obs_t, swe_sim_t)

    # # Spatial performance metrics
    # def calc_melt_kge_grid(self, swe_obs, swe_sim):
    #     """Calculate KGE for melt rates per grid cell"""
    #     return xr.apply_ufunc(self.calc_melt_kge, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True)

    # def calc_melt_nse_grid(self, swe_obs, swe_sim):
    #     """Calculate NSE for melt rates per grid cell"""
    #     return xr.apply_ufunc(self.calc_melt_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True)

    # def calc_snowfall_nse_grid(self, swe_obs, swe_sim):
    #     """Calculate NSE for snowfall per grid cell"""
    #     return xr.apply_ufunc(self.calc_snowfall_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True)

    # def calc_swe_nse_grid(self, swe_obs, swe_sim):
    #     """Calculate NSE for SWE per grid cell"""
    #     return xr.apply_ufunc(self.calc_swe_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True)

    # def calc_melt_kge_catchment(self, swe_obs, swe_sim):
    #     """Calculate KGE for melt rates for catchment"""
    #     swe_obs_t = self._calc_sum2d(swe_obs).to_pandas()
    #     swe_sim_t = self._calc_sum2d(swe_sim).to_pandas()
    #     return self.calc_melt_kge(swe_obs_t, swe_sim_t)

    # def calc_melt_nse_catchment(self, swe_obs, swe_sim):
    #     """Calculate NSE for melt rates for catchment"""
    #     swe_obs_t = self._calc_sum2d(swe_obs).to_pandas()
    #     swe_sim_t = self._calc_sum2d(swe_sim).to_pandas()
    #     return self.calc_melt_nse(swe_obs_t, swe_sim_t)

    # def calc_snowfall_nse_catchment(self, swe_obs, swe_sim):
    #     """Calculate NSE for snowfall for catchment"""
    #     swe_obs_t = self._calc_sum2d(swe_obs).to_pandas()
    #     swe_sim_t = self._calc_sum2d(swe_sim).to_pandas()
    #     return self.calc_snowfall_nse(swe_obs_t, swe_sim_t)

    # def calc_swe_nse_catchment(self, swe_obs, swe_sim):
    #     """Calculate NSE for SWE for catchment"""
    #     swe_obs_t = self._calc_sum2d(swe_obs).to_pandas()
    #     swe_sim_t = self._calc_sum2d(swe_sim).to_pandas()
    #     return self.calc_swe_nse(swe_obs_t, swe_sim_t)

    # def calc_melt_kge_vs_elevation(self, swe_obs, swe_sim):
    #     """Calculate KGE for melt rates vs elevation"""
    #     kges = self.calc_melt_kge_grid(swe_obs, swe_sim)
    #     return self._calc_var_vs_elevation(kges)

    # def calc_melt_nse_vs_elevation(self, swe_obs, swe_sim):
    #     """Calculate NSE for melt rates vs elevation"""
    #     nses = self.calc_melt_nse_grid(swe_obs, swe_sim)
    #     return self._calc_var_vs_elevation(nses)

    # def calc_snowfall_nse_vs_elevation(self, swe_obs, swe_sim):
    #     """Calculate NSE for snowfall vs elevation"""
    #     nses = self.calc_snowfall_nse_grid(swe_obs, swe_sim)
    #     return self._calc_var_vs_elevation(nses)

    # def calc_swe_nse_vs_elevation(self, swe_obs, swe_sim):
    #     """Calculate NSE for SWE vs elevation"""
    #     nses = self.calc_swe_nse_grid(swe_obs, swe_sim)
    #     return self._calc_var_vs_elevation(nses)

    # def calculate_spaef(self, swe_obs, swe_sim):
    #     """
    #     Calculate the SPAEF metric for assessing spatial patterns of SWE.
        
    #     Args:
    #         swe_obs (xr.DataArray): Observed 3D SWE data
    #         swe_sim (xr.DataArray): Simulated 3D SWE data
        
    #     Returns:
    #         float: SPAEF value
    #     """
    #     obs = self.calc_melt_sum_grid(swe_obs)
    #     sim = self.calc_melt_sum_grid(swe_sim)

    #     # Flatten the data and remove NaNs
    #     obs_flat = obs.values.flatten()
    #     sim_flat = sim.values.flatten()
    #     mask = ~np.isnan(obs_flat) & ~np.isnan(sim_flat)
    #     obs_flat = obs_flat[mask]
    #     sim_flat = sim_flat[mask]

    #     # Calculate A: Pearson correlation coefficient
    #     A = pearsonr(obs_flat, sim_flat)[0]

    #     # Calculate B: Fraction of the coefficient of variation
    #     B = (variation(sim_flat) / variation(obs_flat))
        
    #     # Calculate C: Histogram intersection
    #     obs_hist, bin_edges = np.histogram(obs_flat, bins='auto', density=True)
    #     sim_hist, _ = np.histogram(sim_flat, bins=len(obs_hist), density=True)
    #     C = np.sum(np.minimum(obs_hist, sim_hist)) / np.sum(obs_hist)

    #     # Calculate SPAEF
    #     SPAEF = 1 - np.sqrt((A - 1)**2 + (B - 1)**2 + (C - 1)**2)
    #     return SPAEF 