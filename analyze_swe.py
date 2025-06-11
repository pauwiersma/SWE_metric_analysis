import pandas as pd
import numpy as np
from pathlib import Path
import os
from calc_swe_metrics import SWEMetrics
import matplotlib.pyplot as plt
import seaborn as sns

def load_dates():
    """Load and process dates from the raw data."""
    dates = pd.read_csv("Data/raw/Dates.csv", parse_dates=[0], header=None)
    dates = dates.rename(columns={0: 'Date'})
    dates['Year'] = dates['Date'].dt.year 
    dates.loc[dates['Date'].dt.month >= 10, 'Year'] += 1
    return dates

def load_observations(dates):
    """Load observation data from SNOTEL and align with dates."""
    obs_df = pd.read_csv("Data/raw/SNOTEL.csv", header=None)
    obs_df.index = dates['Date']
    obs_df['Year'] = dates['Year'].values
    return obs_df

def load_simulation(product, dates, obs_df):
    """Load simulation data for a specific product and align with observations."""
    sim_df = pd.read_csv(f"Data/raw/{product}.csv", header=None)
    sim_df.index = dates['Date']
    sim_df['Year'] = dates['Year'].values
    # Only keep data where we have observations
    sim_df = sim_df.where(obs_df.notna())
    return sim_df

def calculate_metrics(obs_df, sim_df, metrics,metric_names):
    """Calculate NSE, RMSE, and KGE metrics for each station and year."""
    results = {name: pd.DataFrame(index=obs_df['Year'].unique(), columns=obs_df.columns.drop(['Year'])) for name in metric_names}
    
    for year in sorted(obs_df['Year'].unique()):
        year_obs = obs_df[obs_df['Year'] == year]
        year_sim = sim_df[sim_df['Year'] == year]
        
        for station in obs_df.columns.drop(['Year']):
            obs_values = year_obs[station].values
            sim_values = year_sim[station].values

            if np.isnan(obs_values).all() or np.isnan(sim_values).all():
                continue
            if np.nansum(obs_values) == 0 and np.nansum(sim_values) > 0:
                continue
            if np.nansum(sim_values) == 0 and np.nansum(obs_values) > 0:
                print(f"Only zeros in sim for {station} in {year}")
                continue
            if np.nansum(obs_values) == 0 and np.nansum(sim_values) == 0:
                print(f"Only zeros in both for {station} in {year}")
                continue
            #if the nans of both products don't overlap, skip the station
            if np.isnan(obs_values).sum() != np.isnan(sim_values).sum():
                print(f"Different number of nans for {station} in {year}")
                continue
            #if the SWEsum of either is lower than 100, skip the station
            if np.nansum(obs_values) < 10000 or np.nansum(sim_values) < 10000:
                print(f"Low SWE sum for {station} in {year}: obs={np.nansum(obs_values)}, sim={np.nansum(sim_values)}")
                continue


            for metric_name in metric_names: 
                #residual metrics
                if metric_name == 'NSE':
                    results['NSE'].loc[year, station] = metrics.calc_swe_nse(obs_values, sim_values)
                elif metric_name == 'RMSE':
                    results['RMSE'].loc[year, station] = metrics.calc_swe_rmse(obs_values, sim_values)
                elif metric_name == 'KGE_2009':
                    results['KGE_2009'].loc[year, station] = metrics.calc_swe_kge(obs_values, sim_values)
                #signature metrics
                elif metric_name == 'peakSWE':
                    obs_peakSWE = metrics.calc_swe_max(obs_values)
                    sim_peakSWE = metrics.calc_swe_max(sim_values)
                    results['peakSWE'].loc[year, station] = np.abs(obs_peakSWE - sim_peakSWE)
                elif metric_name == 'melt_NSE':
                    results['melt_NSE'].loc[year, station] = metrics.calc_melt_nse(obs_values, sim_values)
                elif metric_name == 'SWE_appearance':
                    obs_SWE_appearance = metrics.calc_swe_start(obs_values)
                    sim_SWE_appearance = metrics.calc_swe_start(sim_values)
                    results['SWE_appearance'].loc[year, station] = np.abs(obs_SWE_appearance - sim_SWE_appearance)
                elif metric_name == 'SWE_disappearance':
                    obs_SWE_disappearance = metrics.calc_swe_end(obs_values)
                    sim_SWE_disappearance = metrics.calc_swe_end(sim_values)
                    results['SWE_disappearance'].loc[year, station] = np.abs(obs_SWE_disappearance - sim_SWE_disappearance)
                elif metric_name == 'peakSWE_date':
                    obs_peakSWE_date = metrics.calc_t_swe_max(obs_values)
                    sim_peakSWE_date = metrics.calc_t_swe_max(sim_values)
                    results['peakSWE_date'].loc[year, station] = np.abs(obs_peakSWE_date - sim_peakSWE_date)
                elif metric_name == 'snowfall_NSE':
                    results['snowfall_NSE'].loc[year, station] = metrics.calc_snowfall_nse(obs_values, sim_values)
    return results

def save_metrics(results, product, processed_dir):
    """Save calculated metrics to CSV files."""
    for metric_name, metric_df in results.items():
        metric_df.to_csv(processed_dir / f"{product}_{metric_name}.csv")
        print(f"Saved {product} {metric_name} to {processed_dir}/{product}_{metric_name}.csv")

def load_existing_metrics(product, metric_name, processed_dir):
    """Load previously calculated metrics from CSV files."""
    metric_path = processed_dir / f"{product}_{metric_name}.csv"
    if metric_path.exists():
        return pd.read_csv(metric_path, index_col=0)
    return None

def process_swe_data(metric_names):
    """Main function to process SWE data and calculate metrics."""
    # Create processed directory if it doesn't exist
    processed_dir = Path("Data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dates and observations
    dates = load_dates()
    dates.to_csv("Data/processed/Dates.csv")
    obs_df = load_observations(dates)
    
    # Initialize SWEMetrics class
    metrics = SWEMetrics()
    
    # List of simulation products
    sim_products = ['ERA5', 'NHSWE', 'TI']
    
    results = {}

    for product in sim_products:
        print(f"Processing {product}...")
        sim_df = load_simulation(product, dates, obs_df)
        results[product] = calculate_metrics(obs_df, sim_df, metrics,metric_names)
        save_metrics(results[product], product, processed_dir)
    return results

if __name__ == "__main__":
    signature_metrics = ['peakSWE','melt_NSE','SWE_appearance',
                         'SWE_disappearance','peakSWE_date','snowfall_NSE']
    residual_metrics = ['NSE', 'RMSE', 'KGE_2009']
    all_metrics = signature_metrics + residual_metrics

    results = process_swe_data(all_metrics)

    #boxplots per metric 
    for metric in ['NSE', 'RMSE', 'KGE_2009','peakSWE']:
        #make melted df
        melted_df_concat = pd.DataFrame()
        for product in results.keys():
            melted_df = results[product][metric].reset_index()
            melted_df = melted_df.melt(id_vars='index', var_name='Station', value_name='Value')
            melted_df['Product'] = product
            melted_df.rename(columns={'index': 'Year'}, inplace=True)
            melted_df_concat = pd.concat([melted_df_concat, melted_df])
        melted_df_concat = melted_df_concat.reset_index()
        #remove all years before 1980
        melted_df_concat = melted_df_concat[melted_df_concat['Year'] >= 1990]
        plt.figure(figsize=(6,6))
        sns.stripplot(x='Value',  y='Product',hue = 'Year', 
                      data=melted_df_concat,dodge =True,orient = 'h',
                      alpha = 0.1,palette = 'viridis')
        plt.title(f'SWE Metrics - {metric}')
        plt.ylabel('Product')
        plt.xlabel('Metric Value')
        plt.xticks(rotation=45)
        #set xlims to 5 and 95 percentile of the data
        # plt.xlim(melted_df_concat['Value'].quantile(0.05), melted_df_concat['Value'].quantile(0.95))
        plt.xlim(-1,1)

    f1,ax1 = plt.subplots(figsize=(10,10))
    sns.stripplot(x='Value', y='Year', hue='Station',data= melted_df,orient = 'h', ax = ax1)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(72.5,40)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)

    def plot_metric_comparison(results, metric1, metric2):
        """
        Create scatter plots comparing two metrics for each product in results.

        Parameters:
        - results: dict, containing metric data for each product.
        - metric1: str, name of the first metric.
        - metric2: str, name of the second metric.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
        
        for ax, product in zip(axes, results.keys()):
            metric1_df = results[product][metric1]
            metric2_df = results[product][metric2]

            # Remove indices where either metric1_df or metric2_df is NaN
            valid_indices = ~metric1_df.isna() & ~metric2_df.isna()
            metric1_values = metric1_df[valid_indices].values.flatten()
            metric2_values = metric2_df[valid_indices].values.flatten()

            # Create DataFrame for plotting
            joint_df = pd.DataFrame({'metric1': metric1_values, 'metric2': metric2_values})
            joint_df = joint_df.dropna()  # Drop any rows with NaN values
            # Rank metrics, descending for 'NSE' or 'KGE', ascending otherwise
            joint_df['metric1'] = joint_df['metric1'].rank(
                pct=False, method = 'min',
                ascending=False if 'NSE' in metric1 or 'KGE' in metric1 else True
            )
            joint_df['metric2'] = joint_df['metric2'].rank(
                pct=False, method = 'min',
                ascending=False if 'NSE' in metric2 or 'KGE' in metric2 else True
            )
            
            sns.scatterplot(x='metric1', y='metric2', data=joint_df, color='black', alpha=0.1, size=0.1, ax=ax)
            ax.set_title(f'{product}')  # Only the product name in the subplot title
            ax.set_xlabel(f"{metric1} Rank")
            ax.set_ylabel(f"{metric2} Rank")
        
        # Add a suptitle explaining the plot
        fig.suptitle(f"Comparison of {metric1} and {metric2} Across Products\nEach point represents one year x catchment", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit suptitle
        plt.show()
    
    # Example usage of the plot_metric_comparison function
    plot_metric_comparison(results, metric1 ='NSE', metric2 ='peakSWE')
    plot_metric_comparison(results, metric1 ='NSE', metric2 ='RMSE')
    plot_metric_comparison(results, metric1 ='NSE', metric2 ='KGE_2009')
    plot_metric_comparison(results, metric1 ='NSE', metric2 ='melt_NSE')
    plot_metric_comparison(results, metric1 ='NSE', metric2 ='snowfall_NSE')
    plot_metric_comparison(results, metric1 ='NSE', metric2 ='SWE_appearance')
    plot_metric_comparison(results,'NSE','SWE_disappearance')
    plot_metric_comparison(results,'peakSWE','RMSE')
    plot_metric_comparison(results,'peakSWE','KGE_2009')
    plot_metric_comparison(results,'peakSWE','peakSWE_date')

