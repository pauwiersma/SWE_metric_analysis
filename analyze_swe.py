import pandas as pd
import numpy as np
from pathlib import Path
import os
from calc_swe_metrics import SWEMetrics

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

def calculate_metrics(obs_df, sim_df, metrics):
    """Calculate NSE, RMSE, and KGE metrics for each station and year."""
    results = {
        'NSE': pd.DataFrame(index=obs_df['Year'].unique(), columns=obs_df.columns.drop(['Year'])),
        'RMSE': pd.DataFrame(index=obs_df['Year'].unique(), columns=obs_df.columns.drop(['Year'])),
        'KGE_2009': pd.DataFrame(index=obs_df['Year'].unique(), columns=obs_df.columns.drop(['Year']))
    }
    
    for year in sorted(obs_df['Year'].unique()):
        year_obs = obs_df[obs_df['Year'] == year]
        year_sim = sim_df[sim_df['Year'] == year]
        
        for station in obs_df.columns.drop(['Year']):
            obs_values = year_obs[station].values
            sim_values = year_sim[station].values

            if np.isnan(obs_values).all() or np.isnan(sim_values).all():
                continue
            if np.sum(obs_values) == 0 and np.sum(sim_values) > 0:
                continue
            if np.sum(sim_values) == 0 and np.sum(obs_values) > 0:
                print(f"Only zeros in sim for {station} in {year}")
                continue

            results['NSE'].loc[year, station] = metrics.calc_swe_nse(obs_values, sim_values)
            results['RMSE'].loc[year, station] = metrics.calc_swe_rmse(obs_values, sim_values)
            results['KGE_2009'].loc[year, station] = metrics.calc_swe_kge(obs_values, sim_values)
    
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

def process_swe_data():
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
    
    for product in sim_products:
        print(f"Processing {product}...")
        sim_df = load_simulation(product, dates, obs_df)
        results = calculate_metrics(obs_df, sim_df, metrics)
        save_metrics(results, product, processed_dir)

if __name__ == "__main__":
    process_swe_data() 