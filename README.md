# SWE Metric Analysis

This repository contains code to analyze and compare different Snow Water Equivalent (SWE) simulation products against SNOTEL observations. The analysis calculates various metrics to evaluate the performance of different SWE products.

## Data Structure

The `Data` directory contains the following files:
- `Dates.csv`: Dates in DD-Month-YYYY format
- `SNOTEL.csv`: SNOTEL station observations
- `ERA5.csv`: ERA5 reanalysis SWE simulations
- `NHSWE.csv`: NHSWE SWE simulations
- `TI.csv`: TI SWE simulations
- `Metadata.csv`: Station metadata (Latitude, Longitude, Elevation)

The processed results are stored in the `Data/processed` directory, with separate files for each simulation product containing yearly metrics.

## Metrics

Currently implemented metrics:
- Nash-Sutcliffe Efficiency (NSE)

## Usage

Run the analysis script:
```bash
python analyze_swe.py
```

This will:
1. Load all SWE data from the Data directory
2. Calculate yearly metrics for each simulation product
3. Save results to Data/processed directory 