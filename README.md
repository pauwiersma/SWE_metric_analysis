# SWE Metric Analysis

This repository contains code to analyze the correlation between residual-based and signature-based metrics, using observed (SNOTEL) and simulated (ERA5,NHSWE,TI) Snow Water Equivalent (SWE) products. The analysis calculates various metrics to evaluate the performance of different SWE products.

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
- ...
