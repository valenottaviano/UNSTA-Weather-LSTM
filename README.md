# UNSTA Weather Data Time Series Analysis with LSTM

## Objective
The objective of this project is to investigate the trend and pattern of time series data from UNSTA Weather Data using Long Short Term Memory (LSTM) networks. Additionally, the project aims to quantify the uncertainty of time series predictions for target variables.

## Tasks and Ideas
- Incorporate the hour of the day as a predictor.
- Explore and analyze the data using LSTM networks.

## Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import random
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
```

## Importing Dataset
The dataset contains meteorological measurements from the University's meteorological station. We focus on the following variables: 'Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed', 'Wind_Chill', 'Rain_Rate', 'Heat_Index', 'Barometer', 'Solar_Rad', and 'UV_Index'. The data is preprocessed and converted into a format suitable for analysis.

## Dataset Information
- Number of Entries: 10,377
- Columns: 22
- Data Types: Float32
- Variables: 'Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed', 'Wind_Chill', 'Rain_Rate', 'Heat_Index', 'Barometer', 'Solar_Rad', 'UV_Index', and month-related dummy variables.

## Functions Definitions
- `scale_data(data)`: Scales the dataset using Min-Max scaling.
- `split_dataset(data)`: Splits the dataset into training and testing sets.
- `to_supervised(train, n_input, n_out)`: Converts data into supervised format for training.
- `build_model_lstm(train, n_input, n_output, epochs, batch_size)`: Builds an LSTM model for deterministic predictions.

## Model 1 - Deterministic LSTM
### Model Parameters
- Input Sequence Length (n_input): 12
- Output Sequence Length (n_output): 12
- Number of Epochs: 100
- Batch Size: 64

### Model 1.1 - Fitting Model
The data is structured for LSTM and the model is trained.

### Model 1.2 - Predicting and Plotting
Predictions are made and compared with actual values, and the results are plotted.

## Model 2 - LSTM with Monte Carlo Dropout (12 Steps)
### Model Parameters
- Input Sequence Length (n_input): 12
- Output Sequence Length (n_output): 12
- Number of Epochs: 100
- Batch Size: 64

### Model 2.1 - Fitting Model
The LSTM model with Monte Carlo Dropout is defined and trained.

### Model 2.2 - Predictions and Plotting (Validation Set)
Predictions are made for the entire validation set, and the results are plotted with confidence intervals.

### Model 2.3 - Predicting and Plotting Series of 7 Days
Predictions are made for a series of 7 days, and the results are plotted along with confidence intervals.

**Note**: For further details, refer to the project code.
