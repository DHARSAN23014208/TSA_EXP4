# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 11.05.2026
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Load dataset
data = pd.read_csv("FINAL_USO.csv")
data.head()
# Declare required variables and set figure size, and visualise the data
N = 1000
plt.rcParams["figure.figsize"] = [
    12,
    6,
]  # plt.rcParams is a dictionary-like object in Matplotlib
X = data["Close"]
plt.plot(X)
plt.title("Original Data - Gold Price (USO Close)")
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X) // 4, ax=plt.gca())
plt.title("Original Data ACF")
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X) // 4, ax=plt.gca())
plt.title("Original Data PACF")
plt.tight_layout()
plt.show()
# Fitting the ARMA(1,1) model and deriving parameters
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params["ar.L1"]
theta1_arma11 = arma11_model.params["ma.L1"]
# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title("Simulated ARMA(1,1) Process")
plt.xlim([0, 500])
plt.show()
# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
# Fitting the ARMA(2,2) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params["ar.L1"]
phi2_arma22 = arma22_model.params["ar.L2"]
theta1_arma22 = arma22_model.params["ma.L1"]
theta2_arma22 = arma22_model.params["ma.L2"]
# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)
plt.plot(ARMA_2)
plt.title("Simulated ARMA(2,2) Process")
plt.xlim([0, 500])
plt.show()
# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```
OUTPUT:
<img width="986" height="528" alt="image" src="https://github.com/user-attachments/assets/283834da-eb02-49e6-a7df-64a3e1eadb4c" />
<img width="1198" height="590" alt="image" src="https://github.com/user-attachments/assets/a984f224-a5c2-4c0c-81c5-41d413c28b6d" />
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/fde36e90-0179-47bb-95c5-5229b632e85b" />
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/9c00d1aa-15f7-41ee-8955-2dd4833edab7" />
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/343f0fce-f793-4c3e-9601-755e4b3043d3" />
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/fb171e7d-293c-4ea3-a74a-3c25386f90a5" />
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/bffbb889-4b66-4d11-b4ff-7bd48c95cec0" />



RESULT:
Thus, a python program is created to fir ARMA Model successfully.
