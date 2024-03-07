import pandas as pd
import math
import os
import glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Define empty lists to store metrics values
r2_scores = []
mean_square_errors = []
mean_absolute_errors = []
snr = []
name=[]

file_path = "C:\\Users\\neera\\Pictures\\Interpolation\\uniform_all\\"
t1 = glob.glob(file_path + "*.csv")
for file in t1:
    filename = file.replace(file_path, "")  # read file from path into individual files
    filename1 = filename.replace(".csv", "")
    df1 = pd.read_csv(file)
    vinn = df1['vinn Y']
    non_length = len(df1)
    df1.columns.values[0] = "Sim_time"
    df1 = df1.loc[:, ~df1.columns.str.endswith('X')]
    df1 = df1.loc[:, ~df1.columns.str.startswith('net')]

file_path = "C:\\Users\\neera\\Pictures\\Interpolation\\non_uniform_all\\"
t1 = glob.glob(file_path + "*.csv")
for file in t1:
    filename = file.replace(file_path, "")  # read file from path into individual files
    filename1 = filename.replace(".csv", "")
    df = pd.read_csv(file)
    non_length = len(df)
    df.columns.values[0] = "Sim_time"
    # preprocessing: Removing Duplicates X and net Columns
    df = df.loc[:, ~df.columns.str.endswith('X')]
    df = df.loc[:, ~df.columns.str.startswith('net')]
    time = df["Sim_time"]
    df.drop('Sim_time', axis=1)
    final = pd.DataFrame()
    non_time = time.to_numpy()
    uniform_time = np.arange(non_time.min(), non_time.max(), 1e-9)
    final["Sim_time"] = uniform_time
    for i in df.columns:
        conv = df[i].to_numpy()
        interp_func = CubicSpline(non_time, conv)
        interp_column = interp_func(uniform_time)
        final[i] = interp_column
        length = len(interp_column)
    final.to_csv("C:\\Users\\neera\\Pictures\\Interpolation\\interpolated_files\\"+filename1+".csv")
    resampled_signal = final['vinn Y']
    new_data = pd.DataFrame()
    new_data['resampled_data_python'] = resampled_signal
    new_data['uniform_data_simulator'] = vinn
    new_data['Time'] = uniform_time
    new_data["diff"] = new_data["uniform_data_simulator"] - new_data["resampled_data_python"]
    length_of_rows = len(new_data)
    n = length_of_rows - 1
    upper = 0
    lower = 0
    upper = new_data['uniform_data_simulator'].pow(2).sum()
    lower = new_data['diff'].pow(2).sum()
    upper = upper / n
    lower = lower / n
    snrs = 10 * math.log10(upper / lower)
    predvinn = new_data['resampled_data_python']
    vinn = new_data['uniform_data_simulator']
    mae = mean_absolute_error(predvinn, vinn)
    mse = mean_squared_error(predvinn, vinn)
    r2 = r2_score(predvinn, vinn)

    # Append metrics values to lists
    r2_scores.append(r2)
    mean_square_errors.append(mse)
    mean_absolute_errors.append(mae)
    snr.append(snrs)
    name.append(filename1)

 # Step 7: Plot original and resampled signal
    fig = plt.figure(figsize=(10, 5), facecolor='w', edgecolor='k')
    plt.plot(uniform_time, resampled_signal, color="blue", linewidth=3, label='Resampled Signal')
    plt.plot(uniform_time, vinn, color="red", linewidth=3, label='Original Signal')
    plt.title('Signal Comparison Resampled vs Original')
    plt.xlabel('Time (nanoseconds)')
    plt.ylabel('vinn')
    plt.legend()
    plt.tight_layout()
    plt.savefig("C:\\Users\\neera\\Pictures\\Interpolation\\Graphs\\"+filename1+".png")
    # plt.show()
# Create DataFrame from metrics values
metrics_df = pd.DataFrame({
    'Filename': name,
    'R2 Score': r2_scores,
    'Mean Square Error': mean_square_errors,
    'Mean Absolute Error': mean_absolute_errors,
    'SNR': snr
})

# Save DataFrame to CSV
metrics_df.to_csv("C:\\Users\\neera\\Pictures\\Interpolation\\metrics.csv", index=False)


"""
# Plot VINN values comparison
fig = plt.figure(figsize=(10, 5), facecolor='w', edgecolor='k')
plt.plot(vinn, color="blue", linewidth=3, label='Original Uniform Signal VINN')
plt.plot(resampled_signal, color="red", linewidth=3, label='Resampled Signal VINN')   
plt.title('VINN Value Comparison: Uniform Sampled vs Resampled Non-Uniform Data') 
plt.xlabel('VINN (value_count)')
plt.ylabel('VINN Values')
plt.legend()
plt.tight_layout()
plt.savefig("C:/Users/neera/Pictures/Interpolation/non_uniform_all/vinn_value.png")
plt.show()

#no
non_vinn=df['vinn Y']



# Plot original non-uniform signal
fig = plt.figure(figsize=(10, 5), facecolor='w', edgecolor='k')
plt.plot(non_vinn, color="blue", linewidth=3, label='Non-Uniform Sampling (VINN)')
plt.title('Original Non-Uniform Signal')
plt.xlabel('VINN (value_count)')
plt.ylabel('VINN Values')
plt.legend()
plt.tight_layout()
plt.savefig("C:/Users/neera/Pictures/Interpolation/non_uniform_all/non_unf.png")
plt.show()

# Plot resampled non-uniform sampled signal
fig = plt.figure(figsize=(10, 5), facecolor='w', edgecolor='k')
plt.plot(resampled_signal, color="blue", linewidth=3, label='Resampled Uniform Sampling (VINN)')
plt.title('Resampled Non-Uniform Sampled Signal')
plt.xlabel('VINN (value_count)')
plt.ylabel('VINN Values')
plt.legend()
plt.tight_layout()
plt.savefig("C:/Users/neera/Pictures/Interpolation/non_uniform_all/re_unf.png")
plt.show()
"""

