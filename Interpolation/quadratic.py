import pandas as pd
import math
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

def interpolate_files_quadratic(uniform_folder_path, nonuniform_folder_path, output_folder_path):
    r2_scores = []
    mean_square_errors = []
    mean_absolute_errors = []
    snr = []
    names = []

    t1 = glob.glob(uniform_folder_path + "*.csv")
    for file in t1:
        filename = os.path.basename(file).replace(".csv", "")
        df1 = pd.read_csv(file)
        vinn = df1['vinn Y']
        df1.columns.values[0] = "Sim_time"
        df1 = df1.loc[:, ~df1.columns.str.endswith('X')]
        df1 = df1.loc[:, ~df1.columns.str.startswith('net')]

    t1 = glob.glob(nonuniform_folder_path + "*.csv")
    for file in t1:
        filename = os.path.basename(file).replace(".csv", "")
        df = pd.read_csv(file)
        df.columns.values[0] = "Sim_time"
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
            interp_func = interp1d(non_time, conv, kind='quadratic')
            interp_column = interp_func(uniform_time)
            final[i] = interp_column
        final.to_csv(os.path.join(output_folder_path, filename + ".csv"))

        resampled_signal = final['vinn Y']
        new_data = pd.DataFrame()
        new_data['resampled_data_python'] = resampled_signal
        new_data['uniform_data_simulator'] = vinn
        new_data['Time'] = uniform_time
        new_data["diff"] = new_data["uniform_data_simulator"] - new_data["resampled_data_python"]
        length_of_rows = len(new_data)
        n = length_of_rows - 1
        upper = new_data['uniform_data_simulator'].pow(2).sum() / n
        lower = new_data['diff'].pow(2).sum() / n
        snr_value = 10 * math.log10(upper / lower)
        predvinn = new_data['resampled_data_python']
        vinn = new_data['uniform_data_simulator']
        mae = mean_absolute_error(predvinn, vinn)
        mse = mean_squared_error(predvinn, vinn)
        r2 = r2_score(predvinn, vinn)

        r2_scores.append(r2)
        mean_square_errors.append(mse)
        mean_absolute_errors.append(mae)
        snr.append(snr_value)
        names.append(filename)

        # Plot original and resampled signal
        fig = plt.figure(figsize=(10, 5), facecolor='w', edgecolor='k')
        plt.plot(uniform_time, resampled_signal, color="blue", linewidth=3, label='Resampled Signal')
        plt.plot(uniform_time, vinn, color="red", linewidth=3, label='Original Signal')
        plt.title('Signal Comparison Resampled vs Original')
        plt.xlabel('Time (nanoseconds)')
        plt.ylabel('vinn')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_path, "Graphs", filename + ".png"))

    metrics_df = pd.DataFrame({
        'Filename': names,
        'R2 Score': r2_scores,
        'Mean Square Error': mean_square_errors,
        'Mean Absolute Error': mean_absolute_errors,
        'SNR': snr
    })

    metrics_df.to_csv(os.path.join(output_folder_path, "metrics.csv"), index=False)

if __name__ == "__main__":
    interpolate_files_quadratic("D:\\Interpolationnnn\\uniform_all\\",
                 "D:\\Interpolationnnn\\non_uniform_all\\",
                 "D:\\Interpolationnnn\\interpolated_files_quad\\")
