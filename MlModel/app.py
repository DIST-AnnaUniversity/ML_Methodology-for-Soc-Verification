from flask import Flask, render_template, request
from rf import preprocess, RandomForest  # Uncomment if these classes are defined in rf.py
import zipfile
import os

app = Flask(__name__)

def unzip_folder(zip_file, extract_to):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        return f"Error extracting folder: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            train_folder = request.files['train_folder']
            test_folder = request.files['test_folder']
            ml_technique = request.form['ml_technique']

            # Extract uploaded folders
            extract_train_folder = 'train_folder'
            extract_test_folder = 'test_folder'
            unzip_folder(train_folder, extract_train_folder)
            unzip_folder(test_folder, extract_test_folder)

            if ml_technique == 'rf':
                print("program start")
                prs = preprocess()
                rfs = RandomForest()
                prs.specs("DWT conversion process")
                prs.dwt_train()
                prs.dwt_test()
                prs.specs("BEFORE ALL PROCESS")
                prs.combine_files()
                prs.specs("AFTER COMBINING FILES")
                prs.fetch()
                prs.specs("AFTER FETCHING DATA")
                rfs.model_creation()
                prs.specs("AFTER MODEL CREATION")
                rfs.prediction()
                prs.specs("AFTER PREDICTION")
                rfs.idwt()
                rfs.graph_plot()
                rfs.metrics()
                return "ML/DL completed successfully!"
            else:
                return "Invalid ML technique selected!"
        except Exception as e:
            return f"Error processing request: {e}"

if __name__ == "__main__":
    app.run(port=7898)



# import pandas as pd
# import math
# import os
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from flask import Flask, render_template, request
# from cubic import interpolate_files as cubic_interpolate
# from linear import interpolate_files_linear as linear_interpolate
# from scipy.interpolate import interp1d, BarycentricInterpolator
# import zipfile

# app = Flask(__name__)

# def unzip_folder(zip_file, extract_to):
#     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)

# # Define routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# def interpolate_files_quadratic(uniform_folder_path, nonuniform_folder_path, output_folder_path):
#     # Function to perform quadratic interpolation
#     pass

# def interpolate_files_barycentric(uniform_folder_path, nonuniform_folder_path, output_folder_path):
#     # Function to perform barycentric interpolation
#     pass

# def interpolate_files_nearest(uniform_folder_path, nonuniform_folder_path, output_folder_path):
#     # Function to perform nearest neighbor interpolation
#     pass

# @app.route('/upload', methods=['POST'])
# def upload():
#     if request.method == 'POST':
#         uniform_folder = request.files['uniform_folder']
#         non_uniform_folder = request.files['non_uniform_folder']
#         interpolation_technique = request.form['interpolation_technique']
        
#         # Extract uploaded folders
#         unzip_folder(uniform_folder, 'uniform_folder')
#         unzip_folder(non_uniform_folder, 'non_uniform_folder')

#         if interpolation_technique == 'cubic':
#             interpolate_files('uniform_all\\', 'non_uniform_all\\', "interpolated_files\\")
#         elif interpolation_technique == 'linear':
#             interpolate_files_linear('uniform_all\\', 'non_uniform_all\\', "interpolated_files\\")
#         elif interpolation_technique == 'quadratic':
#             interpolate_files_quadratic('uniform_all\\', 'non_uniform_all\\', "interpolated_files\\")
#         elif interpolation_technique == 'barycentric':
#             interpolate_files_barycentric('uniform_all\\', 'non_uniform_all\\', "interpolated_files\\")
#         elif interpolation_technique == 'nearest':
#             interpolate_files_nearest('uniform_all\\', 'non_uniform_all\\', "interpolated_files\\")
        
#         return "Interpolation/Extrapolation completed successfully!"

# if __name__ == "__main__":
#     app.run(port=7894)






# # from flask import Flask, render_template, request
# # from cubic import interpolate_files
# # import zipfile
# # import os


# # app = Flask(__name__)

# # def unzip_folder(zip_file, extract_to):
# #     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
# #         zip_ref.extractall(extract_to)


# # # Define routes
# # @app.route('/')
# # def index():
# #     return render_template('index.html')



# # @app.route('/upload', methods=['POST'])
# # def upload():
# #     if request.method == 'POST':
# #         uniform_folder = request.files['uniform_folder']
# #         non_uniform_folder = request.files['non_uniform_folder']
# #         interpolation_technique = request.form['interpolation_technique']
# #         print(uniform_folder)
# #         print(non_uniform_folder)
# #         print(interpolation_technique)
# #         uni = unzip_folder(uniform_folder,'uniform_folder')
# #         non_uni=unzip_folder(non_uniform_folder,'non_uniform_folder')
# #         print(uni)
# #         print(non_uni)
        
# #         interpolate_files('uniform_all\\','non_uniform_all\\',"interpolated_files\\")
# #         # interpolate_files("D:\\CubicSplineInterpolationnn\\uniform_all\\",
# #         #          "D:\\CubicSplineInterpolationnn\\non_uniform_all\\",
# #         #          "D:\\CubicSplineInterpolationnn\\interpolated_files\\")
# #         # Handle the uploaded files and selected interpolation technique
# #         # Perform interpolation/extrapolation
        
# #         return "Interpolation/Extrapolation completed successfully!"

# # @app.route('/upload_interpolation_files', methods=['POST'])
# # def upload_interpolation_files():
# #     if request.method == 'POST':
# #         interpolation_files = request.files.getlist('interpolation_files')
# #         # Handle uploaded interpolation files
# #         # Process files using selected ML/DL model
        
# #         return "Files uploaded and processed successfully!"

# # if __name__ == "__main__":
# #     app.run(port=7894)
