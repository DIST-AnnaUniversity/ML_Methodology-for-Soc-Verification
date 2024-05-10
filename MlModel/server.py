from flask import Flask, render_template, request
from rf import preprocess, RandomForest  # Assuming these classes are defined in rf.py
from rnn import preprocess, rnn_new
from lstm import preprocess, lstm_new
from mlp import MLP, preprocess
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
            elif ml_technique == 'lstm':
                prs = preprocess()
                rfs = lstm_new()
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
            elif ml_technique == 'rnn':
                prs = preprocess()
                rfs = rnn_new()
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
            elif ml_technique == 'mlp':
                prs = preprocess()
                rfs = MLP()
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
    app.run(port=7897)
