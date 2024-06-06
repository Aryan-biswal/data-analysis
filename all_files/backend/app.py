import matplotlib

matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        analysis_result = analyze_data(df)
        visualization_paths = visualize_data(df)
        return render_template('result.html', tables=[analysis_result.to_html()], visualizations=visualization_paths)


def analyze_data(df):
    summary = df.describe(include='all')
    return summary


def visualize_data(df):
    visualization_paths = []

    # Filter only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_path = os.path.join(app.config['UPLOAD_FOLDER'], 'correlation_matrix.png')
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.savefig(corr_path)
        plt.close()
        visualization_paths.append(corr_path)

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True)
            hist_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{column}_histogram.png')
            plt.savefig(hist_path)
            plt.close()
            visualization_paths.append(hist_path)
        elif pd.api.types.is_string_dtype(df[column]):
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[column], order=df[column].value_counts().index)
            count_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{column}_countplot.png')
            plt.savefig(count_path)
            plt.close()
            visualization_paths.append(count_path)

    return visualization_paths


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
