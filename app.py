from flask import Flask, render_template, request, send_file
import os
import pandas as pd
from modules.eda_pipeline import auto_eda_pipeline
from modules.model_pipeline import train_best_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # FIX: Avoid UnicodeDecodeError
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin1')

        # Run universal EDA
        clean_df, eda_summary = auto_eda_pipeline(df)

        # Save cleaned data
        clean_path = os.path.join(OUTPUT_FOLDER, "cleaned_data.csv")
        clean_df.to_csv(clean_path, index=False)

        # Train best model (auto-detects classification/regression)
        best_model, report = train_best_model(clean_df)

        return render_template('result.html', report=report, clean_path=clean_path)

@app.route('/download')
def download():
    return send_file(os.path.join(OUTPUT_FOLDER, 'cleaned_data.csv'), as_attachment=True)

@app.route('/download_pdf')
def download_pdf():
    return send_file("outputs/eda_report.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
